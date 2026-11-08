"""Score frozen checkpoints on the source validation split under explicit inner_steps.

Why this exists: the main-table baseline was validated with ``inner_steps: 0``
(zero-shot) while a candidate arm may be configured with ``inner_steps: 5``
(support adaptation).  ``validation_q95_log_error`` is then not the same
quantity on the two sides, so a gate that compares them is not like-for-like.
This tool re-scores any checkpoint with the training loop's own ``validate()``
under each requested step count, so the comparison can be made at a matched
protocol.  ``--bare-anchor`` scores a freshly constructed struct anchor
(zero-initialised head) instead of a checkpoint, which is the reference for
doc 118 section 6.2.  Nothing here trains, selects or writes a checkpoint, and
every checkpoint's metadata is checked against the read targets before use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.checkpoints import load_checkpoint  # noqa: E402
from spatial_meta_ce.cli import data_settings  # noqa: E402
from spatial_meta_ce.data import StorePool  # noqa: E402
from spatial_meta_ce.model import SetEstimator  # noqa: E402
from spatial_meta_ce.train import validate, validation_plan  # noqa: E402

TARGET_DATABASES = ("imdb", "stats", "genome", "ergastf1")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_or_none(value):
    """Fixed-budget runs store ``best_metric = inf``; keep the report strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def assert_no_target_exposure(state, databases) -> None:
    metadata = state["metadata"]
    if metadata.get("adapted_database"):
        raise ValueError(
            "Checkpoint was adapted on a database and is not an independent "
            f"initialisation: {metadata['adapted_database']!r}"
        )
    exposed = set(metadata["train_databases"]) | set(metadata["validation_databases"])
    overlap = exposed & set(databases)
    if overlap:
        raise ValueError(
            f"Checkpoint exposure overlaps read targets: {sorted(overlap)}"
        )


def assert_plan_matches(state, config, pool, databases) -> None:
    """A score is only comparable if it came from the checkpoint's own plan and caches."""
    signature = state["metadata"].get("training_signature") or {}
    for key in (
        "validation_seed",
        "validation_runs",
        "query_size",
        "support_sizes",
        "inner_lr",
        "adapt_scope",
        "loss",
    ):
        expected = signature.get(key)
        if expected is not None and expected != config[key]:
            raise ValueError(
                f"Checkpoint plan {key}={expected!r} != config {config[key]!r}"
            )
    manifest_hashes = state["metadata"].get("manifest_hashes") or {}
    for database in databases:
        expected = manifest_hashes.get(database)
        actual = pool.get("pretrain", database).manifest["content_hash"]
        if expected is not None and expected != actual:
            raise ValueError(f"Validation cache changed for {database}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training config with the plan")
    parser.add_argument(
        "--checkpoint",
        action="append",
        metavar="NAME=PATH",
        help="checkpoint to score; repeatable",
    )
    parser.add_argument(
        "--bare-anchor",
        action="store_true",
        help="score a fresh struct anchor instead of checkpoints",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for --bare-anchor construction"
    )
    parser.add_argument(
        "--inner-steps", nargs="+", type=int, default=[0, 5], metavar="N"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if any(step < 0 for step in args.inner_steps):
        parser.error("--inner-steps values must be >= 0")
    checkpoints = {}
    for item in args.checkpoint or []:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error(f"--checkpoint wants NAME=PATH, got {item!r}")
        checkpoints[name] = path
    if args.bare_anchor == bool(checkpoints):
        parser.error("Pass either --bare-anchor or one or more --checkpoint, not both")

    config = json.loads(Path(args.config).read_text())
    data = data_settings(config, args.config)
    databases = data["splits"]["validation"]
    overlap = set(databases) & set(TARGET_DATABASES)
    if overlap:
        parser.error(f"Validation split contains targets: {sorted(overlap)}")

    pool = StorePool(data["cache_root"])
    episodes = validation_plan(pool, databases, config)
    report = {
        "config": args.config,
        "plan": {
            "databases": databases,
            "episodes": len(episodes),
            "support_sizes": config["support_sizes"],
            "validation_runs": config["validation_runs"],
            "validation_seed": config["validation_seed"],
            "query_size": config["query_size"],
            "inner_lr": config["inner_lr"],
            "adapt_scope": config["adapt_scope"],
            "loss": config["loss"],
        },
        "checkpoints": {},
    }

    if args.bare_anchor:
        model_config = config["model"]
        if model_config.get("anchor") != "struct" or model_config.get("use_pg"):
            parser.error("--bare-anchor wants a struct-anchored, PG-free config")
        import torch

        torch.manual_seed(args.seed)
        models = [
            (
                "bare-anchor",
                SetEstimator(**model_config).to(args.device),
                {"seed": args.seed, "model_config": model_config},
            )
        ]
    else:
        models = []
        for name, path in checkpoints.items():
            model, state = load_checkpoint(path, args.device)
            assert_no_target_exposure(state, TARGET_DATABASES)
            assert_plan_matches(state, config, pool, databases)
            models.append(
                (
                    name,
                    model,
                    {
                        "path": path,
                        "sha256": sha256_file(Path(path)),
                        "seed": state["metadata"].get("seed"),
                        "method": state["metadata"].get("method"),
                        "model_config": state["model_config"],
                        "best_step": finite_or_none(state.get("best_metric")),
                    },
                )
            )

    for name, model, identity in models:
        row = {**identity, "scores": {}}
        for steps in args.inner_steps:
            score, records = validate(
                model, pool, episodes, dict(config, inner_steps=steps), args.device
            )
            row["scores"][str(steps)] = {
                "validation_q95_log_error": score,
                "episodes": len(records),
            }
            print(
                json.dumps(
                    {
                        "checkpoint": name,
                        "inner_steps": steps,
                        "validation_q95_log_error": score,
                    }
                ),
                flush=True,
            )
        report["checkpoints"][name] = row

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
