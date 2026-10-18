"""Build the deterministic five-database Meta65 source-holdout plan."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import meta65_holdout_contract as contract


def validate_store(store: object, database: str, expected_unique: int,
                   expected_four_plus: int) -> tuple[list[int], list[str]]:
    cards = store.cards
    offsets = store.offsets
    rows = store.rows
    if database not in contract.EVALUATION or store.path.name != database:
        raise ValueError("Plan builder accepts only five source evaluation stores")
    if (cards.dtype != np.float64 or cards.ndim != 2 or cards.shape[1] != 2
            or not np.isfinite(cards).all() or (cards < 0).any()):
        raise ValueError("Invalid source cards")
    if (offsets.dtype.kind not in "iu" or offsets.ndim != 2 or offsets.shape[1] != 4
            or len(offsets) != len(rows) + 1 or (offsets[0] != 0).any()
            or (np.diff(offsets, axis=0) < 0).any()):
        raise ValueError("Invalid source offsets")
    tables = np.diff(offsets[:, 2])
    if (tables < 1).any() or (tables > 32767).any():
        raise ValueError("Invalid source table counts")
    first: dict[str, int] = {}
    for index, row in enumerate(rows):
        identity = row.get("query_hash")
        query_id = row.get("id")
        if (not isinstance(identity, str) or not identity
                or query_id != f"{database}/pretrain/{index + 1}"):
            raise ValueError("Invalid source identity")
        if identity in first:
            other = first[identity]
            if not np.array_equal(cards[index], cards[other]):
                raise ValueError("Conflicting duplicate source labels")
        else:
            first[identity] = index
    indices = list(first.values())
    hashes = [rows[index]["query_hash"] for index in indices]
    if len(indices) != expected_unique or len(set(hashes)) != expected_unique:
        raise ValueError(f"{database} unique-query denominator changed")
    four_plus = int((tables[indices] >= 4).sum())
    if four_plus != expected_four_plus:
        raise ValueError(f"{database} four-plus denominator changed")
    return indices, hashes


def build_plan(data: dict[str, object], config: dict[str, object]) -> dict[str, object]:
    from spatial_meta_ce.data import FeatureStore

    cache_root = Path(str(data["cache_root"])) / "pretrain"
    plans: dict[str, object] = {}
    expected = config["expected"]
    for database in contract.EVALUATION:
        store = FeatureStore(cache_root / database)
        indices, hashes = validate_store(
            store, database, expected["unique"][database], expected["four_plus"][database]
        )
        plans[database] = [
            contract.build_run_plan(
                database, indices, hashes,
                seed=config["evaluation"]["support_seed"], run=run,
            )
            for run in contract.RUNS
        ]
    contract.certify_plan(
        plans,
        {
            database: validate_store(
                FeatureStore(cache_root / database), database,
                expected["unique"][database], expected["four_plus"][database],
            )
            for database in contract.EVALUATION
        },
    )
    return plans


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/meta65_source_holdout.json")
    parser.add_argument("--data-config", type=Path,
                        default=ROOT / "configs/meta65_source_holdout_data.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = contract.load_config(args.config)
    data = contract.validate_data(contract.read_json(args.data_config))
    output = contract.checked_path(args.out)
    if output.exists():
        raise FileExistsError("Plan output must be fresh")
    if any(part in (*contract.TARGETS, "PRICE-master", "finetune") for part in output.parts):
        raise ValueError("Target/PRICE plan output forbidden")
    plans = build_plan(data, config)
    output.parent.mkdir(parents=True, exist_ok=True)
    contract.write_json(output, plans)
    print(f"plan: {len(plans)} databases, {sum(len(runs) for runs in plans.values())} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
