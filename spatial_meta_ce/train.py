import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .checkpoints import load_checkpoint, restore_rng, save_checkpoint
from .data import StorePool
from .manifests import digest, episode_indices, file_hash, validate_splits
from .meta import (
    JOINT_PRE_WEIGHT,
    OBJECTIVES,
    _check_tau,
    adapt,
    forward,
    loss_value,
    meta_loss,
    meta_loss_terms,
)
from .metrics import summarize
from .model import create_model
from .price_protocol import fixed_model_scope
from .plans import (
    load_training_plan,
    validate_plan_matches_config,
    verify_plan_against_stores,
)
from .tail_risk import capture_loss, tail_terms

FIXED_STEPS = 4000
FIXED_META_BATCH = 4
FIXED_ACCUMULATION = 1
FIXED_QUERY_SIZE = 64
FIXED_SUPPORT_SIZES = (5, 10, 20, 50, 100)
FIXED_LOSS = "smooth_l1"
FIXED_INNER_LR = 0.001
FIXED_INNER_STEPS = 5
FIXED_ADAPT_SCOPE = "all"
FIXED_OUTER_LR = 0.0003
FIXED_MODEL = {
    "width": 64,
    "use_pg": True,
    "arch": "sum",
    "layers": 2,
    "heads": 4,
}
FIXED_SELECTION = "none (fixed 4000 updates, last.pt only)"
FIXED_SEEDS = (42, 43, 44)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_best(best_metric, bad_validations, metric, min_delta):
    if not np.isfinite(metric):
        raise ValueError("Validation metric is not finite")
    return (
        (metric, 0, True)
        if metric < best_metric - min_delta
        else (best_metric, bad_validations + 1, False)
    )


def state_dict_sha256(model):
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        hasher.update(name.encode())
        hasher.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def gradient_sha256(model, gradients):
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if len(named) != len(gradients):
        raise ValueError("Gradient audit received the wrong number of tensors")
    hasher = hashlib.sha256()
    for (name, parameter), gradient in zip(named, gradients):
        if not parameter.requires_grad:
            raise ValueError(f"Audited parameter {name} does not require grad")
        if gradient is None or not torch.isfinite(gradient).all():
            raise FloatingPointError(f"Non-finite audited gradient for {name}")
        hasher.update(name.encode())
        hasher.update(gradient.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def _write_gradient_audit(model, training_plan, stores, config, tau, output_dir, device):
    episode = training_plan["episodes"][0]
    store = stores[episode["db"]]
    support = store.batch(np.asarray(episode["support"], dtype=np.int64), device)
    query = store.batch(np.asarray(episode["query"], dtype=np.int64), device)
    if config["method"] == "fomaml":
        terms = meta_loss_terms(
            model,
            support,
            query,
            config["inner_lr"],
            config["inner_steps"],
            config["adapt_scope"],
            config["loss"],
            tau=tau,
            objective=config["meta_objective"],
        )
        total = terms["total"]
        pre_loss = float(terms["pre"].detach())
        post_loss = float(terms["post"].detach())
    else:
        batch = store.batch(
            np.asarray(
                episode["support"] + episode["query"],
                dtype=np.int64,
            ),
            device,
        )
        total = loss_value(
            model(batch),
            batch["label"],
            config["loss"],
            **({} if tau is None else {"tau": tau}),
        )
        pre_loss = None
        post_loss = float(total.detach())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(total, parameters)
    payload = {
        "schema": 1,
        "step": 1,
        "episode_index": 0,
        "database": episode["db"],
        "k": episode["k"],
        "support": episode["support"],
        "query": episode["query"],
        "support_hashes": episode["support_hashes"],
        "query_hashes": episode["query_hashes"],
        "pre_loss": pre_loss,
        "post_loss": post_loss,
        "total_loss": float(total.detach()),
        "gradient_sha256": gradient_sha256(model, gradients),
        "initial_state_sha256": file_hash(output_dir / "initial_state.pt"),
    }
    (output_dir / "gradient_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _validate_fixed_budget(config):
    if config["method"] == "erm" and config["meta_objective"] != "post":
        raise ValueError("ERM requires meta_objective=post")
    if config["method"] == "fomaml" and config["meta_objective"] not in OBJECTIVES:
        raise ValueError("Unknown meta_objective")
    declared = config.get("preregistered_step_budget")
    if declared is None:
        if config["steps"] != FIXED_STEPS:
            raise ValueError(f"fixed_budget requires steps={FIXED_STEPS}")
    elif (
        type(declared) is not int
        or declared != config["steps"]
        or declared <= FIXED_STEPS
        or not isinstance(config.get("preregistration"), str)
        or not config["preregistration"].strip()
    ):
        raise ValueError(
            "fixed_budget accepts a longer step budget only when the config "
            "declares matching preregistered_step_budget and preregistration"
        )
    if config["meta_batch"] != FIXED_META_BATCH:
        raise ValueError(f"fixed_budget requires meta_batch={FIXED_META_BATCH}")
    if config["accumulation"] != FIXED_ACCUMULATION:
        raise ValueError(f"fixed_budget requires accumulation={FIXED_ACCUMULATION}")
    if config["query_size"] != FIXED_QUERY_SIZE:
        raise ValueError(f"fixed_budget requires query_size={FIXED_QUERY_SIZE}")
    if list(config["support_sizes"]) != list(FIXED_SUPPORT_SIZES):
        raise ValueError(f"fixed_budget requires support_sizes={FIXED_SUPPORT_SIZES}")
    if not config.get("training_plan"):
        raise ValueError("fixed_budget requires a sealed training_plan")
    if config["loss"] != FIXED_LOSS:
        raise ValueError(f"fixed_budget requires loss={FIXED_LOSS}")
    if config["inner_lr"] != FIXED_INNER_LR:
        raise ValueError(f"fixed_budget requires inner_lr={FIXED_INNER_LR}")
    if config["inner_steps"] != FIXED_INNER_STEPS:
        raise ValueError(f"fixed_budget requires inner_steps={FIXED_INNER_STEPS}")
    expected_model, expected_scope = fixed_model_scope(config, FIXED_MODEL, FIXED_ADAPT_SCOPE)
    if config["adapt_scope"] != expected_scope:
        raise ValueError(f"fixed_budget requires adapt_scope={expected_scope}")
    if config["outer_lr"] != FIXED_OUTER_LR:
        raise ValueError(f"fixed_budget requires outer_lr={FIXED_OUTER_LR}")
    if config["model"] != expected_model:
        raise ValueError("fixed_budget requires the frozen model configuration")
    if "complex_query_weight" in config or "tail_risk_mode" in config:
        raise ValueError(
            "fixed_budget forbids complex_query_weight/tail_risk_mode"
        )
    if config["seed"] not in FIXED_SEEDS:
        raise ValueError(f"fixed_budget requires seed in {FIXED_SEEDS}")


def validation_plan(pool, databases, config):
    rng = np.random.default_rng(config["validation_seed"])
    episodes = []
    for db in databases:
        store = pool.get("pretrain", db)
        unique = store.unique_indices()
        for k in config["support_sizes"]:
            for run in range(config["validation_runs"]):
                si, qi = episode_indices(unique, k, config["query_size"], rng)
                episodes.append(
                    {
                        "db": db,
                        "k": k,
                        "run": run,
                        "support": si.tolist(),
                        "query": qi.tolist(),
                        "support_ids": [store.rows[int(i)]["id"] for i in si],
                        "query_ids": [store.rows[int(i)]["id"] for i in qi],
                        "support_hashes": [
                            store.rows[int(i)]["query_hash"] for i in si
                        ],
                        "query_hashes": [store.rows[int(i)]["query_hash"] for i in qi],
                    }
                )
    return episodes


def validate(model, pool, episodes, config, device):
    model.eval()
    records = []
    for ep in episodes:
        store = pool.get("pretrain", ep["db"])
        support = store.batch(ep["support"], device)
        query = store.batch(ep["query"], device)
        from .baselines import PriceAdapter, price_log_predictions, price_support

        is_price = isinstance(model, PriceAdapter)
        if is_price:
            support = price_support(support)
        params = adapt(
            model,
            support,
            config["inner_lr"],
            config["inner_steps"],
            config["adapt_scope"],
            config["loss"],
            training=False,
            **(
                {}
                if config.get("pinball_tau") is None
                else {"tau": config["pinball_tau"]}
            ),
        )
        with torch.no_grad():
            prediction = forward(model, params, query).cpu().numpy()
        if is_price:
            prediction = price_log_predictions(prediction)
        metrics = summarize(prediction, query["true"].cpu().numpy())
        records.append(
            {"db": ep["db"], "k": ep["k"], "run": ep["run"], "metrics": metrics}
        )
    # Equal database, K and repeat counts: macro mean, not pooled query weighting.
    return float(np.mean([r["metrics"]["log_q95"] for r in records])), records


def _legacy_sample_plan(
    config: dict,
    databases: list,
    pools: dict,
    pool: StorePool,
    steps: int,
    generator: dict,
) -> tuple[str, int]:
    rng = np.random.default_rng(config["seed"] + 1)
    previous = hashlib.sha256(b"").hexdigest()
    for _ in range(steps * config["meta_batch"]):
        db = str(rng.choice(databases))
        k = int(rng.choice(config["support_sizes"]))
        si, qi = episode_indices(pools[db], k, config["query_size"], rng)
        store = pool.get("pretrain", db)
        episode = {
            "db": db,
            "support": [store.rows[int(i)]["query_hash"] for i in si],
            "query": [store.rows[int(i)]["query_hash"] for i in qi],
        }
        previous = digest({"previous": previous, "episode": episode})
    if rng.bit_generator.state != generator:
        raise ValueError("Legacy sampling RNG cannot be reconstructed")
    return previous, steps * config["meta_batch"]


def train(config, data_config, output_dir, device="cpu", resume=None):
    stage = config.get("fixed_stage")
    if stage is not None and not isinstance(stage, str):
        raise ValueError("fixed_stage must be a string or null")
    stage = stage or ""
    module_stage = stage == "132-module26"
    if module_stage:
        from .module_experiment import validate_training_config

        validate_training_config(config, data_config, resume, output_dir=output_dir, device=device)
    s6 = "tail_risk_mode" in config
    s5 = "complex_query_weight" in config
    if s6:
        if config["tail_risk_mode"] not in ("mean", "mean_tail10"):
            raise ValueError("tail_risk_mode must be mean or mean_tail10")
        if s5:
            raise ValueError("tail_risk_mode forbids complex_query_weight")
        if (
            config["method"] != "erm"
            or config["loss"] != "smooth_l1"
            or config["accumulation"] != 1
            or config["meta_batch"] != 4
        ):
            raise ValueError(
                "tail_risk_mode requires ERM/SmoothL1/accumulation1/meta_batch4"
            )
        if resume:
            raise ValueError("S6 does not authorize resume; train from scratch")
        loss_evidence = {
            "audit_rows": 0,
            "snapshot_rows": 0,
            "mode": config["tail_risk_mode"],
        }
    if s5:
        weight = config["complex_query_weight"]
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or weight not in (1, 2)
        ):
            raise ValueError("complex_query_weight must be numeric 1 or 2")
        if config["method"] != "erm" or config["loss"] != "smooth_l1":
            raise ValueError(
                "complex_query_weight requires method=erm and loss=smooth_l1"
            )
    # P1-1/P1-2 (doc 93 §3). Both are opt-in; the legacy paths above are untouched.
    tau = config.get("pinball_tau")
    if config["loss"] == "pinball_tau":
        if tau is None:
            raise ValueError("loss=pinball_tau requires a pre-registered pinball_tau")
        tau = _check_tau(tau)
    elif tau is not None:
        raise ValueError("pinball_tau config requires loss=pinball_tau")
    if (s5 or s6) and config["loss"] != "smooth_l1":
        raise ValueError("complex_query_weight/tail_risk_mode requires loss=smooth_l1")
    from .price_capacity import OFFICIAL_TRAIN_DATABASES

    validate_splits(
        data_config["splits"],
        allow_empty_validation=tuple(data_config["splits"]["train"])
        == OFFICIAL_TRAIN_DATABASES,
    )
    for key in [
        "steps",
        "meta_batch",
        "accumulation",
        "query_size",
        "validate_every",
        "validation_runs",
        "patience",
    ]:
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if config["method"] not in ("fomaml", "erm") or min(config["support_sizes"]) < 1:
        raise ValueError("Invalid training method or support sizes")
    fixed_budget = bool(config.get("fixed_budget", False))
    meta_objective = config.get("meta_objective", "post")
    if meta_objective not in OBJECTIVES:
        raise ValueError("Unknown meta_objective")
    if meta_objective == "joint" and config["method"] != "fomaml":
        raise ValueError("joint objective requires method=fomaml")
    if meta_objective == "joint" and not fixed_budget:
        raise ValueError("joint objective requires fixed_budget")
    if (
        config.get("model", {}).get("arch") in ("sum_anchor_only", "price_lite", "price_tiny")
        and not stage.startswith("122-")
        and not module_stage
    ):
        raise ValueError("122 candidate training requires its frozen fixed_stage")
    if config.get("fixed_stage") is not None and not fixed_budget:
        raise ValueError("fixed_stage requires fixed_budget")
    if fixed_budget and not module_stage:
        _validate_fixed_budget({**config, "meta_objective": meta_objective})
    if stage.startswith("122-"):
        if tuple(data_config["splits"]["train"]) == OFFICIAL_TRAIN_DATABASES:
            from .official_split import build_official_capacity, validate_official_splits

            validate_official_splits(data_config["splits"])
            if config["model"].get("capacity") != build_official_capacity(data_config):
                raise ValueError("Official split capacity differs from the 26 caches")
        else:
            from .price_capacity import validate_source_splits
            from .price_protocol import validate_training_preflight

            validate_source_splits(data_config)
            validate_training_preflight(config, data_config)
    if config.get("training_plan") and not fixed_budget:
        raise ValueError("training_plan requires fixed_budget")
    if fixed_budget and resume:
        raise ValueError("fixed_budget does not authorize resume")
    mapped = config.get("model", {}).get("arch") == "sum_map" or (
        module_stage and config["model"].get("relation_attention") is True
    )
    mapping_config = config.get("mapping")
    if mapped and not mapping_config:
        raise ValueError("sum_map training requires mapping configuration")
    if not mapped and mapping_config:
        raise ValueError("Mapping configuration requires an authorized mapped architecture")
    if mapping_config and mapping_config.get("mode") not in (
        "mapping",
        "shuffled_mapping",
    ):
        raise ValueError("Invalid mapping mode")
    seed_all(config["seed"])
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError("Use a new run directory or explicit --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = data_config["splits"]
    sources = splits["train"] if fixed_budget else splits["train"] + splits["validation"]
    # Opening a mapped store parses its whole sidecar (~3.5s). Keep every source
    # store resident so uniform database sampling cannot thrash the LRU.
    if module_stage and mapping_config:
        from .official_mappings import OfficialStorePool

        pool = OfficialStorePool(
            data_config["cache_root"], mapping_config["root"],
            max_open=len(sources), mapping_mode=mapping_config["mode"],
        )
    else:
        pool = StorePool(
            data_config["cache_root"],
            max_open=len(sources),
            mapping_root=mapping_config["root"] if mapping_config else None,
            mapping_mode=mapping_config["mode"] if mapping_config else "mapping",
        )
    hashes = {}
    pools = {}
    stores = {}
    mapping_provenance = {}
    for db in sources:
        store = pool.get("pretrain", db)
        stores[db] = store
        hashes[db] = store.manifest["content_hash"]
        pools[db] = store.unique_indices()
        if mapping_config:
            mapping_provenance[db] = store.mapping.provenance()
    signature = {k: v for k, v in config.items() if k not in ("steps", "data_config")}
    metadata = {
        "method": config["method"],
        "seed": config["seed"],
        "train_databases": splits["train"],
        "validation_databases": [] if fixed_budget else splits["validation"],
        "manifest_hashes": hashes,
        "training_signature": signature,
        "purpose": config.get("purpose", "research"),
        "selection": (
            FIXED_SELECTION
            if fixed_budget
            else "source-validation macro mean log-Q95 after adaptation"
        ),
    }
    if fixed_budget:
        metadata = {
            **metadata,
            "meta_objective": meta_objective,
            "joint_pre_weight": (
                JOINT_PRE_WEIGHT if meta_objective == "joint" else None
            ),
            "fixed_budget": True,
            "fixed_steps": FIXED_STEPS,
            "selection_databases": [],
            "validation_monitor_reads": [],
            "training_plan": str(Path(config["training_plan"]).resolve()),
        }
    if mapping_config:
        metadata = {**metadata, "mapping_provenance": mapping_provenance}
    if module_stage:
        from .module_experiment import checkpoint_metadata

        metadata.update(checkpoint_metadata(config))
    rng = np.random.default_rng(config["seed"] + 1)
    model = create_model(config["model"]).to(device)
    if config.get("fixed_stage"):
        metadata = {**metadata, "fixed_stage": config["fixed_stage"]}
    if fixed_budget and not resume:
        metadata["initial_state_sha256"] = state_dict_sha256(model)
    start, best, bad = 0, float("inf"), 0
    history = []
    sample_plan_digest = hashlib.sha256(b"").hexdigest()
    sample_plan_episodes = 0
    counts = {
        "support_presentations": 0,
        "query_presentations": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    if fixed_budget:
        counts.update({"pre_forward_calls": 0, "joint_pre_grad_calls": 0})
    total_seconds = 0.0
    if s5 or s6 or fixed_budget:
        gradient_stats = {
            "count": 0,
            "total_norm": 0.0,
            "max_norm": 0.0,
            "clipped_count": 0,
        }
    if resume:
        model, state = load_checkpoint(resume, device)
        if state["metadata"] != metadata:
            raise ValueError("Resume configuration or data manifest changed")
        start, best, bad = (
            state["next_step"],
            state["best_metric"],
            state["bad_validations"],
        )
        history = state["extra"]["history"]
        if "sample_plan_digest" in state["extra"]:
            sample_plan_digest = state["extra"]["sample_plan_digest"]
            sample_plan_episodes = state["extra"]["sample_plan_episodes"]
        elif not mapped:
            sample_plan_digest, sample_plan_episodes = _legacy_sample_plan(
                config, splits["train"], pools, pool, start, state["extra"]["generator"]
            )
        else:
            raise ValueError("Mapped checkpoint is missing its sampling provenance")
        rng.bit_generator.state = state["extra"]["generator"]
        counts = state["extra"]["counts"]
        total_seconds = state["extra"]["total_seconds"]
        if s5:
            if "gradient_stats" not in state["extra"]:
                raise ValueError("Resume checkpoint missing gradient_stats")
            gradient_stats = dict(state["extra"]["gradient_stats"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["outer_lr"])
    if resume:
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    if fixed_budget:
        training_plan = load_training_plan(
            config["training_plan"],
            data_config,
            allow_empty_validation=tuple(data_config["splits"]["train"])
            == OFFICIAL_TRAIN_DATABASES,
        )
        validate_plan_matches_config(training_plan, config)
        verify_plan_against_stores(training_plan, stores)
        sample_plan_digest = training_plan["plan_sha256"]
        metadata = {**metadata, "training_plan_sha256": sample_plan_digest}
        episodes = []
        torch.save(
            {"model_type": model.model_type, "model_config": model.config, "state_dict": model.state_dict()},
            output_dir / "initial_state.pt",
        )
        _write_gradient_audit(
            model, training_plan, stores, config, tau, output_dir, device
        )
    else:
        training_plan = None
        episodes = validation_plan(pool, splits["validation"], config)
        (output_dir / "validation_manifest.json").write_text(
            json.dumps(episodes, indent=2)
        )
    (output_dir / "config.json").write_text(
        json.dumps({"training": config, "data": data_config}, indent=2)
    )
    started = time.perf_counter()
    last_step = start
    loss_audit = []
    for window_start in range(start, config["steps"], config["accumulation"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        window = min(config["accumulation"], config["steps"] - window_start)
        losses = []
        pre_losses = []
        post_losses = []
        for _ in range(window):
            for _ in range(config["meta_batch"]):
                if fixed_budget:
                    episode = training_plan["episodes"][sample_plan_episodes]
                    sample_plan_episodes += 1
                    db = episode["db"]
                    k = episode["k"]
                    si = np.asarray(episode["support"], dtype=np.int64)
                    qi = np.asarray(episode["query"], dtype=np.int64)
                    store = stores[db]
                    episode_hashes = {
                        "db": db,
                        "support": episode["support_hashes"],
                        "query": episode["query_hashes"],
                    }
                else:
                    db = str(rng.choice(splits["train"]))
                    k = int(rng.choice(config["support_sizes"]))
                    si, qi = episode_indices(
                        pools[db], k, config["query_size"], rng
                    )
                    store = pool.get("pretrain", db)
                    episode_hashes = {
                        "db": db,
                        "support": [store.rows[int(i)]["query_hash"] for i in si],
                        "query": [store.rows[int(i)]["query_hash"] for i in qi],
                    }
                    sample_plan_digest = digest(
                        {"previous": sample_plan_digest, "episode": episode_hashes}
                    )
                    sample_plan_episodes += 1
                if config["method"] == "fomaml":
                    support, query = store.batch(si, device), store.batch(qi, device)
                    if fixed_budget:
                        terms = meta_loss_terms(
                            model,
                            support,
                            query,
                            config["inner_lr"],
                            config["inner_steps"],
                            config["adapt_scope"],
                            config["loss"],
                            tau=tau,
                            objective=meta_objective,
                        )
                        objective = terms["total"]
                        pre_losses.append(float(terms["pre"].detach()))
                        post_losses.append(float(terms["post"].detach()))
                        counts["pre_forward_calls"] += 1
                        if meta_objective == "joint":
                            counts["joint_pre_grad_calls"] += 1
                    else:
                        objective = meta_loss(
                            model,
                            support,
                            query,
                            config["inner_lr"],
                            config["inner_steps"],
                            config["adapt_scope"],
                            config["loss"],
                            tau=tau,
                        )
                        pre_losses.append(None)
                        post_losses.append(float(objective.detach()))
                    counts["support_presentations"] += k * config["inner_steps"]
                    counts["query_presentations"] += len(qi)
                    counts["backward_calls"] += config["inner_steps"] + 1
                else:
                    batch = store.batch(np.concatenate([si, qi]), device)
                    if s6:
                        pred = model(batch)
                        terms = tail_terms(
                            pred, batch["label"], config["tail_risk_mode"]
                        )
                        objective = (
                            loss_value(pred, batch["label"], config["loss"])
                            if config["tail_risk_mode"] == "mean"
                            else terms["objective"]
                        )
                        snapshot_rows = capture_loss(
                            output_dir,
                            step=window_start + 1,
                            episode_index=sample_plan_episodes,
                            database=db,
                            k=k,
                            mode=config["tail_risk_mode"],
                            sample_plan_digest=sample_plan_digest,
                            terms=terms,
                            objective=objective,
                            pred=pred,
                            batch=batch,
                            hashes=episode_hashes["support"]
                            + episode_hashes["query"],
                            steps=config["steps"],
                        )
                        loss_evidence = {
                            **loss_evidence,
                            "audit_rows": loss_evidence["audit_rows"] + 1,
                            "snapshot_rows": loss_evidence["snapshot_rows"]
                            + snapshot_rows,
                        }
                    elif s5 and config["complex_query_weight"] == 2:
                        weights = torch.where(batch["counts"][:, 2] >= 4, 2.0, 1.0).to(
                            torch.float32
                        )
                        objective = loss_value(
                            model(batch),
                            batch["label"],
                            config["loss"],
                            weights=weights,
                        )
                    else:
                        objective = loss_value(
                            model(batch),
                            batch["label"],
                            config["loss"],
                            **({} if tau is None else {"tau": tau}),
                        )
                    pre_losses.append(None)
                    post_losses.append(float(objective.detach()))
                    counts["query_presentations"] += len(si) + len(qi)
                    counts["backward_calls"] += 1
                losses.append(float(objective.detach()))
                (objective / (window * config["meta_batch"])).backward()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise FloatingPointError("Nonfinite outer gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        if s5 or s6 or fixed_budget:
            norm_value = float(norm)
            gradient_stats = {
                "count": gradient_stats["count"] + 1,
                "total_norm": gradient_stats["total_norm"] + norm_value,
                "max_norm": max(gradient_stats["max_norm"], norm_value),
                "clipped_count": gradient_stats["clipped_count"]
                + int(norm_value > 1.0),
            }
        optimizer.step()
        counts["optimizer_steps"] += 1
        last_step = window_start + window
        if fixed_budget:
            loss_audit.append(
                {
                    "step": last_step,
                    "mean_train_loss": float(np.mean(losses)),
                    "mean_pre_loss": (
                        None
                        if any(value is None for value in pre_losses)
                        else float(np.mean(pre_losses))
                    ),
                    "mean_post_loss": float(np.mean(post_losses)),
                    "grad_norm": norm_value,
                    "grad_clipped": norm_value > 1.0,
                    "counts": dict(counts),
                }
            )
        check = not fixed_budget and (
            last_step // config["validate_every"]
            > window_start // config["validate_every"]
            or last_step == config["steps"]
        )
        improved = False
        if check:
            score, records = validate(model, pool, episodes, config, device)
            best, bad, improved = select_best(best, bad, score, config["min_delta"])
            row = {
                "step": last_step,
                "train_loss": float(np.mean(losses)),
                "validation_macro_log_q95": score,
                "validation": records,
                "counts": dict(counts),
            }
            history.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "validation"}),
                flush=True,
            )
        extra = {
            "history": history,
            "generator": rng.bit_generator.state,
            "counts": counts,
            "sample_plan_digest": sample_plan_digest,
            "sample_plan_episodes": sample_plan_episodes,
            "total_seconds": total_seconds + time.perf_counter() - started,
        }
        if s5 or s6 or fixed_budget:
            extra = {**extra, "gradient_stats": dict(gradient_stats)}
        if s6:
            extra = {**extra, "loss_evidence": dict(loss_evidence)}
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            last_step,
            best,
            bad,
            metadata,
            extra,
        )
        if improved and not fixed_budget:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                last_step,
                best,
                bad,
                metadata,
                extra,
            )
        if check and bad >= config["patience"]:
            break
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    if fixed_budget:
        (output_dir / "loss_audit.jsonl").write_text(
            "\n".join(json.dumps(row) for row in loss_audit) + "\n"
        )
    return {
        "last_step": last_step,
        "best_validation_macro_log_q95": None if fixed_budget else best,
        "counts": counts,
        "total_seconds": total_seconds + time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "purpose": metadata["purpose"],
        **(
            {"gradient_stats": dict(gradient_stats)}
            if s5 or s6 or fixed_budget
            else {}
        ),
        **({"loss_evidence": dict(loss_evidence)} if s6 else {}),
        **(
            {
                "meta_objective": meta_objective,
                "fixed_budget": True,
                "training_plan_sha256": sample_plan_digest,
            }
            if fixed_budget
            else {}
        ),
    }
