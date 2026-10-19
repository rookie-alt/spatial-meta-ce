"""Fixed S8 source identity boundaries and frozen forward capture; no training."""

import hashlib

import torch

ORIGINAL_TRAIN = (
    "accidents",
    "airline",
    "baseball",
    "basketball",
    "carcinogenesis",
    "ccs",
    "consumer",
    "credit",
    "employee",
    "financial",
    "fnhk",
    "grants",
    "hepatitis",
    "hockey",
    "movielens",
    "sakila",
    "sap",
    "seznam",
    "ssb",
    "talkingdata",
    "tournament",
    "tpc_h",
    "tubepricing",
)
HELD_OUT = ("accidents", "grants", "employee", "hockey")
TRAIN = tuple(db for db in ORIGINAL_TRAIN if db not in HELD_OUT)
VALIDATION = ("chembl", "legalacts", "telstra")
TARGETS = ("imdb", "stats", "genome", "ergastf1")
PANEL_SIZE = 1024


def partition_databases(databases):
    if tuple(databases) != ORIGINAL_TRAIN:
        raise ValueError("S8 frozen source database list/order mismatch")
    ordered = sorted(
        databases,
        key=lambda db: (
            hashlib.sha256(f"s8-db-holdout-v1|{db}".encode()).hexdigest(),
            db,
        ),
    )
    if tuple(ordered[:4]) != HELD_OUT:
        raise ValueError("S8 source holdout identity hash mismatch")
    return {
        "train": list(TRAIN),
        "holdout": ordered[:4],
        "validation": list(VALIDATION),
    }


def panel_indices(database, query_hashes):
    if database not in TRAIN:
        raise ValueError("S8 training panel requires a permitted T source database")
    hashes = list(query_hashes)
    if (
        len(hashes) < PANEL_SIZE + 200
        or any(not isinstance(h, str) or not h for h in hashes)
        or len(set(hashes)) != len(hashes)
    ):
        raise ValueError("S8 insufficient/duplicate/invalid unique query pool")
    ordered = sorted(
        range(len(hashes)),
        key=lambda i: (
            hashlib.sha256(
                f"s8-train-panel-v1|{database}|{hashes[i]}".encode()
            ).hexdigest(),
            hashes[i],
        ),
    )
    selected = ordered[:PANEL_SIZE]
    selected_set = set(selected)
    return selected, [i for i in range(len(hashes)) if i not in selected_set]


def extract_representations(model, batch):
    """Capture x/h during the exact original eval forward without editing its code."""
    if (
        model.training
        or model.config.get("width") != 64
        or model.arch != "sum"
        or not model.use_pg
        or model.pg_output_scale != 1.0
        or model.head[0].in_features != 774
        or model.head[-1].in_features != 64
    ):
        raise ValueError(
            "S8 representation capture requires frozen eval sum64/PG anchor1"
        )
    captured = {}

    def capture_x(_module, inputs):
        captured["x"] = inputs[0].detach()

    def capture_h(_module, inputs):
        captured["h"] = inputs[0].detach()

    handles = []
    try:
        handles.append(model.head[0].register_forward_pre_hook(capture_x))
        handles.append(model.head[-1].register_forward_pre_hook(capture_h))
        with torch.inference_mode():
            captured["raw_log_pred"] = model(batch).detach()
            captured["pg_log"] = batch["pg_log"].detach()
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"x", "h", "raw_log_pred", "pg_log"} or any(
        v.dtype != torch.float32 or not torch.isfinite(v).all()
        for v in captured.values()
    ):
        raise ValueError("S8 nonfinite/non-FP32 representation bridge")
    return {key: value.clone() for key, value in captured.items()}
