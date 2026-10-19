"""Measure the P1-2 residual gate g(q) on source validation databases.

Doc 93 §3 lists one failure mode for P1-2: the gate collapsing to zero. This
module reports the distribution so a collapsed gate is visible in the record
instead of being inferred from an unchanged Q95.

Read-only: it never trains, adapts or touches a held-out target database.
"""

import numpy as np
import torch

BATCH = 512
VALIDATION = ("chembl", "legalacts", "telstra")


def _percentiles(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "min": float(values.min()),
        "max": float(values.max()),
        "fraction_below_0_1": float(np.mean(values < 0.1)),
    }


def _gate_values(model, store, device: str) -> np.ndarray:
    indices = store.unique_indices()
    chunks = []
    model.eval()
    for start in range(0, len(indices), BATCH):
        batch = store.batch(indices[start : start + BATCH], device)
        with torch.no_grad():
            gate = model.gate_values(batch).cpu().numpy()
        if not np.isfinite(gate).all():
            raise FloatingPointError("Nonfinite gate value")
        if ((gate < 0.0) | (gate > 1.0)).any():
            raise ValueError("Gate escaped [0, 1]")
        chunks.append(gate)
    if not chunks:
        raise ValueError("Store has no unique queries")
    return np.concatenate(chunks)


def gate_report(model, pool, device: str, databases=VALIDATION) -> dict:
    """Distribution of g(q) over every unique query of each source database."""
    if getattr(model, "gate_mode", None) is None:
        raise ValueError("gate_report requires a model built with gate='sigmoid'")
    per_database = {}
    pooled = []
    for db in databases:
        values = _gate_values(model, pool.get("pretrain", db), device)
        per_database[db] = {"queries": int(len(values)), **_percentiles(values)}
        pooled.append(values)
    combined = np.concatenate(pooled)
    return {
        "queries": int(len(combined)),
        **_percentiles(combined),
        "per_database": per_database,
    }
