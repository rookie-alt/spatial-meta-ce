"""Fixed S6 episode risk and loss evidence; never used by adaptation."""

import json

import torch
from torch.nn import functional as F

SNAPSHOT_STEPS = frozenset((1, 1000, 2000, 4000))


def tail_terms(pred, label, mode):
    """Return FP32 losses and detached, tie-symmetric nominal coefficients."""
    if mode not in ("mean", "mean_tail10"):
        raise ValueError("tail_risk_mode must be mean or mean_tail10")
    for value in (pred, label):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError("S6 predictions and labels must be FP32 tensors")
        if value.ndim != 1 or value.numel() == 0:
            raise ValueError(
                "S6 predictions and labels must be nonempty one-dimensional tensors"
            )
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite S6 prediction or label")
    if pred.shape != label.shape or pred.device != label.device:
        raise ValueError("S6 prediction/label shape or device mismatch")
    losses = F.smooth_l1_loss(pred, label, reduction="none", beta=1)
    if not torch.isfinite(losses).all():
        raise FloatingPointError("Nonfinite S6 per-sample loss")
    detached = losses.detach()
    n = losses.numel()
    m = (n + 9) // 10
    tau = torch.topk(detached, m).values[-1]
    above, equal = detached > tau, detached == tau
    h, e = int(above.sum()), int(equal.sum())
    tail_coefficients = torch.where(
        above,
        torch.full_like(detached, 1.0 / m),
        torch.where(
            equal,
            torch.full_like(detached, (m - h) / (m * e)),
            torch.zeros_like(detached),
        ),
    )
    coefficients = (
        torch.full_like(detached, 1.0 / n)
        if mode == "mean"
        else 0.5 / n + 0.5 * tail_coefficients
    )
    mean_loss = losses.mean()
    tail_mean = (tail_coefficients * detached).sum()
    objective = mean_loss if mode == "mean" else (coefficients * losses).sum()
    if not torch.isfinite(objective):
        raise FloatingPointError("Nonfinite S6 objective")
    return {
        "objective": objective,
        "losses": losses,
        "coefficients": coefficients,
        "n": n,
        "m": m,
        "tau": float(tau),
        "h": h,
        "e": e,
        "mean_loss": float(mean_loss.detach()),
        "tail_mean": float(tail_mean),
    }


def snapshot_step(step, steps):
    return steps == 2 or step in SNAPSHOT_STEPS


def _append_json(path, row):
    # A closed handle per episode also preserves evidence on later training failure.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def capture_loss(
    output_dir,
    *,
    step,
    episode_index,
    database,
    k,
    mode,
    sample_plan_digest,
    terms,
    objective,
    pred,
    batch,
    hashes,
    steps,
):
    """Persist the actual backward objective and full ordered support+query batch."""
    coefficients = terms["coefficients"].detach().to(dtype=torch.float64)
    prediction = pred.detach().to(dtype=torch.float64)
    label = batch["label"].detach().to(dtype=torch.float64)
    tables = batch["counts"][:, 2]
    audit = {
        "step": step,
        "episode_index": episode_index,
        "database": database,
        "k": k,
        **{key: terms[key] for key in ("n", "m", "tau", "h", "e")},
        "mode": mode,
        "mean_loss": terms["mean_loss"],
        "tail_mean": terms["tail_mean"],
        "mixed_loss": float(objective.detach()),
        "coefficient_sum": float(coefficients.sum()),
        "coefficient_square_sum": float(coefficients.square().sum()),
        **{
            name: float(coefficients[mask].sum())
            for name, mask in (
                ("low_mass", tables < 4),
                ("high_mass", tables >= 4),
                ("under_mass", prediction < label),
                ("over_mass", prediction > label),
                ("exact_mass", prediction == label),
            )
        },
        "sample_plan_digest": sample_plan_digest,
    }
    _append_json(output_dir / "loss-audit.jsonl", audit)
    selected = snapshot_step(step, steps)
    if selected:
        losses = terms["losses"].detach().to(dtype=torch.float64)
        samples = [
            {
                "query_hash": query_hash,
                "role": "support" if i < k else "query",
                "log_pred": float(prediction[i]),
                "label": float(label[i]),
                "loss": float(losses[i]),
                "coefficient": float(coefficients[i]),
                "n_tables": int(tables[i]),
            }
            for i, query_hash in enumerate(hashes)
        ]
        _append_json(
            output_dir / "loss-snapshots.jsonl", {"audit": audit, "samples": samples}
        )
    return int(selected)
