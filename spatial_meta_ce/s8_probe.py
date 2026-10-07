"""S8 local numerical probes, not a data runner or mechanism-result verifier.

Callers supply P-only features and the matching FP32 training labels/PG logs.
This module cannot establish their provenance. No clipping, regularization,
solver fallback, model access, or device selection is performed here.
"""

from functools import wraps

import numpy as np
import torch

RCOND = 1e-12
NORMAL_TOL = 1e-8
OBJECTIVE_TOL = 1e-8


def _reject_numeric_failure(function):
    @wraps(function)
    def checked(*args, **kwargs):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return function(*args, **kwargs)
        except (
            FloatingPointError,
            OverflowError,
            np.linalg.LinAlgError,
            RuntimeError,
        ) as exc:
            raise ValueError(f"{function.__name__}: numerical failure: {exc}") from exc

    return checked


def _array64(value, ndim, name):
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric ndarray")
    if value.ndim != ndim or not value.size or any(n == 0 for n in value.shape):
        raise ValueError(f"{name} has invalid or empty shape")
    result = value.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite in FP64")
    return result


@_reject_numeric_failure
def standardize_fit(z):
    """Fit P-only population mean/std, effective scale and P masks in FP64."""
    z = _array64(z, 2, "z")
    mean = z.mean(axis=0)
    constant = np.all(z == z[0], axis=0)
    mean = np.where(constant, z[0], mean)
    centered = z - mean
    spread = np.max(np.abs(centered), axis=0)
    # Population variance (ddof=0), scaled to avoid squaring tiny values.
    unit = centered / np.where(spread == 0, 1.0, spread)
    std = spread * np.sqrt(np.mean(unit**2, axis=0))
    if ((std == 0) & ~constant).any():
        raise ValueError("nonconstant population std underflow")
    zero_mask = std == 0
    tiny_mask = (std > 0) & (std < 1e-12)
    scale = np.where(zero_mask, 1.0, np.maximum(std, 1e-12))
    normalizer = {
        "mean": mean,
        "std": std,
        "scale": scale,
        "zero_mask": zero_mask,
        "tiny_mask": tiny_mask,
    }
    return standardize_apply(z, normalizer), normalizer


@_reject_numeric_failure
def standardize_apply(z, normalizer):
    """Apply only P mean/scale; std and P masks are diagnostic, never applied."""
    z = _array64(z, 2, "z")
    if not isinstance(normalizer, dict) or not {"mean", "scale"} <= normalizer.keys():
        raise ValueError("normalizer must contain mean and scale")
    mean = _array64(normalizer["mean"], 1, "mean")
    scale = _array64(normalizer["scale"], 1, "scale")
    if (
        mean.shape != (z.shape[1],)
        or scale.shape != mean.shape
        or (scale < 1e-12).any()
    ):
        raise ValueError("normalizer shape/denominator is invalid")
    design = np.column_stack(((z - mean) / scale, np.ones(z.shape[0])))
    if not np.isfinite(design).all():
        raise ValueError("standardized design must be finite")
    return design


@_reject_numeric_failure
def residual_targets(label_fp32, pg_log_fp32):
    """Subtract only after separately promoting the original FP32 inputs."""
    for name, value in [("label_fp32", label_fp32), ("pg_log_fp32", pg_log_fp32)]:
        if not isinstance(value, np.ndarray) or value.dtype != np.dtype("float32"):
            raise ValueError(f"{name} must have dtype float32")
    label = _array64(label_fp32, 1, "label_fp32")
    pg = _array64(pg_log_fp32, 1, "pg_log_fp32")
    if label.shape != pg.shape:
        raise ValueError("label/PG shapes must match")
    return label - pg


def _problem(design, target):
    design = _array64(design, 2, "design")
    target = _array64(target, 1, "target")
    if design.shape[0] != target.size:
        raise ValueError("design rows must match target length")
    scale = max(1.0, float(np.linalg.norm(design, "fro") * np.linalg.norm(target)))
    if not np.isfinite(scale):
        raise ValueError("normal-equation denominator overflow")
    return design, target, scale


def _diagnostics(design, target, theta, scale):
    theta = _array64(theta, 1, "theta")
    if theta.shape != (design.shape[1],):
        raise ValueError("theta must match design columns")
    residual = design @ theta - target
    objective = float(np.mean(residual**2))
    normal = float(np.max(np.abs(design.T @ residual)) / scale)
    if not np.isfinite(objective) or not np.isfinite(normal):
        raise ValueError("objective/normal residual must be finite")
    return objective, normal


def _rank_and_singular(rank, singular, shape):
    limit = min(shape)
    if (
        isinstance(rank, (bool, np.bool_))
        or not isinstance(rank, (int, np.integer))
        or not 0 <= rank <= limit
    ):
        raise ValueError("rank must be an integer within the matrix dimensions")
    singular = _array64(singular, 1, "singular_values")
    if (
        singular.shape != (limit,)
        or (singular < 0).any()
        or (np.diff(singular) > 0).any()
    ):
        raise ValueError("singular values have invalid shape/order/sign")
    expected = int(np.count_nonzero(singular > RCOND * singular[0]))
    if not 0 <= rank <= limit or rank != expected:
        raise ValueError("rank is inconsistent with the fixed singular cutoff")
    return int(rank), singular


@_reject_numeric_failure
def solve_ls(design, target):
    """Fixed SVD lstsq, rcond=1e-12. Success alone is not certification."""
    design, target, scale = _problem(design, target)
    theta, _, rank, singular = np.linalg.lstsq(design, target, rcond=RCOND)
    rank, singular = _rank_and_singular(rank, singular, design.shape)
    objective, normal = _diagnostics(design, target, theta, scale)
    return {
        "theta": theta,
        "objective": objective,
        "rank": rank,
        "singular_values": singular,
        "normalized_normal_residual": normal,
    }


def _updates(steps):
    if (
        isinstance(steps, (bool, np.bool_))
        or not isinstance(steps, (int, np.integer))
        or steps < 0
    ):
        raise ValueError("steps/updates must be a nonnegative integer")
    return int(steps)


@_reject_numeric_failure
def solve_adam(design, target, steps=1000):
    """CPU FP64 zero-init full-batch MSE Adam; steps=0 exposes the PG start.

    No RNG operations are needed for this deterministic zero-initialized probe;
    the caller's global RNG state and thread/device settings are not changed.
    """
    design, target, scale = _problem(design, target)
    steps = _updates(steps)
    x = torch.from_numpy(design)
    y = torch.from_numpy(target)
    theta = torch.zeros(
        design.shape[1], dtype=torch.float64, device="cpu", requires_grad=True
    )
    optimizer = torch.optim.Adam(
        [theta], lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((x @ theta - y) ** 2).mean()
        if not torch.isfinite(loss):
            raise ValueError("Adam loss is nonfinite")
        loss.backward()
        if not torch.isfinite(theta.grad).all():
            raise ValueError("Adam gradient is nonfinite")
        optimizer.step()
        state = optimizer.state[theta]
        if not all(
            torch.isfinite(v).all()
            for v in [theta, state["exp_avg"], state["exp_avg_sq"]]
        ):
            raise ValueError("Adam parameters/moments are nonfinite")
    coefficients = theta.detach().numpy().copy()
    objective, _ = _diagnostics(design, target, coefficients, scale)
    return {"theta": coefficients, "objective": objective, "updates": steps}


def _finite_scalar(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (float, int, np.floating, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} must be a nonnegative real scalar")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _matches(reported, computed):
    # Metadata is stricter than the cross-solver certification threshold.
    return bool(np.isclose(reported, computed, rtol=1e-12, atol=1e-12))


@_reject_numeric_failure
def certify_pair(design, target, adam, ls):
    """Recompute P diagnostics; malformed/nonfinite inputs raise ValueError.

    Finite inconsistent diagnostics or failed numerical gates return pass=False.
    pass/solver_verified certify only the fixed-cutoff numerical comparison.
    A fixed-cutoff reference lstsq must also match objective and predictions:
    small normal residuals alone cannot authenticate weak singular directions.
    fullrank_numeric_solver means verified and rank == parameter columns.
    Legacy full_linear_optimality fields always remain False: repeating the same
    numerical solver is not an independent scientific proof of optimality.
    Rank deficiency (even exact redundancy) is not missing-information evidence.
    Certification performs one SVD and one reference lstsq, in addition to the
    caller's initial lstsq: three factorizations in the standard pair workflow.
    """
    design, target, scale = _problem(design, target)
    if (
        not isinstance(adam, dict)
        or not {"theta", "objective", "updates"} <= adam.keys()
    ):
        raise ValueError("Adam result fields are missing")
    fields = {
        "theta",
        "objective",
        "rank",
        "singular_values",
        "normalized_normal_residual",
    }
    if not isinstance(ls, dict) or not fields <= ls.keys():
        raise ValueError("LS result fields are missing")
    _updates(adam["updates"])
    rank, singular = _rank_and_singular(ls["rank"], ls["singular_values"], design.shape)
    actual_singular = np.linalg.svd(design, compute_uv=False)
    actual_rank, _ = _rank_and_singular(rank, actual_singular, design.shape)
    ja, _ = _diagnostics(design, target, adam["theta"], scale)
    jl, normal = _diagnostics(design, target, ls["theta"], scale)
    reported_ja = _finite_scalar(adam["objective"], "Adam objective")
    reported_jl = _finite_scalar(ls["objective"], "LS objective")
    reported_normal = _finite_scalar(
        ls["normalized_normal_residual"], "normal residual"
    )
    metadata_ok = (
        _matches(reported_ja, ja)
        and _matches(reported_jl, jl)
        and _matches(reported_normal, normal)
        and bool(np.allclose(singular, actual_singular, rtol=1e-12, atol=0))
    )
    normal_ok = normal <= NORMAL_TOL
    objective_ok = jl <= ja + OBJECTIVE_TOL * max(1.0, ja)
    # Re-solve the SAME fixed-cutoff problem, never a fallback or cutoff search.
    reference_theta, _, reference_rank, reference_singular = np.linalg.lstsq(
        design, target, rcond=RCOND
    )
    reference_rank, reference_singular = _rank_and_singular(
        reference_rank, reference_singular, design.shape
    )
    if reference_rank != actual_rank:
        raise ValueError("reference rank disagrees with the design SVD")
    reference_objective, _ = _diagnostics(design, target, reference_theta, scale)
    reference_objective_ok = _matches(jl, reference_objective)
    # Use the existing metadata tolerance, not a new projection/rank threshold.
    reference_prediction_ok = bool(
        np.allclose(
            design @ ls["theta"], design @ reference_theta, rtol=1e-12, atol=1e-12
        )
    )
    reference_ok = bool(
        reference_objective_ok
        and reference_prediction_ok
        and np.allclose(reference_singular, actual_singular, rtol=1e-12, atol=0)
    )
    solver_verified = bool(metadata_ok and normal_ok and objective_ok and reference_ok)
    rank_limited = actual_rank < design.shape[1]
    return {
        "pass": solver_verified,
        "solver_verified": solver_verified,
        "rank_limited": rank_limited,
        "numerical_rank_limited": rank_limited,
        "fullrank_numeric_solver": bool(solver_verified and not rank_limited),
        "full_linear_optimality_verified": False,
        "full_linear_optimality": False,
        "reference_objective": reference_objective,
        "reference_objective_pass": reference_objective_ok,
        "reference_prediction_pass": reference_prediction_ok,
        "reference_pass": reference_ok,
        "certification_linalg_calls": 2,
        "normalized_normal_residual": normal,
        "normal_residual_pass": bool(normal_ok),
        "objective_pass": bool(objective_ok),
        "metadata_pass": bool(metadata_ok),
        "objective_ls": jl,
        "objective_adam": ja,
    }
