import numpy as np

# q80 is reported because Q-error comparisons run against percentile columns
# published at 50/80/90 (e.g. PRICE Table 2); 90/95/99 stay the project's own
# reporting set, so this list is additive and never drops a prior key.
PERCENTILES = (50, 80, 90, 95, 99)


def q_values(log_pred, true):
    log_pred = np.asarray(log_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    if (
        not len(true)
        or true.shape != log_pred.shape
        or not np.isfinite(log_pred).all()
        or not np.isfinite(true).all()
        or (true < 0).any()
    ):
        raise ValueError("Invalid or empty predictions/labels")
    log_pred = np.maximum(log_pred, 0)
    log_true = np.log(np.maximum(true, 1))
    log_q = np.abs(log_pred - log_true)
    with np.errstate(over="ignore"):
        q = np.exp(log_q)
        shifted = np.exp(np.abs(np.logaddexp(log_pred, 0) - np.log1p(true)))
    return q, shifted, log_q


def summarize(log_pred, true):
    q, shifted, log_q = q_values(log_pred, true)
    result = {f"q{p}": float(np.percentile(q, p)) for p in PERCENTILES}
    result.update(
        {f"price_shifted_q{p}": float(np.percentile(shifted, p)) for p in PERCENTILES}
    )
    result.update(
        n=len(q), q_mean=float(q.mean()), log_q95=float(np.percentile(log_q, 95))
    )
    return result
