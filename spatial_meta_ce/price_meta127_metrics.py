"""Official PRICE Q-error metrics for the 127 re-evaluation.

The upstream ``utils/model/qerror.py`` of the pinned PRICE commit computes the
Q-error in three steps that are reproduced literally here:

1. the encoded outputs and the encoded labels are exponentiated in float32
   (``np.exp(x - 1) - 1``);
2. the decoded cardinalities are widened to float64 and shifted by
   ``min_val = 1.0``;
3. the Q-error is ``max(label / pred, pred / label)`` and ``np.nan_to_num`` is
   applied with ``nan=np.inf``, so a NaN ratio stays ``+inf`` while a genuine
   ``+inf`` ratio is mapped to the float64 maximum.

Two details are load bearing and must not be "simplified":

* the exponentiation happens in float32, so the float32 rounding of
  ``exp``-then-subtract is observable in the result;
* upstream applies **no clipping** to the exponent, therefore extreme
  predictions overflow to ``+inf`` instead of being bounded.

The module keeps the 126-protocol metric (``standard_raw_qerrors``) as a
companion so the difference between the two statistics stays measurable
instead of being silently replaced.
"""

from __future__ import annotations

import numpy as np

OFFICIAL_PERCENTILES = (50, 80, 90, 95, 99)
OFFICIAL_ROUNDING = 4
OFFICIAL_MIN_VAL = 1.0
STANDARD_PERCENTILES = (50, 90, 95, 99)


def _as_vector(values, name):
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim != 1:
        array = array.reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.dtype == object or array.dtype.kind not in "fiub":
        raise ValueError(f"{name} must be numeric")
    try:
        array = array.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_float32(values, name):
    """Cast to float32 the way the upstream float32 tensors do.

    A finite float64 input that is out of the float32 range becomes ``inf``,
    exactly as it would in the upstream float32 tensor; it is not an input
    error and it is not clipped.
    """
    array = _as_vector(values, name)
    with np.errstate(over="ignore"):
        cast = array.astype(np.float32)
    return cast


def _paired(encoded, encoded_labels):
    predictions = _as_vector(encoded, "encoded")
    labels = _as_vector(encoded_labels, "encoded_labels")
    if predictions.shape != labels.shape:
        raise ValueError(
            f"encoded and encoded_labels must have the same length: "
            f"{predictions.shape} != {labels.shape}"
        )
    return predictions, labels


def decode_official(encoded):
    """Decode encoded outputs exactly as the upstream float32 path does."""
    predictions = _as_float32(encoded, "encoded")
    with np.errstate(over="ignore", under="ignore"):
        decoded = np.exp(predictions - np.float32(1.0)) - np.float32(1.0)
    return decoded.astype(np.float64)


def encode_labels_official(true_cards):
    """Encode true cardinalities as upstream does: ``log(x + 1) + 1``."""
    values = _as_float32(true_cards, "true_cards")
    if (values < 0).any():
        raise ValueError("true_cards must not be negative")
    with np.errstate(over="ignore", under="ignore"):
        encoded = np.log(values + np.float32(1.0)) + np.float32(1.0)
    return encoded.astype(np.float32)


def official_qerrors(encoded, encoded_labels):
    """Per-query official Q-errors, computed exactly like the pinned upstream.

    The arithmetic is float32 up to the ``astype(np.float64)`` widening, no
    clipping is applied and ``np.nan_to_num(..., nan=np.inf)`` keeps the
    upstream mapping of ``+inf`` onto the float64 maximum.

    Only non-finite *inputs* are rejected.  A finite input outside the
    float32 range overflows to ``inf`` the way the upstream float32 tensor
    would, and the resulting non-finite Q-error is reported by
    :func:`summarize_official` instead of being hidden.
    """
    predictions = _as_float32(encoded, "encoded")
    labels = _as_float32(encoded_labels, "encoded_labels")
    if predictions.shape != labels.shape:
        raise ValueError(
            f"encoded and encoded_labels must have the same length: "
            f"{predictions.shape} != {labels.shape}"
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        preds = np.exp(predictions - np.float32(1.0)) - np.float32(1.0)
        labs = np.exp(labels - np.float32(1.0)) - np.float32(1.0)
        preds = preds.astype(np.float64) + OFFICIAL_MIN_VAL
        labs = labs.astype(np.float64) + OFFICIAL_MIN_VAL
        q_errors = np.maximum(labs / preds, preds / labs)
        q_errors = np.nan_to_num(q_errors, nan=np.inf)
    return np.asarray(q_errors, dtype=np.float64)


def decode_standard_log_prediction(encoded):
    """The 126-protocol decode: ``log(max(exp(a) - 1, 1))`` in float64."""
    values = _as_vector(encoded, "encoded")
    a = values - 1.0
    result = np.zeros_like(a)
    positive = a > np.log(2)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result[positive] = a[positive] + np.log1p(-np.exp(-a[positive]))
    return result


def standard_raw_qerrors(encoded, true_cards):
    """Uncapped per-query Q-errors of the 126 protocol, in float64.

    ``exp(abs(max(log_pred, 0) - log(max(true, 1))))``.  Unlike the official
    statistic this one is *not* clipped, so an extreme prediction overflows to
    ``+inf`` and the caller has to decide what to do with it.
    """
    predictions = _as_vector(encoded, "encoded")
    truth = _as_vector(true_cards, "true_cards")
    if predictions.shape != truth.shape:
        raise ValueError(
            f"encoded and true_cards must have the same length: "
            f"{predictions.shape} != {truth.shape}"
        )
    if (truth < 0).any():
        raise ValueError("true_cards must not be negative")
    log_prediction = decode_standard_log_prediction(predictions)
    log_true = np.log(np.maximum(truth, 1.0))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.exp(np.abs(np.maximum(log_prediction, 0.0) - log_true))


def _percentiles(q_errors, percentiles, rounding=OFFICIAL_ROUNDING):
    """Raw upstream percentiles; non-finite values are kept as they are.

    The rounding reproduces the upstream ``round(qerror, 4)`` on a numpy
    float64, which overflows to ``+inf`` for magnitudes above roughly
    ``1e15`` because numpy implements the rounding with a multiplication.
    The Python built-in ``round`` does not overflow, so the numpy value must
    be rounded directly to stay bit-compatible with the upstream output.
    """
    result = {}
    for percentile in percentiles:
        with np.errstate(invalid="ignore"):
            value = float(np.percentile(q_errors, percentile, method="linear"))
        result[f"q{percentile}"] = _upstream_round(value, rounding)
    return result


def _upstream_round(value, rounding=OFFICIAL_ROUNDING):
    """``round()`` applied to a numpy float64, exactly like upstream."""
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.float64(value).round(rounding)
    return float(rounded)


def _json_safe_float(value, rounding=OFFICIAL_ROUNDING):
    """A JSON-safe float; a non-finite result becomes ``None``."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        rounded = np.float64(value).round(rounding)
    if not np.isfinite(rounded):
        return None
    return float(rounded)


def _json_safe_raw_percentiles(q_errors, percentiles):
    """Percentiles without the upstream rounding, JSON safe.

    A non-finite percentile becomes ``None``; the caller reports which keys
    were affected through ``metadata.nonfinite_percentiles``.
    """
    result = {}
    for percentile in percentiles:
        with np.errstate(invalid="ignore"):
            value = float(np.percentile(q_errors, percentile, method="linear"))
        result[f"q{percentile}"] = value if np.isfinite(value) else None
    return result


def _decode_diagnostics(predictions, labels):
    """Diagnostics of the upstream decode, before the Q-error is formed.

    The upstream never clips, so the float32 exponent can overflow to
    ``+inf`` and the widened cardinality can be zero.  Both cases produce a
    degenerate ratio that the upstream ``nan_to_num`` call quietly repairs
    (``+inf`` becomes the float64 maximum, ``NaN`` becomes ``+inf``), so they
    have to be counted explicitly to keep the official number honest.
    """
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        decoded_predictions = np.exp(predictions - np.float32(1.0)) - np.float32(1.0)
        decoded_labels = np.exp(labels - np.float32(1.0)) - np.float32(1.0)
    decoded_predictions = decoded_predictions.astype(np.float64) + OFFICIAL_MIN_VAL
    decoded_labels = decoded_labels.astype(np.float64) + OFFICIAL_MIN_VAL

    prediction_overflow = int(np.sum(~np.isfinite(decoded_predictions)))
    label_overflow = int(np.sum(~np.isfinite(decoded_labels)))
    zero_denominator = int(
        np.sum((decoded_predictions == 0.0) | (decoded_labels == 0.0))
    )
    nonfinite_decoded = prediction_overflow + label_overflow
    return {
        "n_overflow": nonfinite_decoded,
        "n_prediction_overflow": prediction_overflow,
        "n_label_overflow": label_overflow,
        "n_zero_denominator": zero_denominator,
        "n_nonfinite_decoded": nonfinite_decoded,
        "note": (
            "the upstream applies no exponent clipping; an overflowing "
            "prediction decodes to +inf and a vanishing one to 0"
        ),
    }


def official_percentiles(encoded, encoded_labels):
    """The exact upstream percentile outputs, including ``NaN``/``inf``.

    ``numpy`` returns ``NaN`` for a percentile of a sample that contains
    ``+inf``; upstream does not repair that.  Callers that need JSON use
    :func:`summarize_official`, which reports the same numbers but keeps the
    payload JSON-safe.
    """
    return _percentiles(official_qerrors(encoded, encoded_labels), OFFICIAL_PERCENTILES)


def _json_safe_percentiles(q_errors, percentiles, rounding=OFFICIAL_ROUNDING):
    """Percentiles that are safe to serialise, plus the non-finite key list."""
    raw = _percentiles(q_errors, percentiles, rounding=rounding)
    safe = {}
    nonfinite = []
    for key, value in raw.items():
        if np.isfinite(value):
            safe[key] = value
        else:
            safe[key] = None
            nonfinite.append(key)
    return safe, nonfinite


def summarize_official(encoded, encoded_labels, true_cards=None):
    """Official percentiles plus the diagnostics the runtime needs.

    The percentiles are the upstream ones (``numpy`` linear percentile, rounded
    to four decimals).  Because upstream never clips, a single overflowing
    prediction can make ``numpy`` return ``NaN`` for a percentile; that is
    reported through the metadata instead of being silently repaired.

    ``unrounded`` carries the same percentiles *before* the upstream rounding
    so a downstream consumer can keep full precision; the top-level ``q*``
    keys stay the upstream display values.  The unrounded mapping is JSON safe:
    a non-finite percentile becomes ``None`` and is listed in
    ``metadata.nonfinite_percentiles``.
    """
    q_errors = official_qerrors(encoded, encoded_labels)
    finite = q_errors[np.isfinite(q_errors)]
    infinite = int(np.sum(np.isinf(q_errors)))
    nan_count = int(np.sum(np.isnan(q_errors)))
    predictions = _as_float32(encoded, "encoded")
    labels = _as_float32(encoded_labels, "encoded_labels")
    float32_overflow = int(
        np.sum(~np.isfinite(predictions)) + np.sum(~np.isfinite(labels))
    )
    decode = _decode_diagnostics(predictions, labels)

    summary, nonfinite_percentiles = _json_safe_percentiles(
        q_errors, OFFICIAL_PERCENTILES
    )
    percentiles_finite = not nonfinite_percentiles
    unrounded = _json_safe_raw_percentiles(q_errors, OFFICIAL_PERCENTILES)

    metadata = {
        "n": int(q_errors.size),
        "n_finite": int(finite.size),
        "nonfinite_q": {
            "inf": infinite,
            "nan": nan_count,
        },
        "float32_overflow_inputs": float32_overflow,
        "decode": decode,
        "nonfinite_percentiles": nonfinite_percentiles,
        "percentiles_finite": bool(percentiles_finite),
        "official_comparable": bool(
            infinite == 0
            and nan_count == 0
            and decode["n_overflow"] == 0
            and decode["n_zero_denominator"] == 0
            and decode["n_nonfinite_decoded"] == 0
        ),
        "clipping": "none; the pinned upstream applies no exponent clipping",
        "arithmetic": "float32 exp and subtraction, widened to float64 afterwards",
        "nan_to_num": (
            "nan=np.inf; a genuine +inf ratio is mapped onto the float64 "
            "maximum, so it is finite and not counted as non-finite"
        ),
        "percentiles": list(OFFICIAL_PERCENTILES),
        "percentile_method": "numpy linear interpolation",
        "rounding": OFFICIAL_ROUNDING,
        "mean_finite": _json_safe_float(np.mean(finite)) if finite.size else None,
        "max_finite": _json_safe_float(np.max(finite)) if finite.size else None,
    }
    summary["n"] = int(q_errors.size)
    summary["unrounded"] = unrounded
    summary["metadata"] = metadata

    if true_cards is not None:
        standard = standard_raw_qerrors(encoded, true_cards=true_cards)
        with np.errstate(invalid="ignore"):
            difference = np.abs(q_errors - standard)
        finite_difference = difference[np.isfinite(difference)]
        standard_percentiles, standard_nonfinite = _json_safe_percentiles(
            standard, STANDARD_PERCENTILES
        )
        summary["standard_companion"] = {
            "n": int(standard.size),
            "definition": (
                "exp(abs(max(log_pred, 0) - log(max(true, 1)))) with the "
                "126-protocol decode; uncapped"
            ),
            "percentiles": standard_percentiles,
            "nonfinite_percentiles": standard_nonfinite,
            "mean_finite": _json_safe_float(np.mean(standard[np.isfinite(standard)]))
            if np.isfinite(standard).any()
            else None,
            "n_nonfinite": int(np.sum(~np.isfinite(standard))),
            "n_differing": int(np.sum(difference > 1e-12)),
            "max_abs_difference": float(np.max(finite_difference))
            if finite_difference.size
            else 0.0,
        }
    return summary
