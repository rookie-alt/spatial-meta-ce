"""Bounded synthetic review only: no input data, optimizer, runner or GPU calls."""

import json
import math
import resource
import sys
import time
from fractions import Fraction
from unittest.mock import patch

import numpy as np

from spatial_meta_ce import s8_probe

EPS = Fraction(1, 2**52)
RHO = Fraction(1, 10**12)


def dimension_record(dimension):
    """Exact rational arithmetic for the frozen envelope, not a new gate."""
    gamma = dimension * EPS / (1 - dimension * EPS)
    return {
        "dimension": dimension,
        "gamma": float(gamma),
        "gamma_over_rcond": float(gamma / RHO),
        "zero_tail_can_be_stably_discarded": gamma < RHO * (1 - gamma),
        "fullrank_minimum_ratio_required": float(RHO + (1 + RHO) * gamma),
    }


def memory_and_flop_record(rows, columns):
    """Explicit arrays and scalar-op estimates, never a peak-RSS/ETA bound."""
    k = min(rows, columns)
    return {
        "rows": rows,
        "columns": columns,
        "A_bytes": 8 * rows * columns,
        "U_bytes": 8 * rows * k,
        "Vh_bytes": 8 * k * columns,
        "Gram_bytes": 8 * k * k,
        "backend_GEMM_scalar_flops": 4 * rows * k * k + 2 * k * k * columns,
        "standard_pair_factorizations": 3,
        "certification_factorizations": 2,
        "peak_RSS_bound": None,
        "full_P_or_server_time_estimate": None,
    }


def measured_pair(design, name):
    target = np.zeros(design.shape[0])
    dummy_adam = {
        "theta": np.zeros(design.shape[1]),
        "objective": 0.0,
        "updates": 0,
    }
    original_ls, original_svd = np.linalg.lstsq, np.linalg.svd
    calls = []

    def measured(function, kind):
        def call(*args, **kwargs):
            started = time.perf_counter()
            answer = function(*args, **kwargs)
            calls.append({"kind": kind, "seconds": time.perf_counter() - started})
            return answer

        return call

    started = time.perf_counter()
    with (
        patch.object(np.linalg, "lstsq", measured(original_ls, "lstsq")),
        patch.object(np.linalg, "svd", measured(original_svd, "svd")),
    ):
        ls = s8_probe.solve_ls(design, target)
        certificate = s8_probe.certify_pair(design, target, dummy_adam, ls)
    elapsed = time.perf_counter() - started
    assert [entry["kind"] for entry in calls] == ["lstsq", "svd", "lstsq"]
    return {
        "case": name,
        "purpose": "synthetic-only-not-research-not-real-full-P",
        "shape": list(design.shape),
        "dtype": str(design.dtype),
        "input_bytes": design.nbytes,
        "wall_seconds": elapsed,
        "factorizations": calls,
        "inferred_rank": ls["rank"],
        "certificate_pass": certificate["pass"],
        "rank_stability": certificate["rank_stability"],
        "spectrum_gamma": certificate["spectrum_gamma"],
        "backend_pass": certificate["svd_backend_pass"],
        "timing_is_not_full_P_or_server_ETA": True,
    }


def main():
    started = time.perf_counter()
    assert s8_probe.RCOND == float(RHO)
    assert np.finfo(np.float64).eps == float(EPS)
    assert dimension_record(4503)["zero_tail_can_be_stably_discarded"]
    assert not dimension_record(4504)["zero_tail_can_be_stably_discarded"]
    assert not dimension_record(19456)["zero_tail_can_be_stably_discarded"]
    records = []
    # Stored entries are exactly 0/1: the algebraic rank is exactly one.
    for rows in (4503, 4504, 19456):
        design = np.zeros((rows, 2))
        design[0, 0] = 1.0
        record = measured_pair(design, f"exact_rank_one_{rows}x2")
        assert record["inferred_rank"] == 1
        assert record["rank_stability"] == ("stable" if rows == 4503 else "unknown")
        assert record["certificate_pass"] == (rows == 4503)
        record["stored_matrix_algebraic_rank_exact"] = 1
        records.append(record)
    # Dense deterministic synthetic rotations; no RNG, labels, cache or model.
    for rows in (512, 1024, 2048):
        k = 65
        left = math.sqrt(2.0 / rows) * np.cos(
            np.pi * (np.arange(rows)[:, None] + 0.5) * np.arange(k)[None, :] / rows
        )
        left[:, 0] /= math.sqrt(2.0)
        right = math.sqrt(2.0 / k) * np.cos(
            np.pi * (np.arange(k)[:, None] + 0.5) * np.arange(k)[None, :] / k
        )
        right[:, 0] /= math.sqrt(2.0)
        singular = np.linspace(1.0, 0.5, k)
        singular[-1] = 0.0
        design = (left * singular) @ right.T
        records.append(measured_pair(design, f"dense_DCT_synthetic_{rows}x{k}"))
    try:
        from threadpoolctl import threadpool_info

        pools = threadpool_info()
        assert pools and all(pool["num_threads"] == 1 for pool in pools)
    except ImportError:
        pools = "unavailable; actual thread count unverified"
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    maxrss_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    result = {
        "purpose": "bounded-local-synthetic-rank-and-cost-review-only",
        "numpy_version": np.__version__,
        "python_executable": sys.executable,
        "threadpools": pools,
        "elapsed_seconds_excluding_imports": time.perf_counter() - started,
        "process_maxrss_bytes_including_imports_and_all_cases": maxrss_bytes,
        "exact_dimension_records": [dimension_record(d) for d in (1216, 4503, 4504, 19456)],
        "full_P_protocol_shape_costs": [memory_and_flop_record(19456, n) for n in (65, 775)],
        "synthetic_cases": records,
        "server_access": False,
        "real_P_read": False,
        "GPU_used": False,
        "optimizer_updates": 0,
        "smoke_restarted": False,
        "formal_ready": False,
        "original_G1_status": "failed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
