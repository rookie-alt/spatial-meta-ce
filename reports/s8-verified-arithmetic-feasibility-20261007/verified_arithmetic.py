"""Isolated exact-rational feasibility, not a replacement production certifier."""

import math
import resource
import sys
from fractions import Fraction as F

SCALE = 10**80
RHO = F.from_float(1e-12)
RSS_LIMIT = 128 * 2**20
INPUT_BITS_LIMIT = 4096


def budget():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        size = rss
    elif sys.platform.startswith("linux"):
        size = rss * 1024
    else:
        raise ValueError("RSS unit monitoring unavailable")
    if size > RSS_LIMIT:
        raise ValueError("RSS budget exceeded")
    return size


def exact(value):
    if isinstance(value, bool) or type(value) not in (float, int, F):
        raise TypeError("finite binary64/int/Fraction scalar required")
    if isinstance(value, (F, int)):
        result = F(value)
    else:
        if not math.isfinite(value):
            raise ValueError("nonfinite scalar")
        result = F.from_float(value)
    if (
        max(result.numerator.bit_length(), result.denominator.bit_length())
        > INPUT_BITS_LIMIT
    ):
        raise ValueError("input integer bit budget exceeded")
    return result


def sqrt_interval(value):
    budget()
    value = exact(value)
    if value < 0:
        raise ValueError("negative square root")
    k = math.isqrt(value.numerator * SCALE**2 // value.denominator)
    lower = F(k, SCALE)
    upper = lower if lower**2 == value else F(k + 1, SCALE)
    return lower, upper


def dot(a, b):
    budget()
    if len(a) != len(b):
        raise ValueError("dot shape mismatch")
    return sum((x * y for x, y in zip(a, b, strict=True)), F(0))


def matrix(value, rows=None, cols=None):
    if not isinstance(value, list) or not value or not isinstance(value[0], list):
        raise ValueError("nonempty list matrix required")
    m, n = len(value), len(value[0])
    if not 1 <= m <= 16 or not 1 <= n <= 8:
        raise ValueError("synthetic dimension budget exceeded")
    if (rows is not None and m != rows) or (cols is not None and n != cols):
        raise ValueError("matrix shape mismatch")
    if any(not isinstance(row, list) or len(row) != n for row in value):
        raise ValueError("ragged matrix")
    return [[exact(x) for x in row] for row in value]


def fro_upper(values):
    return sqrt_interval(sum((x * x for row in values for x in row), F(0)))[1]


def delta_upper(e):
    if not 0 <= e < 1:
        raise ValueError("orthogonality bound must be below one")
    return e / (1 + sqrt_interval(1 - e)[0])


def certify_rank(a, u, s, vh, reported_rank):
    """Bound true singular locations of stored A; never certify LS/subspace."""
    budget()
    a = matrix(a)
    m, n = len(a), len(a[0])
    k = min(m, n)
    u, vh = matrix(u, m, k), matrix(vh, k, n)
    if not isinstance(s, list) or len(s) != k:
        raise ValueError("complete spectrum required")
    s = [exact(x) for x in s]
    if any(x < 0 for x in s) or any(s[i] < s[i + 1] for i in range(k - 1)):
        raise ValueError("spectrum sign/order invalid")
    if isinstance(reported_rank, bool) or not isinstance(reported_rank, int):
        raise TypeError("integer rank required")
    if not 0 <= reported_rank <= k:
        raise ValueError("rank range invalid")
    residual = [
        [
            a[i][j]
            - dot([u[i][t] * s[t] for t in range(k)], [vh[t][j] for t in range(k)])
            for j in range(n)
        ]
        for i in range(m)
    ]
    gu = [
        [
            dot([u[t][i] for t in range(m)], [u[t][j] for t in range(m)]) - int(i == j)
            for j in range(k)
        ]
        for i in range(k)
    ]
    gv = [[dot(vh[i], vh[j]) - int(i == j) for j in range(k)] for i in range(k)]
    r, eu, ev = fro_upper(residual), fro_upper(gu), fro_upper(gv)
    du, dv = delta_upper(eu), delta_upper(ev)
    beta = r + s[0] * (du + dv + du * dv)
    intervals = [(max(F(0), x - beta), x + beta) for x in s]
    cutoff = (RHO * intervals[0][0], RHO * intervals[0][1])
    keep = [lo > cutoff[1] for lo, _ in intervals]
    discard = [hi < cutoff[0] for _, hi in intervals]
    stable = all(x or y for x, y in zip(keep, discard, strict=True))
    if all(x == 0 for row in a for x in row) and all(x == 0 for x in s):
        stable = True
    rank = sum(keep)
    budget()
    return {
        "rank_status": "verified" if stable and rank == reported_rank else "unknown",
        "certification_object": "stored_A_true_singular_locations_rank_only",
        "LS_or_subspace_or_G1_certified": False,
        "verified_rank": rank if stable else None,
        "beta_upper_exact_rational": str(beta),
        "r_upper": str(r),
        "eU_upper": str(eu),
        "eV_upper": str(ev),
        "cutoff_interval": [str(x) for x in cutoff],
        "singular_intervals": [[str(x) for x in bounds] for bounds in intervals],
    }
