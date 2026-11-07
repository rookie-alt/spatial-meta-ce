"""Structural summary: shared-bin key histograms and the C_struct closed form.

This is the numeric core behind doc 118's structural anchor.  It is pure
statistics: no SQL parsing, no database access, no learned parameters.  The
caller supplies, per query, the join-column histograms and the per-table
selectivity; everything here is arithmetic over those.

`C_struct` is the closed-form structural estimator `C_uncond` with every join
edge bucketed on shared equal-width boundaries:

    C_struct = prod_tables (kept/total) / total ** (distinct - 1)
             * prod_variables sum_bin prod_members count / width ** (k - 1)

The second product is the join-size identity: along one join variable, each
member's key distribution is multiplied and summed.  A bin assumes uniformity
inside itself, which is the only thing a histogram can promise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_BINS = 160


@dataclass(frozen=True)
class ColumnHist:
    """Equal-width histogram of one join column."""

    edges: np.ndarray  # length bins + 1, value-domain bounds
    counts: np.ndarray  # length bins, row counts (NULLs excluded)
    total: int  # all rows of the table
    nonnull: int  # rows whose key is not NULL
    kind: str  # 'int' | 'float' | 'str'

    def rebucket(self, dst_edges: np.ndarray) -> np.ndarray:
        """Redistribute counts onto `dst_edges`, uniform inside a source bin."""
        out = np.zeros(len(dst_edges) - 1, dtype=np.float64)
        if len(self.edges) < 2 or not self.counts.any():
            return out
        src = np.asarray(self.edges, dtype=np.float64)
        dst = np.asarray(dst_edges, dtype=np.float64)
        for index, weight in enumerate(self.counts):
            if not weight:
                continue
            low, high = src[index], src[index + 1]
            if high <= low:
                # A constant column has a zero-width domain; its whole count
                # belongs to the target bin that contains that single value.
                # A value outside the shared domain joins with nothing.
                if dst[0] <= low <= dst[-1]:
                    where = int(np.searchsorted(dst, low, side="right")) - 1
                    out[min(max(where, 0), len(out) - 1)] += weight
                continue
            first = max(int(np.searchsorted(dst, low, side="right") - 1), 0)
            last = min(int(np.searchsorted(dst, high, side="left") - 1), len(out) - 1)
            if self.kind == "int":
                # Integer keys are counted, not measured.  A source bin may
                # straddle a target boundary when the source is finer than the
                # integer span; geometric splitting would leak its mass onto a
                # key it does not contain, so allocate by integer-key overlap.
                first_key = int(np.ceil(low))
                last_key = (
                    int(np.floor(high))
                    if index == len(self.counts) - 1
                    else int(np.floor(high - 1e-9))
                )
                keys = max(last_key - first_key + 1, 1)
                for bin_index in range(first, last + 1):
                    low_key = max(first_key, int(np.ceil(dst[bin_index])))
                    high_key = (
                        int(np.floor(dst[bin_index + 1]))
                        if bin_index == len(out) - 1
                        else int(np.floor(dst[bin_index + 1] - 1e-9))
                    )
                    hits = max(min(last_key, high_key) - low_key + 1, 0)
                    if hits:
                        out[bin_index] += weight * hits / keys
                continue
            for bin_index in range(first, last + 1):
                overlap = min(high, dst[bin_index + 1]) - max(low, dst[bin_index])
                if overlap > 0:
                    out[bin_index] += weight * overlap / (high - low)
        return out


def _numeric_hist(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    low, high = float(np.min(values)), float(np.max(values))
    if high <= low:
        return np.asarray([low, low], dtype=np.float64), np.asarray(
            [float(len(values))]
        )
    edges = np.linspace(low, high, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return edges, counts.astype(np.float64)


def column_kind(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series.dtype):
        return "int"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "float"
    return "str"


def equal_width_hist(
    series: pd.Series, bins: int = DEFAULT_BINS, kind: str | None = None
) -> ColumnHist:
    """Build a `bins`-wide equal-domain histogram for one join column."""
    kind = kind or column_kind(series)
    total = int(len(series))
    if kind == "str":
        values = series.dropna().astype(str).to_numpy()
        if not len(values):
            return ColumnHist(np.zeros(bins + 1), np.zeros(bins), total, 0, kind)
        # Rank-encode strings so equal values land on the same coordinate on
        # both sides of a join edge; the domain is lexicographic.
        order = {value: index for index, value in enumerate(np.unique(values))}
        numeric = np.asarray([order[v] for v in values], dtype=np.float64)
        edges, counts = _numeric_hist(numeric, bins)
        return ColumnHist(edges, counts, total, len(values), kind)
    numeric = pd.to_numeric(series.dropna(), errors="coerce").dropna()
    values = numeric.to_numpy(dtype=np.float64)
    if not len(values):
        return ColumnHist(np.zeros(bins + 1), np.zeros(bins), total, 0, kind)
    edges, counts = _numeric_hist(values, bins)
    return ColumnHist(edges, counts, total, len(values), kind)


def shared_edges(
    hists: list[ColumnHist], bins: int = DEFAULT_BINS, kind: str = "float"
) -> np.ndarray:
    """Union value range of every member of one join variable.

    An integer domain never gets more bins than it has distinct values, so a
    tiny domain cannot be split into sub-integer slivers.
    """
    live = [h for h in hists if len(h.edges) > 1 and h.nonnull]
    if not live:
        return np.linspace(0.0, 1.0, bins + 1)
    wide = [h for h in live if float(h.edges[-1]) > float(h.edges[0])]
    if not wide:
        # Every member is a constant column; the domain is that single value.
        low = min(float(h.edges[0]) for h in live)
        return np.asarray([low, low], dtype=np.float64)
    low = min(float(h.edges[0]) for h in wide)
    high = max(float(h.edges[-1]) for h in wide)
    count = bins
    if kind == "int":
        span = int(round(high - low))
        if span >= 1:
            count = min(bins, span)
    return np.linspace(low, high, count + 1)


def _bin_widths(edges: np.ndarray, kind: str) -> np.ndarray:
    """Effective number of keys per bin.

    Integer domains count distinct values (the last bin is closed on the right),
    which is what `sum_k` actually ranges over when keys are integers.
    """
    edges = np.asarray(edges, dtype=np.float64)
    if kind == "int":
        upper = edges[1:].copy()
        upper[:-1] = np.nextafter(upper[:-1], -np.inf)  # open on the right
        widths = np.floor(upper) - np.ceil(edges[:-1]) + 1
        return np.maximum(widths, 1.0)
    widths = np.diff(edges)
    widths[widths <= 0] = 1.0
    return widths


def bucket_join_sum(hists: list[ColumnHist], bins: int = DEFAULT_BINS) -> float:
    """`sum_k prod_i marginal_i(k)` estimated on shared equal-width bins.

    Inside a bin every member is uniform, so the product integrates to
    `prod_i count_i / width ** (len(live) - 1)`.
    """
    live = list(hists)
    if not live:
        return 0.0
    if any(h.nonnull == 0 for h in live):
        # A join variable with a member that has no non-NULL key joins nothing.
        return 0.0
    kinds = {h.kind for h in live}
    if len(kinds) > 1:
        raise ValueError(f"Mixed join-key kinds: {sorted(kinds)}")
    edges = shared_edges(live, bins, live[0].kind)
    bucketed = [h.rebucket(edges) for h in live]
    widths = _bin_widths(edges, live[0].kind)
    weight = np.ones_like(widths)
    for counts in bucketed:
        weight *= counts
    weight /= widths ** max(len(live) - 1, 0)
    return float(weight.sum())


def variable_classes(
    predicates: tuple[tuple[str, str, str, str], ...], tables: tuple[str, ...]
) -> tuple[dict[int, tuple[tuple[str, str], ...]], dict[str, list[int]]]:
    """Union-find over join predicates; predicates sharing table+column merge.

    Returns the merged variable classes and, per table alias, which class roots
    touch it.  Two predicates on the same hub column therefore stay one
    variable, which keeps the repeated-middle-column shape correct.
    """
    parent = list(range(len(predicates)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owner: dict[tuple[str, str], int] = {}
    for index, (alias_a, col_a, alias_b, col_b) in enumerate(predicates):
        for key in ((alias_a, col_a), (alias_b, col_b)):
            if key in owner:
                root_a, root_b = find(owner[key]), find(index)
                if root_a != root_b:
                    parent[root_a] = root_b
            else:
                owner[key] = index
    classes: dict[int, list[tuple[str, str]]] = {}
    for (alias, column), index in owner.items():
        classes.setdefault(find(index), []).append((alias, column))
    table_vars: dict[str, list[int]] = {alias: [] for alias in tables}
    for root, members in classes.items():
        for alias, _ in members:
            if root not in table_vars[alias]:
                table_vars[alias].append(root)
    for alias in table_vars:
        table_vars[alias].sort()
    return {root: tuple(sorted(m)) for root, m in classes.items()}, table_vars


def c_struct(
    tables: tuple[str, ...],
    predicates: tuple[tuple[str, str, str, str], ...],
    hists: dict[tuple[str, str], ColumnHist],
    selectivities: dict[str, float],
    total_rows: dict[str, int],
    bins: int = DEFAULT_BINS,
) -> float:
    """The closed-form structural estimate for one query.

    `selectivities[alias]` is the fraction of the table surviving its filters;
    `hists[(alias, column)]` must cover every join column the query touches.
    """
    classes, table_vars = variable_classes(predicates, tables)
    scale = 1.0
    for alias in tables:
        total = total_rows[alias]
        if total == 0:
            return 0.0
        kept = selectivities.get(alias, 1.0) * total
        if kept <= 0:
            return 0.0
        distinct = len(table_vars[alias])
        scale *= (kept / total) / (total ** max(distinct - 1, 0))
    product = 1.0
    for members in classes.values():
        product *= bucket_join_sum([hists[key] for key in members], bins)
    return scale * product
