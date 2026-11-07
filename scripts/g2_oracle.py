"""G2a: three-table path conditional key-frequency oracle.

Read-only feasibility check for `实验文档/116-G2三表链条件连接键质检协议.md`.
It never trains, never loads a model and never touches a target database.

For every eligible three-table path query it scores four estimators that vary
one factor at a time:

                 middle keys independent        middle keys joint
    unfiltered   U_indep                        U_joint
    filtered     F_indep                        F_joint  (= exact)

    Q1 (conditioning) = F_indep - U_indep
    Q2 (correlation)  = F_joint - F_indep

Counts are over the *join population* only: rows whose keys are non-NULL.
`M1/M2/M3` are the unfiltered populations, `m1/m2/m3` the filtered ones, and
`s_i = m_i / M_i`.  See protocol §5 for the zero-value branch.

    python scripts/g2_oracle.py plan --workload-dir ... --out DIR
    python scripts/g2_oracle.py run  --plan DIR/plan.json --dataset-dir ... --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from g1_oracle import (  # noqa: E402
    MAIN_DATABASES,
    TARGET_DATABASES,
    Predicate,
    Unsupported,
    _conjuncts,
    _predicate,
    _strip,
    _target_dtype,
    apply_filters,
    evenly_spaced,
    key_counts,
    load_tables,
    parse_ddl,
    percentile,
    q_error,
    write_json,
)
from spatial_meta_ce.manifests import file_hash, read_workload  # noqa: E402

DEFAULT_MAX_PER_DB = 200
MIN_RECHECKS = 20
ZERO_POPULATION_BUCKET = "zero_join_population"
OPERATORS = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
}


@dataclass(frozen=True)
class Spec3:
    left_table: str
    left_alias: str
    left_key: str
    middle_table: str
    middle_alias: str
    mid_key_left: str
    mid_key_right: str
    right_table: str
    right_alias: str
    right_key: str
    filters: tuple[Predicate, ...]

    def for_alias(self, alias: str) -> list[Predicate]:
        return [p for p in self.filters if p.alias == alias]

    def filters_join_key(self) -> bool:
        pairs = {
            (self.left_alias, self.left_key),
            (self.middle_alias, self.mid_key_left),
            (self.middle_alias, self.mid_key_right),
            (self.right_alias, self.right_key),
        }
        return any((p.alias, p.column) in pairs for p in self.filters)


def analyze3(sql: str) -> Spec3:
    """Return the frozen-scope three-table path spec or raise `Unsupported`."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001 - any parse failure is an unsupported row
        raise Unsupported("parse_error") from None
    if not isinstance(tree, exp.Select):
        raise Unsupported("not_select")
    if tree.args.get("with"):
        raise Unsupported("with_clause")
    if list(tree.find_all(exp.Union)):
        raise Unsupported("union")
    if list(tree.find_all(exp.Subquery)):
        raise Unsupported("subquery")
    if (
        tree.args.get("group")
        or tree.args.get("order")
        or tree.args.get("limit")
        or tree.args.get("having")
        or tree.args.get("qualify")
    ):
        raise Unsupported("group_order_limit")
    if len(tree.expressions) != 1:
        raise Unsupported("not_single_projection")
    projection = tree.expressions[0]
    if not isinstance(projection, exp.Count) or not isinstance(
        projection.this, exp.Star
    ):
        raise Unsupported("not_count_star")

    tables = list(tree.find_all(exp.Table))
    if len(tables) != 3:
        raise Unsupported(f"table_count_{len(tables)}")
    if len({t.name for t in tables}) != 3:
        raise Unsupported("self_join")
    alias = {t.alias_or_name: t.name for t in tables}
    if len(alias) != 3:
        raise Unsupported("duplicate_alias")

    explicit = [
        join
        for join in (tree.args.get("joins") or [])
        if join.args.get("on") is not None
        or join.args.get("kind")
        or join.args.get("side")
    ]
    sides = {str(join.args.get("side") or "").upper() for join in explicit}
    kinds = {str(join.args.get("kind") or "").upper() for join in explicit}
    if (sides - {""}) or (kinds - {"", "INNER", "CROSS"}):
        raise Unsupported("outer_join")

    conjuncts: list[exp.Expression] = []
    for join in explicit:
        on = join.args.get("on")
        if on is not None:
            conjuncts.extend(_conjuncts(on))
    where = tree.args.get("where")
    if where is not None:
        conjuncts.extend(_conjuncts(where.this))

    joins, filters, reasons = [], [], []
    for raw_node in conjuncts:
        node = _strip(raw_node)
        if (
            isinstance(node, exp.EQ)
            and isinstance(node.this, exp.Column)
            and isinstance(node.expression, exp.Column)
        ):
            joins.append(node)
        elif type(node) in OPERATORS:
            predicate = _predicate(node)
            if predicate is None:
                reasons.append("non_literal_filter")
            else:
                filters.append(predicate)
        else:
            reasons.append(type(node).__name__.lower())
    if reasons:
        raise Unsupported("predicate_" + sorted(reasons)[0])
    if len(joins) != 2:
        raise Unsupported(f"join_predicate_count_{len(joins)}")

    degree: Counter = Counter()
    edges = []
    for join in joins:
        left, right = join.this, join.expression
        if not left.table or not right.table:
            raise Unsupported("unqualified_join_column")
        if left.table == right.table:
            raise Unsupported("same_table_equality")
        if left.table not in alias or right.table not in alias:
            raise Unsupported("join_alias_mismatch")
        degree[left.table] += 1
        degree[right.table] += 1
        edges.append((left.table, left.name, right.table, right.name))
    if len(degree) != 3 or sorted(degree.values()) != [1, 1, 2]:
        raise Unsupported("not_a_path")
    middle = next(node for node, count in degree.items() if count == 2)

    endpoints: dict[str, tuple[str, str]] = {}
    for alias_a, col_a, alias_b, col_b in edges:
        if alias_a == middle:
            endpoints[alias_b] = (col_a, col_b)
        elif alias_b == middle:
            endpoints[alias_a] = (col_b, col_a)
        else:
            raise Unsupported("not_a_path")
    if len(endpoints) != 2:
        raise Unsupported("not_a_path")
    # Orient by table name, not alias, so the same query written with different
    # aliases still yields the same left/right assignment.
    first, second = sorted(endpoints, key=lambda node: alias[node])
    mid_key_left, left_key = endpoints[first]
    mid_key_right, right_key = endpoints[second]
    if mid_key_left == mid_key_right:
        raise Unsupported("same_middle_key")
    for predicate in filters:
        if predicate.alias not in alias:
            raise Unsupported("filter_alias_mismatch")

    return Spec3(
        left_table=alias[first],
        left_alias=first,
        left_key=left_key,
        middle_table=alias[middle],
        middle_alias=middle,
        mid_key_left=mid_key_left,
        mid_key_right=mid_key_right,
        right_table=alias[second],
        right_alias=second,
        right_key=right_key,
        filters=tuple(filters),
    )


def _normaliser(dtype: object):
    if _target_dtype(dtype, dtype) == "string":
        return str
    return float


def pair_counts(frame: pd.DataFrame, key_a: str, key_b: str) -> pd.Series:
    """Counts over rows where BOTH keys are non-NULL; NULL keys never join."""
    subset = frame[[key_a, key_b]].dropna()
    if subset.empty:
        return pd.Series(dtype="int64")
    return subset.groupby([key_a, key_b], sort=False).size()


def _as_dict(counts: pd.Series, normalise) -> dict:
    return {normalise(key): float(value) for key, value in counts.items()}


def _pair_dict(counts: pd.Series, normalise_a, normalise_b) -> dict:
    return {
        (normalise_a(pair[0]), normalise_b(pair[1])): float(value)
        for pair, value in counts.items()
    }


def _discrepancy(exact: int, truth: float) -> float:
    if truth == 0:
        return 0.0 if exact == 0 else 1.0
    return abs(exact - truth) / truth


def score_query3(
    spec: Spec3, tables: dict[str, pd.DataFrame], truth: float
) -> dict[str, object]:
    left = tables[spec.left_table]
    middle = tables[spec.middle_table]
    right = tables[spec.right_table]

    left_f = apply_filters(left, spec.for_alias(spec.left_alias))
    mid_f = apply_filters(middle, spec.for_alias(spec.middle_alias))
    right_f = apply_filters(right, spec.for_alias(spec.right_alias))

    phi1 = key_counts(left_f, spec.left_key)
    phi3 = key_counts(right_f, spec.right_key)
    phi2 = pair_counts(mid_f, spec.mid_key_left, spec.mid_key_right)
    c1 = key_counts(left, spec.left_key)
    c3 = key_counts(right, spec.right_key)
    c2 = pair_counts(middle, spec.mid_key_left, spec.mid_key_right)

    M1, M2, M3 = int(c1.sum()), int(c2.sum()), int(c3.sum())
    m1, m2, m3 = int(phi1.sum()), int(phi2.sum()), int(phi3.sum())
    base = {"M1": M1, "M2": M2, "M3": M3, "m1": m1, "m2": m2, "m3": m3}

    zero_population = min(M1, M2, M3) == 0
    if zero_population or min(m1, m2, m3) == 0:
        return {
            **base,
            "truth": truth,
            "sel_left": float(m1 / M1) if M1 else float("nan"),
            "sel_middle": float(m2 / M2) if M2 else float("nan"),
            "sel_right": float(m3 / M3) if M3 else float("nan"),
            "u_indep": 0.0,
            "u_joint": 0.0,
            "f_indep": 0.0,
            "f_joint": 0.0,
            "exact": 0,
            "zero_population": zero_population,
            "q_u_indep": q_error(0.0, truth),
            "q_u_joint": q_error(0.0, truth),
            "q_f_indep": q_error(0.0, truth),
            "q_f_joint": q_error(0.0, truth),
            "discrepancy": _discrepancy(0, truth),
        }

    normalise_left = _normaliser(left[spec.left_key].dtype)
    normalise_right = _normaliser(right[spec.right_key].dtype)
    normalise_mid_left = _normaliser(middle[spec.mid_key_left].dtype)
    normalise_mid_right = _normaliser(middle[spec.mid_key_right].dtype)

    phi1_d = _as_dict(phi1, normalise_left)
    phi3_d = _as_dict(phi3, normalise_right)
    phi2_d = _pair_dict(phi2, normalise_mid_left, normalise_mid_right)
    c1_d = _as_dict(c1, normalise_left)
    c3_d = _as_dict(c3, normalise_right)
    c2_d = _pair_dict(c2, normalise_mid_left, normalise_mid_right)

    # Middle-table marginals, both summed from the same joint counts.
    phi2a: dict = {}
    phi2b: dict = {}
    for (key_a, key_b), value in phi2_d.items():
        phi2a[key_a] = phi2a.get(key_a, 0.0) + value
        phi2b[key_b] = phi2b.get(key_b, 0.0) + value
    c2a: dict = {}
    c2b: dict = {}
    for (key_a, key_b), value in c2_d.items():
        c2a[key_a] = c2a.get(key_a, 0.0) + value
        c2b[key_b] = c2b.get(key_b, 0.0) + value

    f_joint = sum(
        value * phi1_d.get(key_a, 0.0) * phi3_d.get(key_b, 0.0)
        for (key_a, key_b), value in phi2_d.items()
    )
    f_indep = (
        sum(phi1_d.get(key, 0.0) * value for key, value in phi2a.items())
        * sum(value * phi3_d.get(key, 0.0) for key, value in phi2b.items())
        / m2
    )

    # Row-level pass: algebraically equal to `f_joint` but a different code path.
    mid_keys = mid_f[[spec.mid_key_left, spec.mid_key_right]].dropna()
    if len(mid_keys):
        left_contrib = mid_keys[spec.mid_key_left].map(
            lambda value: phi1_d.get(normalise_mid_left(value), 0.0)
        )
        right_contrib = mid_keys[spec.mid_key_right].map(
            lambda value: phi3_d.get(normalise_mid_right(value), 0.0)
        )
        exact = float((left_contrib * right_contrib).sum())
    else:
        exact = 0.0

    s1, s2, s3 = m1 / M1, m2 / M2, m3 / M3
    u_joint = (
        s1
        * s2
        * s3
        * sum(
            value * c1_d.get(key_a, 0.0) * c3_d.get(key_b, 0.0)
            for (key_a, key_b), value in c2_d.items()
        )
    )
    u_indep = (
        s1
        * s2
        * s3
        * sum(c1_d.get(key, 0.0) * value for key, value in c2a.items())
        * sum(value * c3_d.get(key, 0.0) for key, value in c2b.items())
        / M2
    )

    return {
        **base,
        "truth": truth,
        "sel_left": s1,
        "sel_middle": s2,
        "sel_right": s3,
        "u_indep": u_indep,
        "u_joint": u_joint,
        "f_indep": f_indep,
        "f_joint": f_joint,
        "exact": int(round(exact)),
        "zero_population": False,
        "q_u_indep": q_error(u_indep, truth),
        "q_u_joint": q_error(u_joint, truth),
        "q_f_indep": q_error(f_indep, truth),
        "q_f_joint": q_error(f_joint, truth),
        "discrepancy": _discrepancy(int(round(exact)), truth),
    }


def independent_recheck3(
    spec: Spec3, tables: dict[str, pd.DataFrame], reported: dict
) -> None:
    """Recompute `f_joint` with plain Python counters; no pandas groupby."""
    left = apply_filters(tables[spec.left_table], spec.for_alias(spec.left_alias))
    middle = apply_filters(tables[spec.middle_table], spec.for_alias(spec.middle_alias))
    right = apply_filters(tables[spec.right_table], spec.for_alias(spec.right_alias))
    normalise_left = _normaliser(tables[spec.left_table][spec.left_key].dtype)
    normalise_right = _normaliser(tables[spec.right_table][spec.right_key].dtype)
    normalise_mid_left = _normaliser(tables[spec.middle_table][spec.mid_key_left].dtype)
    normalise_mid_right = _normaliser(
        tables[spec.middle_table][spec.mid_key_right].dtype
    )

    def tally(series: pd.Series) -> dict:
        counts: Counter = Counter()
        for value in series.tolist():
            if value is None or value is pd.NA:
                continue
            counts[value] += 1
        return counts

    left_counts = {normalise_left(k): v for k, v in tally(left[spec.left_key]).items()}
    right_counts = {
        normalise_right(k): v for k, v in tally(right[spec.right_key]).items()
    }
    total = 0.0
    for row_a, row_b in zip(
        middle[spec.mid_key_left].tolist(), middle[spec.mid_key_right].tolist()
    ):
        if row_a is None or row_a is pd.NA or row_b is None or row_b is pd.NA:
            continue
        total += left_counts.get(normalise_mid_left(row_a), 0) * right_counts.get(
            normalise_mid_right(row_b), 0
        )
    if int(round(total)) != int(reported["f_joint"]):
        raise AssertionError(f"recheck {total} != f_joint {reported['f_joint']}")
    if int(round(total)) != int(reported["exact"]):
        raise AssertionError(f"recheck {total} != exact {reported['exact']}")


def counterexample_correlation() -> dict[str, object]:
    """Gate 4 positive control: same marginals, different joint."""
    left = pd.DataFrame({"a": ["A"] * 10 + ["B"]})
    right = pd.DataFrame({"b": ["X"] * 10 + ["Y"]})
    worlds = [
        pd.DataFrame({"a": ["A", "B"], "b": ["X", "Y"]}),
        pd.DataFrame({"a": ["A", "B"], "b": ["Y", "X"]}),
    ]
    spec = Spec3("l", "l", "a", "m", "m", "a", "b", "r", "r", "b", ())
    scored = [
        score_query3(spec, {"l": left, "m": world, "r": right}, truth=1.0)
        for world in worlds
    ]
    f_joint = [float(s["f_joint"]) for s in scored]
    u_indep = [float(s["u_indep"]) for s in scored]
    return {
        "f_joint": f_joint,
        "u_indep": u_indep,
        "passed": f_joint[0] != f_joint[1] and u_indep[0] == u_indep[1],
    }


def counterexample_constant_marginal() -> dict[str, object]:
    """Gate 4 negative control: a constant side forces the two to agree."""
    left = pd.DataFrame({"a": ["A"] * 5})
    right = pd.DataFrame({"b": ["X"] * 5 + ["Y"] * 5})
    middle = pd.DataFrame({"a": ["A", "A"], "b": ["X", "Y"]})
    spec = Spec3("l", "l", "a", "m", "m", "a", "b", "r", "r", "b", ())
    scored = score_query3(spec, {"l": left, "m": middle, "r": right}, truth=1.0)
    return {
        "f_joint": float(scored["f_joint"]),
        "f_indep": float(scored["f_indep"]),
        "passed": abs(float(scored["f_joint"]) - float(scored["f_indep"])) < 1e-9,
    }


def _json_spec(spec: Spec3) -> dict[str, object]:
    return {
        "left_table": spec.left_table,
        "left_alias": spec.left_alias,
        "left_key": spec.left_key,
        "middle_table": spec.middle_table,
        "middle_alias": spec.middle_alias,
        "mid_key_left": spec.mid_key_left,
        "mid_key_right": spec.mid_key_right,
        "right_table": spec.right_table,
        "right_alias": spec.right_alias,
        "right_key": spec.right_key,
        "filters": [
            {
                "alias": p.alias,
                "column": p.column,
                "op": p.op,
                "raw": p.raw,
                "is_string": p.is_string,
            }
            for p in spec.filters
        ],
    }


def _spec_from_json(value: dict[str, object]) -> Spec3:
    return Spec3(
        left_table=value["left_table"],
        left_alias=value["left_alias"],
        left_key=value["left_key"],
        middle_table=value["middle_table"],
        middle_alias=value["middle_alias"],
        mid_key_left=value["mid_key_left"],
        mid_key_right=value["mid_key_right"],
        right_table=value["right_table"],
        right_alias=value["right_alias"],
        right_key=value["right_key"],
        filters=tuple(Predicate(**f) for f in value["filters"]),
    )


def _classify_unsupported(reason: str) -> str:
    if reason.startswith("table_count_"):
        return reason.removeprefix("table_count_")
    if reason in {
        "not_a_path",
        "same_middle_key",
        "join_predicate_count_0",
        "join_predicate_count_1",
        "join_predicate_count_2",
    }:
        return "3_not_path"
    return "other"


def build_plan(
    workload_dir: Path, databases: list[str], max_per_db: int, out: Path
) -> dict[str, object]:
    plan: dict[str, object] = {"databases": {}, "max_per_db": max_per_db}
    for db in databases:
        if db in TARGET_DATABASES:
            raise ValueError(f"Refusing to read target database {db}")
        path = workload_dir / db / "workloads.sql"
        rows = read_workload(path, db, "pretrain")
        unique: dict[str, dict[str, object]] = {}
        for row in rows:
            held = unique.get(row["query_hash"])
            if held is None:
                unique[row["query_hash"]] = row
            elif not math.isclose(held["true"], row["true"], rel_tol=1e-6):
                raise ValueError(f"{db}: inconsistent true cardinality")
            elif not math.isclose(held["pg"], row["pg"], rel_tol=1e-6):
                raise ValueError(f"{db}: inconsistent pg estimate")

        unsupported: Counter = Counter()
        table_counts: Counter = Counter()
        eligible: list[dict[str, object]] = []
        for query_hash in sorted(unique):
            row = unique[query_hash]
            try:
                spec = analyze3(row["sql"])
            except Unsupported as reason:
                text = str(reason)
                unsupported[text] += 1
                table_counts[_classify_unsupported(text)] += 1
                continue
            table_counts["3"] += 1
            eligible.append(
                {
                    "query_hash": query_hash,
                    "sql": row["sql"],
                    "true": row["true"],
                    "pg": row["pg"],
                    "spec": _json_spec(spec),
                }
            )
        specs = [_spec_from_json(item["spec"]) for item in eligible]
        picks = evenly_spaced([e["query_hash"] for e in eligible], max_per_db)
        plan["databases"][db] = {
            "workload_sha256": file_hash(path),
            "rows": len(rows),
            "unique_queries": len(unique),
            "eligible": len(eligible),
            "eligible_fraction": len(eligible) / len(unique),
            "table_count_histogram": dict(
                sorted(table_counts.items(), key=lambda kv: (len(kv[0]), kv[0]))
            ),
            "filter_count_histogram": dict(
                sorted(Counter(len(s.filters) for s in specs).items())
            ),
            "filters_join_key_count": sum(s.filters_join_key() for s in specs),
            "triple_histogram": dict(
                sorted(
                    Counter(
                        f"{s.left_table}|{s.middle_table}|{s.right_table}"
                        for s in specs
                    ).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ),
            "unsupported": dict(sorted(unsupported.items())),
            "queries": [eligible[i] for i in picks],
        }
        print(
            f"{db}: rows={len(rows)} unique={len(unique)} "
            f"eligible={len(eligible)} ({len(eligible) / len(unique):.2%}) "
            f"sampled={len(picks)}",
            flush=True,
        )
    write_json(out / "plan.json", plan)
    return plan


def _selectivity_bucket(value: float) -> str:
    for upper, label in (
        (1e-3, "<1e-3"),
        (1e-2, "1e-3..1e-2"),
        (1e-1, "1e-2..1e-1"),
        (0.5, "1e-1..0.5"),
    ):
        if value < upper:
            return label
    return ">=0.5"


def breakdown(frame: pd.DataFrame) -> dict[str, object]:
    scored = frame.assign(
        selectivity=frame[["sel_left", "sel_middle", "sel_right"]].min(axis=1)
    )
    scored["bucket"] = [
        ZERO_POPULATION_BUCKET if flag else _selectivity_bucket(float(value))
        for flag, value in zip(scored["zero_population"], scored["selectivity"])
    ]
    rows = []
    for (database, bucket), group in scored.groupby(["database", "bucket"]):
        rows.append(
            {
                "database": database,
                "selectivity_bucket": bucket,
                "n": int(len(group)),
                "median_q_u_indep": percentile(group["q_u_indep"].tolist(), 50),
                "q95_q_u_indep": percentile(group["q_u_indep"].tolist(), 95),
                "median_q_f_indep": percentile(group["q_f_indep"].tolist(), 50),
                "q95_q_f_indep": percentile(group["q_f_indep"].tolist(), 95),
                "q95_q_f_joint": percentile(group["q_f_joint"].tolist(), 95),
            }
        )
    return {
        "selectivity": "min(sel_left, sel_middle, sel_right)",
        "note": (
            "descriptive only; does not enter the go/no-go gate. Queries with "
            f"M_i = 0 have no defined selectivity and sit in {ZERO_POPULATION_BUCKET}."
        ),
        "rows": sorted(rows, key=lambda r: (r["database"], r["selectivity_bucket"])),
    }


def _summarise(frame: pd.DataFrame, column: str) -> dict[str, float]:
    values = [float(v) for v in frame[column]]
    return {
        "median": percentile(values, 50),
        "q95": percentile(values, 95),
        "max": max(values),
        "mean": float(np.mean(values)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("plan", "run"))
    parser.add_argument("--workload-dir", type=Path)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--databases", nargs="+", default=list(MAIN_DATABASES))
    parser.add_argument("--max-per-db", type=int, default=DEFAULT_MAX_PER_DB)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "plan":
        if args.workload_dir is None:
            parser.error("plan requires --workload-dir")
        plan = build_plan(args.workload_dir, args.databases, args.max_per_db, args.out)
        protocol = {
            "stage": "plan",
            "protocol": "116-G2-v3",
            "sqlglot": sqlglot.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "databases": args.databases,
            "max_per_db": args.max_per_db,
            "scope": "three tables, path, two single-key equi-joins, a != b",
            "estimators": {
                "u_indep": "unfiltered marginals, middle keys independent",
                "u_joint": "unfiltered joint, scaled by selectivities",
                "f_indep": "filtered marginals, middle keys independent",
                "f_joint": "filtered joint (oracle)",
            },
            "workload_hashes": {
                db: plan["databases"][db]["workload_sha256"] for db in args.databases
            },
        }
        write_json(args.out / "protocol.json", protocol)
        digest = hashlib.sha256(
            json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        (args.out / "plan.sha256").write_text(digest + "\n")
        print(f"plan_sha256={digest}")
        return 0

    if args.plan is None or args.dataset_dir is None:
        parser.error("run requires --plan and --dataset-dir")
    plan = json.loads(args.plan.read_text())
    expected = (args.plan.parent / "plan.sha256").read_text().strip()
    actual = hashlib.sha256(
        json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    if expected != actual:
        raise SystemExit(f"Plan SHA mismatch: {actual} != {expected}")

    rows: list[dict[str, object]] = []
    coverage: dict[str, object] = {}
    loaded: dict[str, dict[str, pd.DataFrame]] = {}
    rejected: list[dict[str, object]] = []
    for db, block in plan["databases"].items():
        if db in TARGET_DATABASES:
            raise SystemExit(f"Refusing to read target database {db}")
        dataset = args.dataset_dir / db
        kinds, files = parse_ddl(dataset / f"postgres_create_{db}.sql")
        tables = load_tables(dataset, kinds, files)
        loaded[db] = tables
        representable = 0
        for item in block["queries"]:
            spec = _spec_from_json(item["spec"])
            try:
                if _json_spec(analyze3(item["sql"])) == item["spec"]:
                    representable += 1
            except Unsupported:
                pass
            try:
                scored = score_query3(spec, tables, float(item["true"]))
            except Unsupported as reason:
                rejected.append(
                    {
                        "database": db,
                        "query_hash": item["query_hash"],
                        "reason": str(reason),
                    }
                )
                continue
            rows.append(
                {
                    "database": db,
                    "query_hash": item["query_hash"],
                    "sql": item["sql"],
                    "left_table": spec.left_table,
                    "left_key": spec.left_key,
                    "middle_table": spec.middle_table,
                    "mid_key_left": spec.mid_key_left,
                    "mid_key_right": spec.mid_key_right,
                    "right_table": spec.right_table,
                    "right_key": spec.right_key,
                    "n_filters": len(spec.filters),
                    "filters_join_key": spec.filters_join_key(),
                    **scored,
                }
            )
        coverage[db] = {
            "rows": block["rows"],
            "unique_queries": block["unique_queries"],
            "eligible": block["eligible"],
            "eligible_fraction": block["eligible_fraction"],
            "sampled": len(block["queries"]),
            "sampled_representable": representable,
            "sampled_representable_fraction": representable
            / max(len(block["queries"]), 1),
            "run_rejected": sum(1 for item in rejected if item["database"] == db),
            "filters_join_key_count": block["filters_join_key_count"],
            "triple_histogram": block["triple_histogram"],
        }
        print(
            f"{db}: scored {len(block['queries'])} queries "
            f"(run_rejected={coverage[db]['run_rejected']})",
            flush=True,
        )

    for db, block in plan["databases"].items():
        by_hash = {item["query_hash"]: item for item in block["queries"]}
        picks = [r for r in rows if r["database"] == db][:MIN_RECHECKS]
        for row in picks:
            independent_recheck3(
                _spec_from_json(by_hash[row["query_hash"]]["spec"]),
                loaded[db],
                row,
            )
        print(f"{db}: independent recheck {len(picks)} ok", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("All sampled queries were rejected; nothing to score")
    frame.to_csv(args.out / "query_results.csv", index=False)

    summary = {
        name: _summarise(frame, column)
        for name, column in (
            ("u_indep", "q_u_indep"),
            ("u_joint", "q_u_joint"),
            ("f_indep", "q_f_indep"),
            ("f_joint", "q_f_joint"),
        )
    }
    write_json(args.out / "breakdown.json", breakdown(frame))

    correctness = bool(
        (frame["f_joint"].round(6) == frame["exact"].astype(float)).all()
    )
    max_discrepancy = float(frame["discrepancy"].max())
    coverage_ok = bool(
        coverage
        and all(
            block["sampled_representable_fraction"] >= 1.0
            and block["run_rejected"] == 0
            for block in coverage.values()
        )
    )
    q1 = bool(summary["f_indep"]["q95"] <= 0.5 * summary["u_indep"]["q95"])
    q2 = bool(summary["f_joint"]["q95"] <= 0.5 * summary["f_indep"]["q95"])
    positive = counterexample_correlation()
    negative = counterexample_constant_marginal()
    counterexample_ok = bool(positive["passed"] and negative["passed"])

    blocked = (
        not correctness
        or max_discrepancy > 0.05
        or not coverage_ok
        or not counterexample_ok
    )
    if blocked:
        verdict = "blocked"
    elif q1 and q2:
        verdict = "both_thresholds_met"
    elif q1:
        verdict = "q1_threshold_met_only"
    elif q2:
        verdict = "q2_threshold_met_only"
    else:
        verdict = "neither_threshold_met"

    go_no_go = {
        "verdict": verdict,
        "protocol": "116-G2-v3",
        "gates": {
            "1_correctness": {
                "passed": correctness,
                "f_joint_equals_exact": correctness,
            },
            "1b_source_check": {
                "passed": max_discrepancy <= 0.05,
                "max_relative_discrepancy": max_discrepancy,
                "threshold": 0.05,
            },
            "2_execution_consistency": {
                "passed": coverage_ok,
                "rule": "sampled spec identical 100%, run_rejected == 0",
                "per_database": coverage,
            },
            "3a_q1_conditioning": {
                "passed": q1,
                "rule": "q95(F_indep) <= 0.5 * q95(U_indep)",
                "ratio": summary["f_indep"]["q95"] / summary["u_indep"]["q95"],
            },
            "3b_q2_correlation": {
                "passed": q2,
                "rule": "q95(F_joint) <= 0.5 * q95(F_indep)",
                "ratio": summary["f_joint"]["q95"] / summary["f_indep"]["q95"],
            },
            "4_counterexamples": {
                "passed": counterexample_ok,
                "positive": positive,
                "negative": negative,
            },
        },
        "summary": summary,
        "forbidden_inferences": [
            "3a met and 3b not met does NOT mean the joint table is unnecessary",
            "3b met does NOT mean a 2-D joint store is required",
            "both met does NOT authorize the architecture: no compression, "
            "cross-database transfer or training was tested",
        ],
    }
    write_json(args.out / "coverage.json", coverage)
    write_json(args.out / "rejected.json", rejected)
    write_json(args.out / "go_no_go.json", go_no_go)
    artifacts = [
        "protocol.json",
        "plan.json",
        "plan.sha256",
        "query_results.csv",
        "coverage.json",
        "rejected.json",
        "breakdown.json",
        "go_no_go.json",
    ]
    (args.out / "artifacts.sha256").write_text(
        "".join(
            f"{file_hash(args.out / name)}  {name}\n"
            for name in artifacts
            if (args.out / name).exists()
        )
    )
    print(json.dumps(summary, indent=2))
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
