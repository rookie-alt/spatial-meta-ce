"""G1a: two-table, single-equi-join conditional key-frequency oracle.

Read-only feasibility check for the conditional-summary line described in
`实验文档/114-联合分布G1可行性质检协议.md`. It never trains, never loads a
model and never touches a target database. It answers one question:

    does `P(join key | filters)` carry information that
    `|R|^P` plus the *unfiltered* key distribution does not?

For every eligible query it recomputes the exact cardinality on the verified
CSV snapshot (G0 report: `reports/g0-source-identity-20261102/`) and compares
three estimators:

    B0       = |R|^P * |S|^P / max(dom(K_R), dom(K_S))
    C_uncond = |R|^P * |S|^P * sum_k p_R(k) p_S(k)     (p = unfiltered)
    C_cond   = sum_k phi_R^P(k) phi_S^P(k)             (filtered)

The script runs in two stages so the query plan can be frozen before any
result is visible:

    python scripts/g1_oracle.py plan --workload-dir ... --out DIR
    python scripts/g1_oracle.py run  --plan DIR/plan.json --dataset-dir ... --out DIR

The `run` stage refuses a plan whose SHA256 does not match `plan.sha256`.
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
from spatial_meta_ce.manifests import file_hash, read_workload  # noqa: E402

MAIN_DATABASES = ("hepatitis", "carcinogenesis", "basketball")
TARGET_DATABASES = ("imdb", "stats", "genome", "ergastf1")
DEFAULT_MAX_PER_DB = 200
MIN_NUMPY_RECHECKS = 20

OPERATORS = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
}
FLIPPED = {"=": "=", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}

# Mirrors PRICE's `utils/statistics/tools.py:load_tbls_cols_types` so that the
# oracle sees the same dtypes the frozen statistics were generated from.
DTYPES = {"int": "Int64", "str": "string", "float": "Float64"}


class Unsupported(Exception):
    """The query is outside the frozen G1 scope; the reason is the argument."""


@dataclass(frozen=True)
class Predicate:
    alias: str
    column: str
    op: str
    raw: str
    # Provenance only: whether the literal was quoted in SQL. Comparison casts
    # the literal to the *column's* DDL type and never reads this flag.
    is_string: bool


@dataclass(frozen=True)
class Spec:
    left_table: str
    left_alias: str
    left_key: str
    right_table: str
    right_alias: str
    right_key: str
    filters: tuple[Predicate, ...]

    def for_alias(self, alias: str) -> list[Predicate]:
        return [p for p in self.filters if p.alias == alias]

    def filters_join_key(self) -> bool:
        pairs = {
            (self.left_alias, self.left_key),
            (self.right_alias, self.right_key),
        }
        return any((p.alias, p.column) in pairs for p in self.filters)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def _strip(node: exp.Expression) -> exp.Expression:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _conjuncts(node: exp.Expression):
    node = _strip(node)
    if isinstance(node, exp.And):
        yield from _conjuncts(node.this)
        yield from _conjuncts(node.expression)
    else:
        yield node


def _literal(node: exp.Expression) -> tuple[str, bool] | None:
    node = _strip(node)
    sign = ""
    if isinstance(node, exp.Neg) and isinstance(node.this, exp.Literal):
        sign, node = "-", node.this
    if not isinstance(node, exp.Literal):
        return None
    return sign + str(node.this), bool(node.is_string)


def _predicate(node: exp.Expression) -> Predicate | None:
    """Build a `col op literal` predicate, flipping the operator if needed."""
    op = OPERATORS[type(node)]
    left, right = _strip(node.this), _strip(node.expression)
    literal = _literal(right)
    if isinstance(left, exp.Column) and literal is not None:
        column, (raw, is_string) = left, literal
    else:
        literal = _literal(left)
        if not isinstance(right, exp.Column) or literal is None:
            return None
        column, (raw, is_string) = right, literal
        op = FLIPPED[op]
    if not column.table:
        return None
    return Predicate(column.table, column.name, op, raw, is_string)


def analyze(sql: str) -> Spec:
    """Return the frozen-scope spec for `sql` or raise `Unsupported(reason)`."""
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
    if len(tables) != 2:
        raise Unsupported(f"table_count_{len(tables)}")
    if tables[0].name == tables[1].name:
        raise Unsupported("self_join")
    alias = {t.alias_or_name: t.name for t in tables}
    if len(alias) != 2:
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

    # ON-clause conjuncts and WHERE conjuncts share one classification path, so a
    # non-equality predicate in `ON` is rejected rather than silently dropped, and a
    # CROSS JOIN (no `on` at all) cannot raise KeyError.
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

    if len(joins) != 1:
        raise Unsupported(f"join_predicate_count_{len(joins)}")
    join = joins[0]
    left, right = join.this, join.expression
    if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
        raise Unsupported("join_not_column_equality")
    if not left.table or not right.table:
        raise Unsupported("unqualified_join_column")
    if left.table == right.table:
        raise Unsupported("same_table_equality")
    if {left.table, right.table} != set(alias):
        raise Unsupported("join_alias_mismatch")
    for predicate in filters:
        if predicate.alias not in alias:
            raise Unsupported("filter_alias_mismatch")

    spec = Spec(
        left_table=alias[left.table],
        left_alias=left.table,
        left_key=left.name,
        right_table=alias[right.table],
        right_alias=right.table,
        right_key=right.name,
        filters=tuple(filters),
    )
    if (spec.left_table, spec.left_key) > (spec.right_table, spec.right_key):
        spec = Spec(
            left_table=spec.right_table,
            left_alias=spec.right_alias,
            left_key=spec.right_key,
            right_table=spec.left_table,
            right_alias=spec.left_alias,
            right_key=spec.left_key,
            filters=spec.filters,
        )
    return spec


def parse_ddl(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Column kinds and `\\copy` table->file map, read the way PRICE reads it."""
    kinds: dict[str, dict[str, str]] = {}
    files: dict[str, str] = {}
    table = None
    with path.open() as handle:
        for line in handle:
            low = line.lower()
            if line.startswith("\\copy") or line.startswith("\\COPY"):
                parts = line.split()
                files[parts[1].strip("'").strip('"')] = parts[3].strip("'")
                continue
            if "create table" in low:
                # Take the name from the original line: a quoted identifier
                # keeps its case, and `low` would have folded it away.
                rest = line[low.index("create table") + len("create table") :]
                tokens = rest.split()
                if [t.lower() for t in tokens[:3]] == ["if", "not", "exists"]:
                    tokens = tokens[3:]
                if not tokens:
                    continue
                table = tokens[0].replace('"', "").split(".")[-1].rstrip("(")
                kinds[table] = {}
                continue
            if table is None:
                continue
            column = line.strip().split(" ")[0].replace('"', "")
            if "integer" in low or "bigint" in low or "smallint" in low:
                kinds[table][column] = "int"
            elif "character" in low or "varchar(" in low or "char(" in low:
                kinds[table][column] = "str"
            elif " text" in low or "boolean" in low:
                kinds[table][column] = "str"
            elif "decimal(" in low or "numeric(" in low or "double precision" in low:
                kinds[table][column] = "float"
    return kinds, files


def load_tables(
    dataset_dir: Path, kinds: dict[str, dict[str, str]], files: dict[str, str]
) -> dict[str, pd.DataFrame]:
    tables = {}
    for table, filename in files.items():
        if table not in kinds:
            match = next((k for k in kinds if k.lower() == table.lower()), None)
            if match is None:
                raise ValueError(f"Table {table} missing from DDL")
            table = match
        dtype = {c: DTYPES[k] for c, k in kinds[table].items()}
        tables[table] = pd.read_csv(
            dataset_dir / filename,
            sep="|",
            quotechar='"',
            escapechar="\\",
            dtype=dtype,
            keep_default_na=False,
            na_values=["NULL"],
        )
    return tables


def _coerce(series: pd.Series, predicate: Predicate) -> object:
    """Cast the literal to the column's DDL type, matching PostgreSQL semantics.

    PostgreSQL rejects `integer_col = 1.5` and `integer_col = 'abc'` instead of
    truncating or coercing, so those raise `Unsupported` rather than quietly
    comparing a different value.
    """
    if pd.api.types.is_integer_dtype(series.dtype):
        raw = predicate.raw.strip()
        if not raw.lstrip("+-").isdigit():
            raise Unsupported("non_integer_literal_on_int_column")
        return int(raw)
    if pd.api.types.is_float_dtype(series.dtype):
        return float(predicate.raw)
    return predicate.raw


def _mask(series: pd.Series, predicate: Predicate) -> pd.Series:
    target = _coerce(series, predicate)
    op = predicate.op
    if op == "=":
        mask = series == target
    elif op == "!=":
        mask = series != target
    elif op == "<":
        mask = series < target
    elif op == "<=":
        mask = series <= target
    elif op == ">":
        mask = series > target
    else:
        mask = series >= target
    return mask.fillna(False).astype(bool)


def apply_filters(frame: pd.DataFrame, predicates: list[Predicate]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for predicate in predicates:
        if predicate.column not in frame.columns:
            raise ValueError(f"Unknown column {predicate.column}")
        mask &= _mask(frame[predicate.column], predicate)
    return frame[mask.astype(bool)]


def key_counts(frame: pd.DataFrame, key: str) -> pd.Series:
    """Non-NULL key frequencies: SQL `=` never matches NULL."""
    values = frame[key]
    return values[values.notna()].value_counts()


def _key_kind(dtype: object) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    return "str"


def _target_dtype(left: object, right: object) -> str:
    """Pick one dtype for two key columns; refuse an ambiguous cross-kind join.

    Never falls back to `astype(str)`: that would merge int 6 with str "6" and
    split int 1995 from float 1995.0, neither of which matches PostgreSQL.
    """
    left_kind, right_kind = _key_kind(left), _key_kind(right)
    if "str" in (left_kind, right_kind) and left_kind != right_kind:
        raise Unsupported(f"join_key_kind_mismatch_{left_kind}_{right_kind}")
    if left_kind == "str":
        return "string"
    if left_kind == right_kind == "int":
        return "Int64"
    # integer vs numeric is an implicit cast in PostgreSQL.
    return "Float64"


def _aligned(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align two `value_counts()` results, whose *index* carries the keys."""
    target = _target_dtype(a.index.dtype, b.index.dtype)
    a = a.set_axis(a.index.astype(target))
    b = b.set_axis(b.index.astype(target))
    common = a.index.intersection(b.index)
    return a.loc[common].astype("float64"), b.loc[common].astype("float64")


def _merge_keys(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    target = _target_dtype(a.dtype, b.dtype)
    return pd.DataFrame({"__k": a.astype(target)}).merge(
        pd.DataFrame({"__k": b.astype(target)}), on="__k", how="inner"
    )


def q_error(estimate: float, truth: float) -> float:
    """Project-standard q-error: `exp(|log max(est,1) - log max(true,1)|)`."""
    return math.exp(abs(math.log(max(estimate, 1.0)) - math.log(max(truth, 1.0))))


def score_query(
    spec: Spec, tables: dict[str, pd.DataFrame], truth: float
) -> dict[str, object]:
    left = tables[spec.left_table]
    right = tables[spec.right_table]
    left_f = apply_filters(left, spec.for_alias(spec.left_alias))
    right_f = apply_filters(right, spec.for_alias(spec.right_alias))

    phi_a, phi_b = _aligned(
        key_counts(left_f, spec.left_key), key_counts(right_f, spec.right_key)
    )
    c_cond = float((phi_a * phi_b).sum())

    counts_left = key_counts(left, spec.left_key)
    counts_right = key_counts(right, spec.right_key)
    uncond_a, uncond_b = _aligned(counts_left, counts_right)
    n_left, n_right = len(left), len(right)
    collision = (
        float(((uncond_a / n_left) * (uncond_b / n_right)).sum())
        if n_left and n_right
        else 0.0
    )
    n_left_f, n_right_f = len(left_f), len(right_f)
    c_uncond = n_left_f * n_right_f * collision

    domain = max(len(counts_left), len(counts_right))
    b0 = (n_left_f * n_right_f / domain) if domain else 0.0

    exact = len(
        _merge_keys(left_f[spec.left_key].dropna(), right_f[spec.right_key].dropna())
    )
    if exact == truth:
        discrepancy = 0.0
    elif truth > 0:
        discrepancy = abs(exact - truth) / truth
    else:
        discrepancy = 1.0

    return {
        "exact": exact,
        "true": truth,
        "b0": b0,
        "c_uncond": c_uncond,
        "c_cond": c_cond,
        "n_left": n_left,
        "n_right": n_right,
        "n_left_f": n_left_f,
        "n_right_f": n_right_f,
        "dom_left": int(len(counts_left)),
        "dom_right": int(len(counts_right)),
        "domain": int(domain),
        "collision": collision,
        "sel_left": n_left_f / n_left if n_left else 0.0,
        "sel_right": n_right_f / n_right if n_right else 0.0,
        "q_b0": q_error(b0, truth),
        "q_uncond": q_error(c_uncond, truth),
        "q_cond": q_error(c_cond, truth),
        "discrepancy": discrepancy,
    }


def independent_recheck(
    spec: Spec, tables: dict[str, pd.DataFrame], reported: dict[str, object]
) -> None:
    """Independent NumPy recomputation: no pandas groupby, no merge."""

    def matches(value: object, target: object, op: str, numeric: bool) -> bool:
        if value is None or value is pd.NA:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        got = float(value) if numeric else str(value)
        if op == "=":
            return got == target
        if op == "!=":
            return got != target
        if op == "<":
            return got < target
        if op == "<=":
            return got <= target
        if op == ">":
            return got > target
        return got >= target

    def surviving(frame: pd.DataFrame, predicates: list[Predicate]) -> list[dict]:
        keep = [True] * len(frame)
        records = frame.to_dict("records")
        for predicate in predicates:
            numeric = pd.api.types.is_integer_dtype(frame[predicate.column].dtype)
            numeric = numeric or pd.api.types.is_float_dtype(
                frame[predicate.column].dtype
            )
            target = float(predicate.raw) if numeric else predicate.raw
            for i, row in enumerate(records):
                if keep[i] and not matches(
                    row[predicate.column], target, predicate.op, numeric
                ):
                    keep[i] = False
        return [row for row, ok in zip(records, keep) if ok]

    left_rows = surviving(tables[spec.left_table], spec.for_alias(spec.left_alias))
    right_rows = surviving(tables[spec.right_table], spec.for_alias(spec.right_alias))
    # Normalise keys exactly like `_target_dtype`, so this check agrees with
    # `score_query` instead of raising a false alarm on int-vs-float joins.
    target = _target_dtype(
        tables[spec.left_table][spec.left_key].dtype,
        tables[spec.right_table][spec.right_key].dtype,
    )
    numeric_keys = target != "string"

    def normalise(value: object) -> object:
        if value is None or value is pd.NA:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value) if numeric_keys else str(value)

    tally_right: Counter = Counter()
    for row in right_rows:
        key = normalise(row[spec.right_key])
        if key is not None:
            tally_right[key] += 1
    total = 0
    for row in left_rows:
        key = normalise(row[spec.left_key])
        if key is not None:
            total += tally_right.get(key, 0)
    if total != int(reported["c_cond"]):
        raise AssertionError(f"NumPy recheck {total} != c_cond {reported['c_cond']}")
    if total != int(reported["exact"]):
        raise AssertionError(f"NumPy recheck {total} != exact {reported['exact']}")


def evenly_spaced(items: list[str], limit: int) -> list[int]:
    if len(items) <= limit:
        return list(range(len(items)))
    picks = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return sorted({int(i) for i in picks})


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
        eligible: list[dict[str, object]] = []
        for query_hash in sorted(unique):
            row = unique[query_hash]
            try:
                spec = analyze(row["sql"])
            except Unsupported as reason:
                unsupported[str(reason)] += 1
                continue
            eligible.append(
                {
                    "query_hash": query_hash,
                    "sql": row["sql"],
                    "true": row["true"],
                    "pg": row["pg"],
                    "spec": _json_spec(spec),
                }
            )
        picks = evenly_spaced([e["query_hash"] for e in eligible], max_per_db)
        selected = [eligible[i] for i in picks]
        # 114 §4.3 requires these distributions to be frozen before the run.
        table_counts: Counter = Counter({"2": len(eligible)})
        for reason, count in unsupported.items():
            if reason.startswith("table_count_"):
                table_counts[reason.removeprefix("table_count_")] += count
            else:
                table_counts["unknown"] += count
        specs = [_spec_from_json(item["spec"]) for item in eligible]
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
            "table_pair_histogram": dict(
                sorted(
                    Counter(f"{s.left_table}|{s.right_table}" for s in specs).items(),
                    key=lambda kv: (-kv[1], kv[0]),
                )
            ),
            "unsupported": dict(sorted(unsupported.items())),
            "queries": selected,
        }
        print(
            f"{db}: rows={len(rows)} unique={len(unique)} "
            f"eligible={len(eligible)} ({len(eligible) / len(unique):.2%}) "
            f"sampled={len(selected)}",
            flush=True,
        )
    write_json(out / "plan.json", plan)
    return plan


def _json_spec(spec: Spec) -> dict[str, object]:
    return {
        "left_table": spec.left_table,
        "left_alias": spec.left_alias,
        "left_key": spec.left_key,
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


def _spec_from_json(value: dict[str, object]) -> Spec:
    return Spec(
        left_table=value["left_table"],
        left_alias=value["left_alias"],
        left_key=value["left_key"],
        right_table=value["right_table"],
        right_alias=value["right_alias"],
        right_key=value["right_key"],
        filters=tuple(Predicate(**f) for f in value["filters"]),
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _selectivity_bucket(value: float) -> str:
    """Protocol §7.3 buckets, frozen before any bucket was inspected."""
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
    """§7.3 descriptive split: join-key filtering, selectivity, and database."""
    scored = frame.assign(selectivity=frame[["sel_left", "sel_right"]].min(axis=1))
    scored["selectivity_bucket"] = scored["selectivity"].map(_selectivity_bucket)
    rows = []
    for (database, join_key, bucket), group in scored.groupby(
        ["database", "filters_join_key", "selectivity_bucket"], dropna=False
    ):
        rows.append(
            {
                "database": database,
                "filters_join_key": bool(join_key),
                "selectivity_bucket": bucket,
                "n": int(len(group)),
                "median_q_uncond": percentile(group["q_uncond"].tolist(), 50),
                "median_q_cond": percentile(group["q_cond"].tolist(), 50),
                "q95_q_uncond": percentile(group["q_uncond"].tolist(), 95),
                "q95_q_cond": percentile(group["q_cond"].tolist(), 95),
            }
        )
    return {
        "selectivity": "min(sel_left, sel_right), the more selective side",
        "buckets": [
            "<1e-3",
            "1e-3..1e-2",
            "1e-2..1e-1",
            "1e-1..0.5",
            ">=0.5",
        ],
        "rows": sorted(
            rows,
            key=lambda r: (
                r["database"],
                r["filters_join_key"],
                r["selectivity_bucket"],
            ),
        ),
        "note": (
            "descriptive only; it does not change the frozen go/no-go gate. "
            "Strata are listed per database because the sampled table pairs are "
            "not balanced across databases."
        ),
    }


def two_world_counterexample() -> dict[str, object]:
    """Protocol §6: identical marginals, different conditional answer."""
    world_a = pd.DataFrame({"k": ["a", "b"], "x": [1, 0]})
    world_b = pd.DataFrame({"k": ["a", "b"], "x": [0, 1]})
    other = pd.DataFrame({"k": ["a"]})
    predicate = Predicate("r", "x", "=", "1", False)

    def cond(frame: pd.DataFrame) -> float:
        a, b = _aligned(
            key_counts(apply_filters(frame, [predicate]), "k"),
            key_counts(other, "k"),
        )
        return float((a * b).sum())

    def uncond(frame: pd.DataFrame) -> float:
        a, b = _aligned(key_counts(frame, "k"), key_counts(other, "k"))
        collision = float(((a / len(frame)) * (b / len(other))).sum())
        return len(apply_filters(frame, [predicate])) * len(other) * collision

    return {
        "c_cond": [cond(world_a), cond(world_b)],
        "c_uncond": [uncond(world_a), uncond(world_b)],
        "passed": cond(world_a) != cond(world_b) and uncond(world_a) == uncond(world_b),
        "note": "worlds differ only in which key carries x=1",
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
            "sqlglot": sqlglot.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "databases": args.databases,
            "max_per_db": args.max_per_db,
            "scope": "two tables, one equi-join, AND-only literal predicates",
            "metric": "q = exp(abs(log(max(est,1)) - log(max(true,1))))",
            "estimators": {
                "b0": "|R|^P |S|^P / max(dom_R, dom_S)",
                "c_uncond": "|R|^P |S|^P sum_k p_R(k) p_S(k), p unfiltered",
                "c_cond": "sum_k phi_R^P(k) phi_S^P(k)",
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
            # Gate 2: every sampled query must re-analyze to the frozen spec.
            try:
                if _json_spec(analyze(item["sql"])) == item["spec"]:
                    representable += 1
            except Unsupported:
                pass
            try:
                scored = score_query(spec, tables, float(item["true"]))
            except Unsupported as reason:
                # A literal or key type the frozen scope cannot represent is a
                # recorded rejection, not a crash and not a silent drop.
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
            "unsupported": block["unsupported"],
            "filters_join_key_count": block["filters_join_key_count"],
            "run_rejected": sum(1 for item in rejected if item["database"] == db),
        }
        print(
            f"{db}: scored {len(block['queries'])} queries "
            f"(run_rejected={coverage[db]['run_rejected']})",
            flush=True,
        )

    for db, block in plan["databases"].items():
        by_hash = {item["query_hash"]: item for item in block["queries"]}
        picks = [r for r in rows if r["database"] == db][:MIN_NUMPY_RECHECKS]
        for row in picks:
            independent_recheck(
                _spec_from_json(by_hash[row["query_hash"]]["spec"]),
                loaded[db],
                row,
            )
        print(f"{db}: independent recheck {len(picks)} ok", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("All sampled queries were rejected; nothing to score")
    frame.to_csv(args.out / "query_results.csv", index=False)

    def summarise(column: str) -> dict[str, float]:
        values = [float(v) for v in frame[column]]
        return {
            "median": percentile(values, 50),
            "q95": percentile(values, 95),
            "max": max(values),
            "mean": float(np.mean(values)),
        }

    summary = {
        "b0": summarise("q_b0"),
        "c_uncond": summarise("q_uncond"),
        "c_cond": summarise("q_cond"),
    }
    write_json(args.out / "breakdown.json", breakdown(frame))
    max_discrepancy = float(frame["discrepancy"].max())
    correctness = bool((frame["c_cond"] == frame["exact"]).all())
    conditional = bool(
        summary["c_cond"]["q95"] <= 0.5 * summary["c_uncond"]["q95"]
        and summary["c_cond"]["median"] <= 0.5 * summary["c_uncond"]["median"]
    )
    counterexample = two_world_counterexample()
    coverage_ok = bool(
        coverage
        and all(
            block["sampled_representable_fraction"] >= 0.95
            and block["run_rejected"] == 0
            for block in coverage.values()
        )
    )

    blocked = (
        not correctness
        or max_discrepancy > 0.05
        or not counterexample["passed"]
        or not coverage_ok
    )
    verdict = (
        "blocked"
        if blocked
        else "conditional_signal"
        if conditional
        else "conditional_signal_not_observed"
    )
    go_no_go = {
        "verdict": verdict,
        "gates": {
            "1_oracle_correctness": {
                "passed": correctness,
                "c_cond_equals_exact": correctness,
            },
            "1b_snapshot_vs_workload": {
                "passed": max_discrepancy <= 0.05,
                "max_relative_discrepancy": max_discrepancy,
                "threshold": 0.05,
            },
            "2_coverage": {
                "passed": coverage_ok,
                "rule": "every database: >=95% of sampled queries re-analyze "
                "to the frozen spec",
                "per_database": coverage,
            },
            "3_conditional_signal": {
                "passed": conditional,
                "rule": "q95 and median both <= 50% of c_uncond",
                "ratio_q95": summary["c_cond"]["q95"] / summary["c_uncond"]["q95"],
                "ratio_median": summary["c_cond"]["median"]
                / summary["c_uncond"]["median"],
            },
            "4_counterexample": counterexample,
            "5_compression": {
                "status": "not_run",
                "reason": "G1b is a separate stage",
            },
        },
        "summary": summary,
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
    manifest = "".join(
        f"{file_hash(args.out / name)}  {name}\n"
        for name in artifacts
        if (args.out / name).exists()
    )
    (args.out / "artifacts.sha256").write_text(manifest)
    print(json.dumps(summary, indent=2))
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
