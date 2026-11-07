"""Structural join-size identity over an arbitrary join tree.

Answers one question with no model and no PG estimate:

    does the closed-form structural estimator beat PostgreSQL's own estimate
    on queries with four or more tables?

Two estimators are reported per query:

    C_cond   = exact assembly of the filtered per-table key distributions
               (the oracle; equals the true join size)
    C_uncond = filtered table sizes times unfiltered 1-D key marginals,
               with keys across the tree assumed independent

`C_uncond` needs only per-table row counts and one-dimensional key counts --
exactly what PRICE's frozen statistics already contain -- so it is the
candidate that could replace a PG anchor without reading target databases.

Variable elimination runs along the join tree.  Join predicates that share a
table and a column denote the same variable, which keeps the repeated
middle-column shape (two tables both joining the same hub column) correct.

    python scripts/tree_identity.py plan --workload-dir ... --out DIR --min-tables 4
    python scripts/tree_identity.py run  --plan DIR/plan.json --dataset-dir ... --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
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
    apply_filters,
    evenly_spaced,
    load_tables,
    parse_ddl,
    percentile,
    q_error,
    write_json,
)
from spatial_meta_ce.manifests import file_hash, read_workload  # noqa: E402

OPERATORS = {
    exp.EQ: "=",
    exp.NEQ: "!=",
    exp.LT: "<",
    exp.LTE: "<=",
    exp.GT: ">",
    exp.GTE: ">=",
}
DEFAULT_MIN_TABLES = 4
DEFAULT_MAX_PER_DB = 200


@dataclass(frozen=True)
class TreeSpec:
    tables: tuple[str, ...]  # aliases, sorted by real table name
    names: tuple[str, ...]  # real table names, same order as `tables`
    predicates: tuple[tuple[str, str, str, str], ...]
    filters: tuple[Predicate, ...]

    def for_alias(self, alias: str) -> list[Predicate]:
        return [p for p in self.filters if p.alias == alias]

    def filters_join_key(self) -> bool:
        join_columns = set()
        for alias_a, col_a, alias_b, col_b in self.predicates:
            join_columns.add((alias_a, col_a))
            join_columns.add((alias_b, col_b))
        return any((p.alias, p.column) in join_columns for p in self.filters)


def analyze_tree(sql: str, min_tables: int = 2) -> TreeSpec:
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:  # noqa: BLE001
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

    table_nodes = list(tree.find_all(exp.Table))
    if len(table_nodes) != len({t.name for t in table_nodes}):
        raise Unsupported("self_join")
    alias = {t.alias_or_name: t.name for t in table_nodes}
    if len(alias) < min_tables:
        raise Unsupported(f"table_count_{len(table_nodes)}")

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

    predicates, filters, reasons = [], [], []
    for raw_node in conjuncts:
        node = _strip(raw_node)
        if (
            isinstance(node, exp.EQ)
            and isinstance(node.this, exp.Column)
            and isinstance(node.expression, exp.Column)
        ):
            left, right = node.this, node.expression
            if not left.table or not right.table:
                raise Unsupported("unqualified_join_column")
            if left.table not in alias or right.table not in alias:
                raise Unsupported("join_alias_mismatch")
            if left.table == right.table:
                raise Unsupported("same_table_equality")
            predicates.append((left.table, left.name, right.table, right.name))
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
    if len(predicates) != len(alias) - 1:
        raise Unsupported(f"join_predicate_count_{len(predicates)}")

    parent = {node: node for node in alias}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for alias_a, _, alias_b, _ in predicates:
        root_a, root_b = find(alias_a), find(alias_b)
        if root_a == root_b:
            raise Unsupported("not_a_tree")
        parent[root_a] = root_b
    if len({find(node) for node in alias}) != 1:
        raise Unsupported("not_a_tree")

    for predicate in filters:
        if predicate.alias not in alias:
            raise Unsupported("filter_alias_mismatch")

    order = sorted(alias, key=lambda node: alias[node])
    return TreeSpec(
        tables=tuple(order),
        names=tuple(alias[node] for node in order),
        predicates=tuple(predicates),
        filters=tuple(filters),
    )


def variable_classes(spec: TreeSpec) -> tuple[dict[int, tuple], dict[str, list[int]]]:
    """Union-find over join predicates; predicates sharing table+column merge."""
    parent = list(range(len(spec.predicates)))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    owner: dict[tuple[str, str], int] = {}
    for index, (alias_a, col_a, alias_b, col_b) in enumerate(spec.predicates):
        for key in ((alias_a, col_a), (alias_b, col_b)):
            if key in owner:
                root_a, root_b = find(owner[key]), find(index)
                if root_a != root_b:
                    parent[root_a] = root_b
            else:
                owner[key] = index
    classes: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for (alias, column), index in owner.items():
        classes[find(index)].append((alias, column))
    table_vars: dict[str, list[int]] = {alias: [] for alias in spec.tables}
    for root, members in classes.items():
        for alias, _ in members:
            if root not in table_vars[alias]:
                table_vars[alias].append(root)
    for alias in table_vars:
        table_vars[alias].sort()
    return {root: tuple(sorted(m)) for root, m in classes.items()}, table_vars


def _normaliser(dtype: object):
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        return float
    return str


def _factor(
    frame: pd.DataFrame, columns: list[str], normalisers: list
) -> dict[tuple, float]:
    if not columns:
        return {(): float(len(frame))}
    subset = frame[columns].dropna()
    if subset.empty:
        return {}
    counts = subset.groupby(columns, sort=False).size()
    result = {}
    for key, value in counts.items():
        key = key if isinstance(key, tuple) else (key,)
        result[tuple(norm(v) for norm, v in zip(normalisers, key))] = float(value)
    return result


def _eliminate(
    node: str,
    parent: str | None,
    adjacency: dict[str, list[tuple[str, int]]],
    own_vars: dict[str, list[int]],
    factors: dict[str, dict],
) -> dict:
    acc = dict(factors[node])
    for child, var in adjacency[node]:
        if child == parent:
            continue
        message = _eliminate(child, node, adjacency, own_vars, factors)
        position = own_vars[node].index(var)
        trimmed = {}
        for key, value in acc.items():
            weight = message.get(key[position])
            if weight:
                trimmed[key] = value * weight
        acc = trimmed
    keep = None
    for child, var in adjacency[node]:
        if child == parent:
            keep = var
    if keep is None:
        return {None: float(sum(acc.values()))}
    position = own_vars[node].index(keep)
    out: dict = {}
    for key, value in acc.items():
        out[key[position]] = out.get(key[position], 0.0) + value
    return out


def _assemble(
    spec: TreeSpec,
    adjacency: dict[str, list[tuple[str, int]]],
    own_vars: dict[str, list[int]],
    factors: dict[str, dict],
) -> float:
    message = _eliminate(spec.tables[0], None, adjacency, own_vars, factors)
    return float(sum(message.values()))


def score_query_tree(
    spec: TreeSpec, tables: dict[str, pd.DataFrame], truth: float
) -> dict[str, object]:
    classes, table_vars = variable_classes(spec)
    column_of = {}
    for index, (alias_a, col_a, alias_b, col_b) in enumerate(spec.predicates):
        column_of[(alias_a, index)] = col_a
        column_of[(alias_b, index)] = col_b
    adjacency: dict[str, list[tuple[str, int]]] = {alias: [] for alias in spec.tables}
    for index, (alias_a, _, alias_b, _) in enumerate(spec.predicates):
        var = next(
            root
            for root, members in classes.items()
            if (alias_a, column_of[(alias_a, index)]) in members
        )
        adjacency[alias_a].append((alias_b, var))
        adjacency[alias_b].append((alias_a, var))

    frames = {alias: tables[spec.names[i]] for i, alias in enumerate(spec.tables)}
    filtered = {a: apply_filters(f, spec.for_alias(a)) for a, f in frames.items()}

    filtered_factors, marginals = {}, {}
    columns_by_alias: dict[str, list[str]] = {}
    for alias in spec.tables:
        columns, normalisers = [], []
        for var in table_vars[alias]:
            column = next(c for a, c in classes[var] if a == alias)
            columns.append(column)
            normalisers.append(_normaliser(frames[alias][column].dtype))
        columns_by_alias[alias] = columns
        filtered_factors[alias] = _factor(filtered[alias], columns, normalisers)
        # Unfiltered marginals: the filter is applied through `scale`, so using
        # the filtered counts here as well would count selectivity twice.
        marginals[alias] = {
            column: {
                norm(v): float(c)
                for v, c in frames[alias][column].dropna().value_counts().items()
            }
            for column, norm in zip(columns, normalisers)
        }

    c_cond = _assemble(spec, adjacency, table_vars, filtered_factors)

    scale = 1.0
    for alias in spec.tables:
        total = len(frames[alias])
        kept = len(filtered[alias])
        distinct = len(table_vars[alias])
        if total == 0 or kept == 0:
            scale = 0.0
            break
        scale *= (kept / total) / (total ** max(distinct - 1, 0))

    product = 1.0
    if scale:
        for var, members in classes.items():
            per_alias: dict[str, list[dict]] = defaultdict(list)
            for alias, column in members:
                per_alias[alias].append(marginals[alias][column])
            keys = None
            for series_list in per_alias.values():
                for series in series_list:
                    domain = set(series)
                    keys = domain if keys is None else keys & domain
            total = 0.0
            for key in keys or ():
                weight = 1.0
                for series_list in per_alias.values():
                    for series in series_list:
                        weight *= series.get(key, 0.0)
                total += weight
            product *= total
    c_uncond = scale * product

    exact = int(round(c_cond))
    if truth == 0:
        discrepancy = 0.0 if exact == 0 else 1.0
    else:
        discrepancy = abs(exact - truth) / truth

    return {
        "truth": truth,
        "n_tables": len(spec.tables),
        "c_cond": c_cond,
        "c_uncond": c_uncond,
        "exact": exact,
        "q_cond": q_error(c_cond, truth),
        "q_uncond": q_error(c_uncond, truth),
        "discrepancy": discrepancy,
    }


def _json_spec(spec: TreeSpec) -> dict[str, object]:
    return {
        "tables": list(spec.tables),
        "names": list(spec.names),
        "predicates": [list(p) for p in spec.predicates],
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


def _spec_from_json(value: dict[str, object]) -> TreeSpec:
    return TreeSpec(
        tables=tuple(value["tables"]),
        names=tuple(value["names"]),
        predicates=tuple(tuple(p) for p in value["predicates"]),
        filters=tuple(Predicate(**f) for f in value["filters"]),
    )


def build_plan(
    workload_dir: Path,
    databases: list[str],
    min_tables: int,
    max_per_db: int,
    out: Path,
) -> dict[str, object]:
    plan: dict[str, object] = {"databases": {}, "min_tables": min_tables}
    for db in databases:
        if db in TARGET_DATABASES:
            raise ValueError(f"Refusing to read target database {db}")
        path = workload_dir / db / "workloads.sql"
        rows = read_workload(path, db, "pretrain")
        unique: dict[str, dict[str, object]] = {}
        for row in rows:
            unique.setdefault(row["query_hash"], row)
        unsupported: Counter = Counter()
        eligible: list[dict[str, object]] = []
        for query_hash in sorted(unique):
            row = unique[query_hash]
            try:
                spec = analyze_tree(row["sql"], min_tables)
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
        plan["databases"][db] = {
            "workload_sha256": file_hash(path),
            "rows": len(rows),
            "unique_queries": len(unique),
            "eligible": len(eligible),
            "eligible_fraction": len(eligible) / len(unique),
            "table_count_histogram": dict(
                sorted(
                    Counter(len(item["spec"]["tables"]) for item in eligible).items()
                )
            ),
            "unsupported": dict(sorted(unsupported.items())),
            "queries": [eligible[i] for i in picks],
        }
        print(
            f"{db}: eligible(>={min_tables} tables)={len(eligible)} "
            f"sampled={len(picks)}",
            flush=True,
        )
    write_json(out / "plan.json", plan)
    return plan


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
    parser.add_argument("--min-tables", type=int, default=DEFAULT_MIN_TABLES)
    parser.add_argument("--max-per-db", type=int, default=DEFAULT_MAX_PER_DB)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.stage == "plan":
        if args.workload_dir is None:
            parser.error("plan requires --workload-dir")
        plan = build_plan(
            args.workload_dir,
            args.databases,
            args.min_tables,
            args.max_per_db,
            args.out,
        )
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

    rows = []
    for db, block in plan["databases"].items():
        if db in TARGET_DATABASES:
            raise SystemExit(f"Refusing to read target database {db}")
        dataset = args.dataset_dir / db
        kinds, files = parse_ddl(dataset / f"postgres_create_{db}.sql")
        tables = load_tables(dataset, kinds, files)
        for item in block["queries"]:
            spec = _spec_from_json(item["spec"])
            scored = score_query_tree(spec, tables, float(item["true"]))
            rows.append(
                {
                    "database": db,
                    "query_hash": item["query_hash"],
                    "sql": item["sql"],
                    "pg": float(item["pg"]),
                    "q_pg": q_error(float(item["pg"]), float(item["true"])),
                    **scored,
                }
            )
        print(f"{db}: scored {len(block['queries'])}", flush=True)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise SystemExit("No scorable queries")
    frame.to_csv(args.out / "query_results.csv", index=False)

    summary = {
        "pg": _summarise(frame, "q_pg"),
        "c_uncond": _summarise(frame, "q_uncond"),
        "c_cond": _summarise(frame, "q_cond"),
    }
    by_size = {}
    for size, group in frame.groupby("n_tables"):
        by_size[int(size)] = {
            "n": int(len(group)),
            "pg_q95": percentile(group["q_pg"].tolist(), 95),
            "c_uncond_q95": percentile(group["q_uncond"].tolist(), 95),
            "c_cond_q95": percentile(group["q_cond"].tolist(), 95),
        }
    result = {
        "summary": summary,
        "by_table_count": by_size,
        "max_discrepancy": float(frame["discrepancy"].max()),
        "beats_pg": bool(summary["c_uncond"]["q95"] < summary["pg"]["q95"]),
        "q95_ratio_c_uncond_over_pg": summary["c_uncond"]["q95"] / summary["pg"]["q95"],
    }
    write_json(args.out / "go_no_go.json", result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
