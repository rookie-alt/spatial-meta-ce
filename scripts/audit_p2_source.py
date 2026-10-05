"""P2 source-only gate and independent SQL structure audit. No model or database writes.

The SQLGlot parser is shared with the historical identity pipeline. This auditor
constructs its own reference structure; it does not call the PRICE parser.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sqlglot
from sqlglot import exp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_meta_ce.manifests import canonical_sql, file_hash

DATABASES = ("chembl", "legalacts", "telstra")
EXPECTED_UNIQUE = (50000, 49504, 50000)
COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
PREDICATES = (*COMPARISONS, exp.In, exp.Between, exp.Like, exp.Is)
SUPPORTED = {"EQ", "LT", "GTE"}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def three_valued(operator: str, left: bool | None, right: bool | None = None) -> bool | None:
    if operator == "NOT":
        return None if left is None else not left
    if operator == "AND":
        return False if left is False or right is False else (
            None if left is None or right is None else True
        )
    if operator == "OR":
        return True if left is True or right is True else (
            None if left is None or right is None else False
        )
    raise ValueError("Unsupported logical operation")


def safe_reference_sql(table: str, column: str, value: object) -> tuple[str, tuple[object]]:
    # Defensive illustration only; the source auditor never sends SQL to a DB.
    def quoted(name: str) -> str:
        if not name or any(ch in name for ch in ";\n\r\x00"):
            raise ValueError("Unsafe identifier")
        return '"' + name.replace('"', '""') + '"'

    return f"SELECT COUNT(*) FROM {quoted(table)} WHERE {quoted(column)} = %s", (value,)


def parse_query(sql: str) -> dict:
    tree = sqlglot.parse_one(sql, read="postgres")
    if not isinstance(tree, exp.Select):
        raise ValueError("Expected one SELECT")
    table_nodes = list(tree.find_all(exp.Table))
    tables = [t.alias_or_name for t in table_nodes]
    aliases = {t.alias_or_name: t.name for t in table_nodes}
    where = tree.args.get("where")
    predicates = list(tree.find_all(*PREDICATES))
    columns = list(dict.fromkeys(str(c) for c in tree.find_all(exp.Column)))
    joins = [p for p in predicates if isinstance(p, exp.EQ)
             and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Column)]
    join_columns = {str(c) for p in joins for c in (p.this, p.expression)}
    filters = [p for p in predicates if p not in joins]
    filters_by_col = defaultdict(list)
    for p in filters:
        cs = list(p.find_all(exp.Column))
        for c in cs:
            kind = type(p).__name__
            if isinstance(p.expression, exp.Column) and not isinstance(p.this, exp.Column):
                kind = {"LT": "GT", "LTE": "GTE", "GT": "LT", "GTE": "LTE"}.get(kind, kind)
            filters_by_col[str(c)].append(kind)
    # Use column identities, never the sidecar's order, to generate the reference.
    ordered_join = [c for t in tables for c in columns if c in join_columns and c.split('.')[0] == t]
    ordered_filter = [c for t in tables for c in columns if c in filters_by_col and c.split('.')[0] == t]
    unsupported = set()
    if not where:
        unsupported.add("missing_where")
    if len(table_nodes) != len(tables) or len(set(tables)) != len(tables):
        unsupported.add("duplicate_alias")
    if len({t.name for t in table_nodes}) != len(table_nodes):
        unsupported.add("self_join")
    if len(list(tree.find_all(exp.Select))) != 1 or list(tree.find_all(exp.Subquery)):
        unsupported.add("subquery")
    for name, kind in (("OR", exp.Or), ("NOT", exp.Not), ("IN", exp.In),
                       ("BETWEEN", exp.Between), ("LIKE", exp.Like),
                       ("IS", exp.Is), ("NEQ", exp.NEQ)):
        if tree.find(kind):
            unsupported.add(name)
    if any(j.args.get("on") is not None for j in tree.find_all(exp.Join)):
        unsupported.add("join_on")
    if any(j.args.get("side") in ("LEFT", "RIGHT", "FULL") for j in tree.find_all(exp.Join)):
        unsupported.add("outer_join")
    for c in tree.find_all(exp.Column):
        if not c.table:
            unsupported.add("unqualified_column")
        elif c.table not in aliases:
            unsupported.add("unknown_alias")
    for p in predicates:
        if isinstance(p, COMPARISONS) and isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Column) and not isinstance(p, exp.EQ):
            unsupported.add("non_equi_join")
        if isinstance(p, COMPARISONS) and not isinstance(p.this, exp.Column) and not isinstance(p.expression, exp.Column):
            unsupported.add("non_column_predicate")
    for node, name in ((exp.Func, "function"), (exp.Add, "arithmetic"),
                       (exp.Sub, "arithmetic"), (exp.Mul, "arithmetic"),
                       (exp.Div, "arithmetic"), (exp.Cast, "cast")):
        if where and where.find(node):
            unsupported.add(name)
    if any(x.args.get("quoted") for x in tree.find_all(exp.Identifier)):
        unsupported.add("quoted_identifier")
    if len(joins) != len(tables) - 1:
        unsupported.add("non_tree_join")
    if any(kind not in SUPPORTED for kind in [type(p).__name__ for p in joins] + [name for names in filters_by_col.values() for name in names]):
        unsupported.add("unsupported_predicate")
    if any(not list(p.find_all(exp.Column)) for p in filters):
        unsupported.add("unresolved_filter")
    return dict(
        tables=tables, table_names=[aliases[t] for t in tables],
        join_columns=ordered_join, filter_columns=ordered_filter,
        predicates=sorted({type(p).__name__ for p in joins} | {k for kinds in filters_by_col.values() for k in kinds}),
        boolean_tree=where.this.sql(dialect="postgres") if where else None,
        on_trees=[j.args["on"].sql(dialect="postgres") for j in tree.find_all(exp.Join) if j.args.get("on")],
        repeated={c: kinds for c, kinds in filters_by_col.items() if len(kinds) > 1},
        unsupported=sorted(unsupported),
        counts=[len(ordered_join), 2 * len(joins), len(tables), len(ordered_filter)],
        join_column_tables=[tables.index(c.split('.')[0]) for c in ordered_join],
        filter_column_tables=[tables.index(c.split('.')[0]) for c in ordered_filter],
        filter_expressions=[p.sql(dialect="postgres") for p in filters],
    )


def source_files(project: Path, source: Path, db: str) -> dict[str, Path]:
    cache = project / "artifacts/cache/pretrain" / db
    stats = source / "datas/statistics/pretrain" / db
    return {
        "workload": source / "datas/workloads/pretrain" / db / "workloads.sql",
        "source_features": source / "setup/features/pretrain" / db / "features40.pkl",
        "extractor": project / "vendor/price_features/features_tool.py",
        **{f"statistics/{p.name}": p for p in stats.glob("*.pkl")},
        **{f"cache/{p.name}": p for p in cache.iterdir() if p.is_file()},
    }


def report_digest(path: Path) -> str:
    payload = json.dumps(json.loads(path.read_text()), sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_db(project: Path, source: Path, db: str, matrix: dict) -> dict:
    sidecar = project / "runs/mapping-sidecars-source26-20261004" / db
    cache = project / "artifacts/cache/pretrain" / db
    mpath = sidecar / "manifest.json"
    manifest = json.loads(mpath.read_text())
    frozen = json.loads((cache / "manifest.json").read_text())
    if manifest["database"] != db or manifest["split"] != "pretrain":
        raise ValueError(f"Wrong manifest identity: {db}")
    files = source_files(project, source, db)
    checked = {}
    required = {"workload", "source_features", "extractor", "cache/manifest.json",
                "cache/cards.npy", "cache/offsets.npy", "cache/group2.npy",
                "cache/group3.npy", "statistics/histogram40.pkl", "statistics/size.pkl"}
    if not required.issubset(manifest["input_sha256"]):
        raise ValueError(f"Missing frozen input hashes: {db}")
    for name, expected in manifest["input_sha256"].items():
        path = files.get(name)
        if path is None or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Input SHA256 mismatch: {db}/{name}")
        checked[name] = expected
    if manifest["cache_content_hash"] != frozen["content_hash"]:
        raise ValueError(f"Cache content hash mismatch: {db}")
    if file_hash(sidecar / "mapping.jsonl") != manifest["jsonl_sha256"]:
        raise ValueError(f"Mapping JSONL mismatch: {db}")
    arm_rows = [r for r in matrix["runs"] if r["arm"] in ("true", "shuffled")]
    if len(arm_rows) != 6 or {r["mapping_provenance"][db]["manifest_hash"] for r in arm_rows} != {file_hash(mpath)}:
        raise ValueError(f"Matrix mapping manifest mismatch: {db}")
    if {r["manifest_hashes"][db] for r in matrix["runs"]} != {frozen["content_hash"]}:
        raise ValueError(f"Matrix cache content mismatch: {db}")
    reports = list((project / "runs/p1-input-audit-20261002").glob(f"group-*/{db}.json"))
    if len(reports) != 1 or report_digest(reports[0]) != manifest["audit_report_sha256"]:
        raise ValueError(f"Frozen P1 audit report mismatch: {db}")
    return dict(input_sha256=checked, mapping_manifest_sha256=file_hash(mpath),
                mapping_jsonl_sha256=manifest["jsonl_sha256"],
                audit_report_sha256=manifest["audit_report_sha256"],
                cache_content_hash=frozen["content_hash"],
                cache_manifest=frozen, source_path=str(files["workload"]),
                mapping_path=str(sidecar / "mapping.jsonl"),
                cache_path=str(cache))


def scan_database(project: Path, source: Path, db: str, gate: dict, out: Path) -> dict:
    cache = Path(gate["cache_path"])
    manifest = gate["cache_manifest"]
    by_line = {r["source_line"]: (i, r) for i, r in enumerate(manifest["rows"])}
    if len(by_line) != len(manifest["rows"]):
        raise ValueError(f"Duplicate cache line: {db}")
    offsets = np.load(cache / "offsets.npy", mmap_mode="r", allow_pickle=False)
    cards = np.load(cache / "cards.npy", mmap_mode="r", allow_pickle=False)
    if offsets.shape != (len(by_line) + 1, 4) or len(cards) != len(by_line):
        raise ValueError(f"Wrong frozen array shapes: {db}")
    maps = {}
    covered = set()
    with Path(gate["mapping_path"]).open() as stream:
        for raw in stream:
            row = json.loads(raw)
            h = row["query_hash"]
            if h in maps or row["source_line"] not in row["source_lines"]:
                raise ValueError(f"Duplicate/broken mapping group: {db}/{h}")
            maps[h] = row
            for line in row["source_lines"]:
                if line in covered:
                    raise ValueError(f"Source line covered twice: {db}/{line}")
                covered.add(line)
    if len(maps) != EXPECTED_UNIQUE[DATABASES.index(db)]:
        raise ValueError(f"Unexpected unique-query count: {db}")
    rows, unique, mismatches, risk, old_risk = 0, set(), Counter(), Counter(), Counter()
    repeats, examples = Counter(), []
    with Path(gate["source_path"]).open() as stream, (out / f"{db}-parse_rows.jsonl").open("w") as sink:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank workload row: {db}/{line_no}")
            parts = line.rstrip("\n").split("||")
            if len(parts) < 3:
                raise ValueError(f"Malformed workload row: {db}/{line_no}")
            sql = parts[0].strip()
            identity = hashlib.sha256(canonical_sql(sql).encode()).hexdigest()
            exact = hashlib.sha256(sql.encode()).hexdigest()
            row = maps.get(identity)
            if row is None or line_no not in row["source_lines"]:
                raise ValueError(f"Missing unique identity: {db}/{line_no}")
            if exact != row["exact_hash"]:
                raise ValueError(f"Exact SQL mismatch in repeated group: {db}/{line_no}")
            index, frozen = by_line[line_no]
            if any(frozen.get(key) != value for key, value in (
                ("id", f"{db}/pretrain/{line_no}"), ("exact_hash", exact),
                ("query_hash", identity))):
                raise ValueError(f"Frozen row identity mismatch: {db}/{line_no}")
            if (not math.isclose(float(parts[1]), float(cards[index, 0]), rel_tol=1e-6)
                    or not math.isclose(float(parts[2]), float(cards[index, 1]), rel_tol=1e-6)):
                raise ValueError(f"Frozen label mismatch: {db}/{line_no}")
            parsed = parse_query(sql)
            old = row["mapping"]
            independent = parsed["counts"]
            baseline = old["counts"]
            errors = []
            for key in ("tables", "table_names", "join_columns", "filter_columns",
                        "join_column_tables", "filter_column_tables", "predicates"):
                target = "predicate_types" if key == "predicates" else key
                if parsed[key] != old[target]:
                    errors.append(key)
            if independent != baseline or independent != (offsets[index + 1] - offsets[index]).tolist():
                errors.append("counts/offsets")
            if errors:
                mismatches.update(errors)
                if len(examples) < 12:
                    examples.append({"line": line_no, "errors": errors})
            risk.update(parsed["unsupported"])
            old_risk.update(old["flags"])
            repeats[identity] += 1
            unique.add(identity)
            record = dict(database=db, split="pretrain", source_line=line_no,
                          exact_hash=exact, query_hash=identity, status="mismatch" if errors else
                          ("unsupported" if parsed["unsupported"] else "parsed"),
                          differences=errors, **parsed)
            sink.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            rows += 1
    if rows != 50000 or covered != set(range(1, rows + 1)) or len(unique) != len(maps):
        raise ValueError(f"Incomplete source rows / unique-query mapping: {db}")
    if any(repeats[h] != len(row["source_lines"]) for h, row in maps.items()):
        raise ValueError(f"Inconsistent repeated query group: {db}")
    summary = dict(database=db, status="completed" if not mismatches and not risk else "failed",
                   source_rows=rows, unique_queries=len(unique), duplicate_groups=sum(v>1 for v in repeats.values()),
                   mismatches=dict(mismatches), unsupported=dict(risk), legacy_flags=dict(old_risk),
                   mismatch_examples=examples, parse_rows_file=f"{db}-parse_rows.jsonl",
                   parse_rows_sha256=file_hash(out / f"{db}-parse_rows.jsonl"))
    write_json(out / f"{db}-parse_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    project, source, out = args.project.resolve(), args.source.resolve(), args.out.resolve()
    if out == project or not out.is_relative_to(project / "runs") or out.exists():
        raise ValueError("Output must be a fresh runs/ subdirectory")
    if sqlglot.__version__ != "18.17.0":
        raise ValueError("Frozen SQLGlot version required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES= to enforce CPU-only execution")
    out.mkdir(parents=True)
    matrix_path = project / "runs/p1-mapping-ablation-20261004/ablation_runs.json"
    matrix = json.loads(matrix_path.read_text())
    protocol_path = project / "实验文档/13-P2源侧语义与真实选择率审计协议.md"
    metadata = dict(created_at=datetime.now(timezone.utc).astimezone().isoformat(),
                    protocol_sha256=file_hash(protocol_path), matrix_sha256=file_hash(matrix_path),
                    script_sha256=file_hash(Path(__file__)), python=platform.python_version(),
                    numpy=np.__version__, sqlglot=sqlglot.__version__,
                    source=str(source), databases=DATABASES, workers=1, gpu_visible=False,
                    training=False, inference=False, target_data_read=False,
                    database_access=False)
    write_json(out / "protocol.json", metadata)
    gates = {}
    try:
        for db in DATABASES:
            gates[db] = verify_db(project, source, db, matrix)
    except (OSError, KeyError, ValueError) as error:
        write_json(out / "preflight.json", dict(status="failed", reason=str(error), databases=list(gates)))
        raise
    write_json(out / "preflight.json", dict(status="completed", databases={db: {
        k:v for k,v in info.items() if k not in ("cache_manifest", "mapping_path", "cache_path", "source_path")
    } for db, info in gates.items()}))
    results=[]
    for db in DATABASES:
        try:
            results.append(scan_database(project, source, db, gates[db], out))
        except (OSError, KeyError, ValueError, sqlglot.errors.ParseError) as error:
            write_json(out / "summary.json", dict(status="failed", database=db, reason=str(error), completed=results))
            raise
    status="completed" if all(x["status"]=="completed" for x in results) else "failed"
    write_json(out / "summary.json", dict(status=status, databases=results, phase_b="planned", phase_g1="planned", phase_c="planned",
                                          training=False, inference=False, target_data_read=False))
    return 0 if status=="completed" else 1


if __name__ == "__main__":
    sys.exit(main())
