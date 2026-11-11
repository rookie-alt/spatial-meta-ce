#!/usr/bin/env python3
"""Fill the missing IMDB finetune statistics that PRICE ships incomplete.

The released `datas/statistics/finetune/imdb` covers only 11.3% of the 50k
finetune workload, so feature generation raises KeyError on join columns such
as `imdb_mk.keyword_id`. This script computes only the missing histogram,
summary and fanout entries from the raw IMDB CSVs, reusing PRICE's own
statistics semantics (utils/statistics/*.py), then merges them into copies of
the shipped pickles.

Run with `--validate` first: it recomputes entries that already exist in the
shipped files and aborts on any mismatch, proving the semantics match before
new entries are trusted.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sqlglot

BIN_SIZE = 40


def lower_except_quotes(text: str) -> str:
    inside_quote = False
    quote_char = ""
    out = []
    for char in text:
        if char in "'\"" and (not inside_quote or quote_char == char):
            inside_quote = not inside_quote
            quote_char = "" if not inside_quote else char
        out.append(char)
    return "".join(out)


def read_schema(ddl_path: Path):
    """Return ({table: {column: kind}}, {table: csv_filename})."""
    kinds = {}
    copies = {}
    table = None
    for line in ddl_path.read_text().splitlines():
        low = line.lower().strip()
        if low.startswith("create table"):
            table = line.split("create table")[1].split()[0].strip('"()')
            kinds[table] = {}
        elif low.startswith("\\copy"):
            parts = line.split()
            copies[parts[1].strip("'")] = parts[3].strip("'")
        elif table and line.startswith("    ") and line.strip():
            column = line.strip().split(" ")[0].strip('"')
            if "integer" in low or "bigint" in low or "smallint" in low:
                kinds[table][column] = "int"
            elif "character" in low or "varchar(" in low or "char(" in low:
                kinds[table][column] = "str"
            elif "double precision" in low:
                kinds[table][column] = "float"
            elif "decimal(" in low:
                kinds[table][column] = "decimal"
    return kinds, copies


def workload_requirements(workload_path: Path):
    """Return (histogram columns, summary columns, directed join pairs)."""
    hist, summary, pairs = set(), set(), set()
    predicate_free = 0
    for line in workload_path.read_text().splitlines():
        tree = sqlglot.parse_one(lower_except_quotes(line.split("||")[0]))
        where = tree.args.get("where")
        joins = []
        if where is not None:
            for eq in where.find_all(sqlglot.exp.EQ):
                if isinstance(eq.args.get("expression"), sqlglot.exp.Column):
                    joins.append((str(eq.args["this"]), str(eq.args["expression"])))
        join_columns = {column for pair in joins for column in pair}
        all_columns = {str(column) for column in tree.find_all(sqlglot.exp.Column)}
        filter_columns = all_columns - join_columns
        eq_columns, range_columns = set(), set()
        if where is not None:
            for eq in where.find_all(sqlglot.exp.EQ):
                if isinstance(eq.args.get("expression"), sqlglot.exp.Column):
                    continue
                name = str(eq.args["this"])
                if name not in join_columns:
                    eq_columns.add(name)
            for node in where.find_all(
                sqlglot.exp.GT,
                sqlglot.exp.LT,
                sqlglot.exp.GTE,
                sqlglot.exp.LTE,
                sqlglot.exp.Between,
            ):
                for column in node.find_all(sqlglot.exp.Column):
                    name = str(column)
                    if name not in join_columns:
                        range_columns.add(name)
        bare = filter_columns - eq_columns - range_columns
        predicate_free += len(bare)
        hist |= {tuple(column.split(".")) for column in join_columns | range_columns | bare}
        summary |= {tuple(column.split(".")) for column in eq_columns}
        pairs |= set(joins)
    print(f"columns used without a predicate: {predicate_free}")
    return hist, summary, pairs


def load_columns(csv_path: Path, columns, kinds):
    dtype = {
        column: {"int": "Int64", "float": "Float64", "str": "string", "decimal": "float64"}[
            kinds[column]
        ]
        for column in columns
    }
    frame = pd.read_csv(
        csv_path,
        sep="|",
        quotechar='"',
        escapechar="\\",
        dtype=dtype,
        usecols=sorted(columns),
        keep_default_na=False,
        na_values=["NULL"],
    )
    return {column: frame[column] for column in columns}


def histogram(series, bin_size: int = BIN_SIZE):
    values = series.dropna().astype(float)
    length = int(series.shape[0])
    if len(values) == 0:
        return {
            "hist": np.zeros(bin_size),
            "bin_edges": np.zeros(bin_size),
            "len": 0,
            "min_value": 0,
            "max_value": 0,
        }
    hist, bin_edges = np.histogram(values, bins=bin_size, density=False)
    return {
        "hist": hist.astype(float),
        "bin_edges": bin_edges,
        "len": length,
        "min_value": min(values),
        "max_value": max(values),
    }


def summary(series):
    counts = series.dropna().value_counts()
    keys = [key.item() if hasattr(key, "item") else key for key in counts.index]
    return {"keys": keys, "values": [int(value) for value in counts.to_numpy()]}


def pair_fanout(left: str, right: str, tables, histograms, bin_size: int = BIN_SIZE):
    left_table, left_column = left.split(".")
    right_table, right_column = right.split(".")
    counts = pd.DataFrame(tables[right_table][right_column].value_counts())
    count_column = counts.columns[0]
    merged = pd.merge(
        tables[left_table][left_column],
        counts,
        left_on=left_column,
        right_on=right_column,
        how="left",
    )
    edges = histograms[left_table][left_column]["bin_edges"]
    values = []
    for index in range(len(edges) - 1):
        if index != bin_size - 1:
            mask = (merged[left_column] >= edges[index]) & (
                merged[left_column] < edges[index + 1]
            )
        else:
            mask = (merged[left_column] >= edges[index]) & (
                merged[left_column] <= edges[index + 1]
            )
        selected = merged.loc[mask, count_column]
        total = selected.sum()
        values.append(0 if total == 0 else total / len(selected))
    return values


def validate(tables, histograms, shipped, samples, alias_to_table):
    alias_tables = {
        alias: tables[table]
        for alias, table in alias_to_table.items()
        if table in tables
    }
    for table, column in samples["histogram"]:
        expected = shipped["histogram"][table][column]
        actual = histograms[table][column]
        if not np.allclose(expected["hist"], actual["hist"], rtol=1e-12, atol=0):
            raise SystemExit(f"histogram mismatch at {table}.{column}")
        if not np.allclose(expected["bin_edges"], actual["bin_edges"], rtol=1e-12, atol=0):
            raise SystemExit(f"bin edge mismatch at {table}.{column}")
        if int(expected["len"]) != int(actual["len"]):
            raise SystemExit(f"row count mismatch at {table}.{column}")
        print(f"histogram ok: {table}.{column}")
    for table, column in samples["summary"]:
        expected = dict(
            zip(
                shipped["summary"][table][column]["keys"],
                shipped["summary"][table][column]["values"],
            )
        )
        actual = summary(alias_tables[table][column])
        if expected != dict(zip(actual["keys"], actual["values"])):
            raise SystemExit(f"summary mismatch at {table}.{column}")
        print(f"summary ok: {table}.{column}")
    for left, right in samples["fanout"]:
        expected = shipped["fanout"].get((left, right))
        if expected is None:
            print(f"fanout sample skipped (absent upstream): {left} = {right}")
            continue
        if not np.allclose(
            expected[0], pair_fanout(left, right, alias_tables, histograms), rtol=1e-12, atol=0
        ):
            raise SystemExit(f"fanout mismatch at {left} = {right}")
        if not np.allclose(
            expected[1], pair_fanout(right, left, alias_tables, histograms), rtol=1e-12, atol=0
        ):
            raise SystemExit(f"inverse fanout mismatch at {left} = {right}")
        print(f"fanout ok: {left} = {right}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="/var/lib/ds/datasets")
    parser.add_argument("--database", default="imdb")
    parser.add_argument(
        "--stats", default="/gly/weilin/PRICE-master/datas/statistics/finetune/imdb"
    )
    parser.add_argument(
        "--workload",
        default="/gly/weilin/PRICE-master/datas/workloads/finetune/imdb/workloads.sql",
    )
    parser.add_argument(
        "--out", default="/gly/weilin/PRICE-derived/datas/statistics/finetune/imdb"
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    stats_dir = Path(args.stats)
    dataset_dir = Path(args.datasets) / args.database
    kinds, copies = read_schema(dataset_dir / f"postgres_create_{args.database}.sql")

    shipped = {
        "histogram": pickle.load(open(stats_dir / "histogram40.pkl", "rb")),
        "summary": pickle.load(open(stats_dir / "summary40.pkl", "rb")),
        "fanout": pickle.load(open(stats_dir / "fanout40.pkl", "rb")),
        "coltype": pickle.load(open(stats_dir / "abbrev_col_type.pkl", "rb")),
        "size": pickle.load(open(stats_dir / "size.pkl", "rb")),
    }
    alias_to_table = {alias: table for table, alias in shipped["coltype"]["abbrev"].items()}

    hist_cols, summary_cols, pairs = workload_requirements(Path(args.workload))
    missing_hist = {
        column
        for column in hist_cols
        if column[1] not in shipped["histogram"].get(column[0], {})
    }
    missing_summary = {
        column
        for column in summary_cols
        if column[1] not in shipped["summary"].get(column[0], {})
    }
    missing_pairs = {pair for pair in pairs if pair not in shipped["fanout"]}
    print(
        f"missing histogram {len(missing_hist)} summary {len(missing_summary)} "
        f"fanout {len(missing_pairs)}"
    )

    samples = {
        "histogram": [("imdb_ci", "movie_id"), ("imdb_t", "id"), ("imdb_t", "production_year")],
        "summary": [("imdb_ci", "role_id"), ("imdb_t", "kind_id")],
        "fanout": [("imdb_ci.movie_id", "imdb_t.id")],
    }
    required = missing_hist | missing_summary
    if args.validate:
        required |= {tuple(item) for item in samples["histogram"] + samples["summary"]}

    needed = {}
    for alias, column in required:
        table = alias_to_table[alias]
        if column not in kinds.get(table, {}):
            raise SystemExit(f"unknown column {alias}.{column}")
        if (alias, column) in missing_hist and kinds[table][column] not in {
            "int",
            "float",
            "decimal",
        }:
            raise SystemExit(f"non-numeric histogram column {alias}.{column}")
        needed.setdefault(table, set()).add(column)

    tables = {}
    for table, columns in sorted(needed.items()):
        for column, series in load_columns(
            dataset_dir / copies[table], columns, kinds[table]
        ).items():
            tables.setdefault(table, {})[column] = series
        print(f"loaded {table}: {sorted(columns)}")

    hist_needed = set(missing_hist)
    if args.validate:
        hist_needed |= {tuple(item) for item in samples["histogram"]}
    histograms = {}
    for alias, column in sorted(hist_needed):
        histograms.setdefault(alias, {})[column] = histogram(
            tables[alias_to_table[alias]][column]
        )

    if args.validate:
        validate(tables, histograms, shipped, samples, alias_to_table)
        print("validation passed")
        return

    if not args.write:
        raise SystemExit("choose --validate or --write")

    for alias, column in sorted(missing_hist):
        shipped["histogram"].setdefault(alias, {})[column] = histograms[alias][column]
        shipped["coltype"]["col_type"][alias]["ctn"].append(column)
    for alias, column in sorted(missing_summary):
        shipped["summary"].setdefault(alias, {})[column] = summary(
            tables[alias_to_table[alias]][column]
        )
        if column not in shipped["coltype"]["col_type"][alias]["dsct"]:
            shipped["coltype"]["col_type"][alias]["dsct"].append(column)

    done = set()
    for left, right in sorted(missing_pairs):
        if (left, right) in done:
            continue
        left_fanout = pair_fanout(left, right, tables, shipped["histogram"])
        right_fanout = pair_fanout(right, left, tables, shipped["histogram"])
        shipped["fanout"][(left, right)] = [left_fanout, right_fanout]
        shipped["fanout"][(right, left)] = [right_fanout, left_fanout]
        done.update({(left, right), (right, left)})
        print(f"fanout added: {left} = {right}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("histogram", "summary", "fanout"):
        pickle.dump(shipped[name], open(out / f"{name}40.pkl", "wb"))
    pickle.dump(shipped["coltype"], open(out / "abbrev_col_type.pkl", "wb"))
    pickle.dump(shipped["size"], open(out / "size.pkl", "wb"))
    print("written", out)


if __name__ == "__main__":
    main()