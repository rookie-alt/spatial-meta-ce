import hashlib
import json
import math
from pathlib import Path

import numpy as np


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def validate_splits(splits, *, allow_empty_validation=False):
    keys = ("train", "validation", "test")
    for key in keys:
        if not splits.get(key) or len(set(splits[key])) != len(splits[key]):
            if not (allow_empty_validation and key == "validation" and splits.get(key) == []):
                raise ValueError(f"Invalid database list: {key}")
        if any("/" in db or db in (".", "..") for db in splits[key]):
            raise ValueError("Invalid database name")
    for i, key in enumerate(keys):
        for other in keys[i + 1 :]:
            if set(splits[key]) & set(splits[other]):
                raise ValueError(f"Database leakage between {key} and {other}")


def canonical_sql(sql):
    import sqlglot

    if sqlglot.__version__ != "18.17.0":
        raise ValueError(
            "Use PRICE environment with sqlglot==18.17.0 for stable identities"
        )
    # Normalize unquoted identifiers but never literal case/value. Not an
    # exhaustive SQL equivalence proof; join order remains significant.
    tree = sqlglot.parse_one(sql, read="postgres")
    from sqlglot import exp

    for node in tree.find_all(exp.Identifier):
        if not node.args.get("quoted"):
            node.set("this", node.this.lower())
    return tree.sql(dialect="postgres", pretty=False, comments=False).rstrip(";")


def read_workload(path, database, split):
    rows = []
    seen = {}
    with Path(path).open() as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            fields = line.rstrip().split("||")
            if len(fields) < 3:
                raise ValueError(f"Malformed workload line {line_number}")
            sql, true, pg = fields[0].strip(), float(fields[1]), float(fields[2])
            if not math.isfinite(true) or not math.isfinite(pg) or true < 0 or pg < 0:
                raise ValueError(f"Invalid cardinality at line {line_number}")
            normalized = canonical_sql(sql)
            h = hashlib.sha256(normalized.encode()).hexdigest()
            if h in seen and not np.isclose(seen[h], true, rtol=1e-6, atol=0):
                raise ValueError(
                    f"Inconsistent labels for repeated SQL at {line_number}"
                )
            seen[h] = true
            rows.append(
                {
                    "id": f"{database}/{split}/{line_number}",
                    "query_hash": h,
                    "exact_hash": hashlib.sha256(sql.encode()).hexdigest(),
                    "source_line": line_number,
                    "sql": sql,
                    "true": true,
                    "pg": pg,
                }
            )
    if not rows:
        raise ValueError("Empty workload")
    return rows


def episode_indices(pool, support_size, query_size, rng):
    pool = np.asarray(pool, dtype=np.int64)
    if (
        support_size < 0
        or query_size < 1
        or len(pool) < support_size + query_size
        or len(np.unique(pool)) != len(pool)
    ):
        raise ValueError("Insufficient unique queries or invalid episode size")
    selected = rng.choice(pool, size=support_size + query_size, replace=False)
    return selected[:support_size], selected[support_size:]
