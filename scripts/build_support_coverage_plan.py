#!/usr/bin/env python3
"""Build the protocol-90 stage-A support-coverage plan (45 fresh draws).

Protocol 90 section 3-A3 asks for 45 support draws beyond the five frozen
protocol-81 runs.  This builder writes them to a NEW file; the frozen
``plan-20261018.json`` is read-only and is never rewritten or merged into.

Sampling rule (protocol 91 section 3), identical in shape to protocol 81:

    sample  = random.Random(f"{seed}:support-coverage:{db}:{run}").sample(indices, 100)
    support = sample
    query   = [i for i in indices if i not in set(sample)]
    supports["20"] = sample[:20]

``indices`` comes from ``build_meta65_holdout_plan.validate_store`` so the
unique-query denominator is the same function, not a copy, as protocol 81.

Only the three source validation databases are accepted.  Target databases
imdb/stats/genome/ergastf1 and the holdout databases hepatitis/sakila are
rejected before any path is constructed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_meta65_holdout_plan as holdout_plan  # noqa: E402
from scripts import meta65_holdout_contract as contract  # noqa: E402

DATABASES = ("chembl", "legalacts", "telstra")
FORBIDDEN = ("imdb", "stats", "genome", "ergastf1")
EXCLUDED = ("hepatitis", "sakila")
SEED_NEW = 20261019
RUNS = tuple(range(5, 50))
SUPPORT_SIZE = 100
K = 20
SHOTS = (5, 10, 20, 50, 100)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def guard_name(database):
    """Reject every database that stage A must not touch."""
    if database in FORBIDDEN:
        raise ValueError("Target database reached protocol 90 stage A: %s" % database)
    if database in EXCLUDED:
        raise ValueError("Holdout database excluded from stage A: %s" % database)
    if database not in DATABASES:
        raise ValueError("Unexpected database in protocol 90 stage A: %s" % database)


def guard_path(path):
    for part in Path(path).parts:
        if part in FORBIDDEN:
            raise ValueError("Target database appears in a path: %s" % path)


def draw(indices, hashes, database, run):
    """Draw one support set and derive its complementary query set."""
    sample = random.Random(
        "%d:support-coverage:%s:%d" % (SEED_NEW, database, run)
    ).sample(indices, SUPPORT_SIZE)
    excluded = set(sample)
    query = [index for index in indices if index not in excluded]
    identity = dict(zip(indices, hashes, strict=True))
    plan = {
        "database": database,
        "run": run,
        "support_max": sample,
        "supports": {str(k): sample[:k] for k in SHOTS},
        "query": query,
        "query_digest": contract.digest([(index, identity[index]) for index in query]),
        "support_digests": {
            str(k): contract.digest([(index, identity[index]) for index in sample[:k]])
            for k in SHOTS
        },
    }
    certify_draw(plan, indices, hashes, database, run)
    return plan


def certify_draw(plan, indices, hashes, database, run):
    """Re-check the invariants protocol 91 section 0 requires jointly."""
    # Indices are cache row numbers, not positions inside ``indices``: they are
    # bounded by the store length and must all belong to the first-unique set.
    population = set(indices)
    support, query = plan["support_max"], plan["query"]
    if plan["database"] != database or plan["run"] != run:
        raise ValueError("Plan database/run mismatch")
    if len(support) != SUPPORT_SIZE or len(set(support)) != SUPPORT_SIZE:
        raise ValueError("Support must hold %d unique indices" % SUPPORT_SIZE)
    if (any(type(value) is not int or value not in population
            for value in (*support, *query))
            or len(set(query)) != len(query) or set(support) & set(query)):
        raise ValueError("Plan changed support exclusion or denominator")
    expected = [index for index in indices if index not in set(support)]
    if query != expected:
        raise ValueError("Query is not the full population minus the support set")
    identity = dict(zip(indices, hashes, strict=True))
    if plan["query_digest"] != contract.digest([(i, identity[i]) for i in query]):
        raise ValueError("Query digest changed")
    if set(plan["supports"]) != {str(k) for k in SHOTS}:
        raise ValueError("Full nested support plan required")
    for k in SHOTS:
        if (plan["supports"][str(k)] != support[:k]
                or plan["support_digests"][str(k)]
                != contract.digest([(i, identity[i]) for i in support[:k]])):
            raise ValueError("Support prefix/digest changed")


def build(data, expected):
    from spatial_meta_ce.data import FeatureStore

    cache_root = Path(str(data["cache_root"])) / "pretrain"
    plans = {}
    for database in DATABASES:
        guard_name(database)
        store = FeatureStore(cache_root / database)
        indices, hashes = holdout_plan.validate_store(
            store, database, expected["unique"][database], expected["four_plus"][database]
        )
        plans[database] = [draw(indices, hashes, database, run) for run in RUNS]
    return plans


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", type=Path,
                        default=ROOT / "configs/meta65_source_holdout_data.json")
    parser.add_argument("--expected-config", type=Path,
                        default=ROOT / "configs/meta65_source_holdout.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    guard_path(args.out)
    if args.out.exists():
        raise FileExistsError("Plan output must be fresh")

    data = contract.validate_data(contract.read_json(args.data_config))
    expected = contract.load_config(args.expected_config)["expected"]
    plans = build(data, expected)
    report = {
        "schema": 1,
        "protocol": "90 阶段A support-coverage plan",
        "seed_new": SEED_NEW,
        "support_size": SUPPORT_SIZE,
        "k": K,
        "runs": list(RUNS),
        "reused_from_81": list(contract.RUNS),
        "databases": plans,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    contract.write_json(args.out, report)
    print(json.dumps({
        "out": str(args.out),
        "sha256": sha256(args.out),
        "databases": {db: len(runs) for db, runs in plans.items()},
        "runs": [RUNS[0], RUNS[-1]],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
