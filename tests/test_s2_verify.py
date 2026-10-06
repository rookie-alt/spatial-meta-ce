"""Synthetic S2 oracle tests, without production replay/statistics imports."""

import copy
import hashlib
import json
import unittest

import numpy as np

from scripts.s2_verify import verify


CARDS = [
    "0",
    "(0,10)",
    "[10,100)",
    "[100,1000)",
    "[1000,10000)",
    "[10000,100000)",
    "[100000,inf)",
]
TABLES = ["1-3", "4", "5", "6+"]
BUCKETS = (
    [("all", "all")]
    + [("cardinality", x) for x in CARDS]
    + [("n_tables", x) for x in TABLES]
    + [("cross", f"{c}|{t}") for c in CARDS for t in TABLES]
)
ACTIVE = {"all", "[10,100)", "5", "[10,100)|5"}


def fixture():
    train, validation = ["a", "b"], ["v"]
    caches = {
        db: {
            "true": np.full(n, 10.0),
            "pg": np.full(n, 10.0),
            "n_tables": np.full(n, 5),
            "raw_rows": 2 * n + 3,
            "indices": np.arange(n) * 2,
            "query_hash": [str(i) for i in range(n)],
        }
        for db, n in [("a", 240), ("b", 220), ("v", 200)]
    }
    config = dict(
        seed=42,
        steps=4000,
        meta_batch=1,
        query_size=1,
        support_sizes=[1, 2],
        accumulation=1,
    )
    rng = np.random.default_rng(43)
    chain = hashlib.sha256(b"").hexdigest()
    states = {
        db: dict(
            support=np.zeros(len(caches[db]["true"]), dtype=int),
            query=np.zeros(len(caches[db]["true"]), dtype=int),
            mass=np.zeros(len(caches[db]["true"])),
            episodes=0,
            k_counts={1: 0, 2: 0},
        )
        for db in train
    }
    replay = {"prefixes": {}}
    total = 0
    for step in range(1, 4001):
        db, k = str(rng.choice(train)), int(rng.choice([1, 2]))
        raw = rng.choice(caches[db]["indices"], k + 1, replace=False)
        chosen = raw // 2
        episode = dict(
            db=db, support=[str(i) for i in chosen[:k]], query=[str(chosen[-1])]
        )
        chain = hashlib.sha256(
            json.dumps(
                dict(previous=chain, episode=episode),
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        states[db]["support"][chosen[:k]] += 1
        states[db]["query"][chosen[-1]] += 1
        states[db]["mass"][chosen] += 1 / (k + 1)
        states[db]["episodes"] += 1
        states[db]["k_counts"][k] += 1
        total += k + 1
        if step in (1000, 2000, 4000):
            replay["prefixes"][step] = dict(
                databases=copy.deepcopy(states),
                digest=chain,
                episodes=step,
                presentations=total,
            )
    replay["digest"] = chain
    tables = dict(pool=[], exposure=[], residuals=[], comparison=[])
    groups = [(db, [db], False) for db in train + validation] + [
        ("train_pooled", train, False),
        ("train_db_equal_pool", train, True),
        ("validation_db_equal_pool", validation, True),
    ]

    def residual(name, members, pop, step, weights):
        n = sum(len(caches[d]["true"]) for d in members)
        observed = n if weights is None else np.count_nonzero(weights)
        mass = n if weights is None else float(weights.sum())
        for axis, bucket in BUCKETS:
            active = bucket in ACTIVE
            ok = active and observed >= 200
            row = dict(
                database=name,
                population=pop,
                step=step,
                axis=axis,
                bucket=bucket,
                source_db_coverage=len(members) if active else 0,
                pool_unique=n if active else 0,
                observed_unique=int(observed) if active else 0,
                weight_mass=mass if active else 0.0,
                status="ok" if ok else "sparse",
                quantile_method="linear" if weights is None else "inverse_ecdf",
            )
            row.update(
                {
                    f"{kind}_{stat}": 0.0 if ok else None
                    for kind in ["r_ref", "r_fp32"]
                    for stat in ["mean", "median", "p10", "p90"]
                }
            )
            row.update(
                {
                    f"{sign}_share": float(sign == "zero") if ok else None
                    for sign in ["positive", "negative", "zero"]
                }
            )
            tables["residuals"].append(row)

    for name, members, equal in groups:
        n = sum(len(caches[d]["true"]) for d in members)
        weights = (
            np.concatenate(
                [
                    np.full(len(caches[d]["true"]), 1 / len(caches[d]["true"]))
                    for d in members
                ]
            )
            if equal
            else None
        )
        pop = "db_equal_pool" if equal else "pool_unique"
        for axis, bucket in BUCKETS:
            active = bucket in ACTIVE
            tables["pool"].append(
                dict(
                    database=name,
                    population=pop,
                    axis=axis,
                    bucket=bucket,
                    raw_rows=2 * n + 3 * len(members),
                    duplicate_rows=n + 3 * len(members),
                    pool_unique=n if active else 0,
                    pool_total=n,
                    weight_mass=(len(members) if equal else n) if active else 0,
                    share=float(active),
                    source_db_coverage=len(members) if active else 0,
                )
            )
        residual(name, members, pop, 0, weights)
    summaries = {}
    for step, prefix in replay["prefixes"].items():
        summaries[str(step)] = {}
        global_sums = {
            key: sum(s[key].sum() for s in prefix["databases"].values())
            for key in ["support", "query", "mass"]
        }
        global_sums["combined"] = global_sums["support"] + global_sums["query"]
        for name, members in [(d, [d]) for d in train] + [("train_pooled", train)]:
            arrays = {
                key: np.concatenate([prefix["databases"][d][key] for d in members])
                for key in ["support", "query", "mass"]
            }
            arrays["combined"] = arrays["support"] + arrays["query"]
            n = len(arrays["mass"])
            observed = int(np.count_nonzero(arrays["combined"]))
            for axis, bucket in BUCKETS:
                active = bucket in ACTIVE
                row = dict(
                    database=name,
                    step=step,
                    axis=axis,
                    bucket=bucket,
                    pool_unique=n if active else 0,
                    observed_unique=observed if active else 0,
                    coverage=observed / n if active else "undefined",
                    source_db_coverage=len(members) if active else 0,
                )
                for key, label in [
                    ("support", "support"),
                    ("query", "query"),
                    ("combined", "combined_presentations"),
                    ("mass", "nominal_mass"),
                ]:
                    value = float(arrays[key].sum()) if active else 0
                    row.update(
                        {
                            label: value,
                            label + "_share": float(active),
                            label + "_global_share": value / global_sums[key],
                        }
                    )
                tables["exposure"].append(row)
            residual(name, members, "combined_presentations", step, arrays["combined"])
            residual(name, members, "nominal_mass", step, arrays["mass"])
            if len(members) == 1:
                repeat = arrays["combined"]
                counts = dict(
                    zip(
                        ["0", "1", "2", "3-5", "6-10", "11+"],
                        map(
                            int,
                            np.histogram(
                                repeat, [-0.5, 0.5, 1.5, 2.5, 5.5, 10.5, np.inf]
                            )[0],
                        ),
                    )
                )
                summaries[str(step)][name] = dict(
                    episodes=prefix["databases"][name]["episodes"],
                    k_counts=prefix["databases"][name]["k_counts"],
                    repeat_counts=counts,
                    repeat_shares={key: value / n for key, value in counts.items()},
                    repeat_quantile_method="linear",
                    repeat_quantiles=dict(
                        zip(
                            ["min", "p10", "median", "p90", "max"],
                            map(float, np.quantile(repeat, [0, 0.1, 0.5, 0.9, 1])),
                        )
                    ),
                )
        for axis, bucket in BUCKETS[1:]:
            active = bucket in ACTIVE
            for source in [
                "train_db_equal_pool",
                "train_pooled_pool",
                "combined_presentations",
                "nominal_mass",
            ]:
                for target in ["v", "validation_db_equal_pool"]:
                    tables["comparison"].append(
                        dict(
                            step=step,
                            axis=axis,
                            bucket=bucket,
                            source_population=source,
                            validation_database=target,
                            source_share=float(active),
                            validation_share=float(active),
                            difference=0.0,
                            ratio=1.0 if active else "undefined",
                            source_db_coverage=2 if active else 0,
                            axis_tv=0.0,
                        )
                    )
    tables["distribution_summary"] = dict(
        prefixes=summaries,
        minimum_unique=200,
        weighted_quantiles="inverse_ecdf",
        unweighted_quantiles="linear",
    )
    return caches, replay, tables, train, validation, config


class VerifyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = fixture()

    def test_complete_fixture(self):
        result = verify(*self.inputs)
        self.assertTrue(result["checks"])
        self.assertEqual(result["prefixes"], [1000, 2000, 4000])

    def test_corruption_rejected(self):
        for table, field, value in [
            ("pool", "share", 0.2),
            ("exposure", "nominal_mass", -1),
            ("residuals", "r_ref_mean", 0.1),
            ("comparison", "axis_tv", 0.3),
        ]:
            args = copy.deepcopy(self.inputs)
            args[2][table][0][field] = value
            with self.subTest(table=table), self.assertRaises(ValueError):
                verify(*args)

    def test_missing_duplicate_nan_rejected(self):
        for mutation in ["missing", "duplicate", "nan"]:
            args = copy.deepcopy(self.inputs)
            if mutation == "missing":
                args[2]["pool"].pop()
            elif mutation == "duplicate":
                args[2]["pool"].append(args[2]["pool"][0])
            else:
                args[2]["pool"][0]["share"] = float("nan")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify(*args)

    def test_replay_corruption(self):
        for field in ["support", "query", "mass", "episodes", "k_counts", "digest"]:
            args = copy.deepcopy(self.inputs)
            state = args[1]["prefixes"][1000]["databases"]["a"]
            if field in ["support", "query", "mass"]:
                state[field][0] += 1
            elif field == "episodes":
                state[field] += 1
            elif field == "k_counts":
                state[field][1] += 1
            else:
                args[1]["prefixes"][1000]["digest"] = "bad"
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify(*args)


if __name__ == "__main__":
    unittest.main()
