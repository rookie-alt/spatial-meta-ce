"""Guards for the experiment-126 target evaluation plan.

Synthetic fixtures only: no real PRICE cache is read and no target file is
touched.  The fake stores carry the arrays that the real canonicalisation
reads, so the canonical path, the support pool and the frozen denominators
are exercised rather than mocked away.
"""

import copy
import hashlib
import importlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


def _module():
    return importlib.import_module("spatial_meta_ce.price_meta126_plans")


def _hash(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _rows(db, split, unique, raw):
    """``unique`` distinct hashes, the remaining raw rows repeat the first."""
    rows = []
    for index in range(unique):
        rows.append(
            {
                "id": f"{db}/{split}/{index + 1}",
                "query_hash": _hash(f"{db}/{split}/{index}"),
                "source_line": index + 1,
            }
        )
    for index in range(raw - unique):
        source = rows[index]
        rows.append(
            {
                "id": f"{db}/{split}/{unique + index + 1}",
                "query_hash": source["query_hash"],
                "source_line": unique + index + 1,
            }
        )
    return rows


class FakeStore:
    """Synthetic PRICE cache stand-in with the arrays canonicalisation reads."""

    def __init__(self, db, split, unique, raw, max_counts, shared=()):
        self.path = Path(db)
        self.split = split
        self.rows = _rows(db, split, unique, raw)
        for index, value in enumerate(shared):
            self.rows[index]["query_hash"] = value
        count = len(self.rows)
        self.cards = np.zeros((count, 2))
        self.offsets = np.zeros((count + 1, 4), dtype=np.int64)
        self.groups = [np.zeros((0, 1)) for _ in range(4)]
        self.manifest = {
            "content_hash": _hash(f"{db}/{split}/content"),
            "inspection": {"max_counts": list(max_counts)},
            "provenance": {"source": {"database": db, "split": split}},
            "rows": self.rows,
        }


class FakeStores:
    """Synthetic pool with the same public surface as TargetStores."""

    def __init__(
        self,
        cache_root,
        max_open=4,
        finetune_dbs=("stats", "ergastf1", "genome"),
        sizes=None,
        maxima=None,
        shared_hashes=(),
    ):
        module = _module()
        self.root = Path(cache_root)
        self.databases = module.TARGET_DATABASES
        self.finetune_dbs = set(finetune_dbs)
        self.opened = []
        self.values = {}
        # Real frozen denominators, so the shipped constants -- not a
        # fixture override -- are what the plan is checked against.
        self.sizes = sizes or {
            db: (module.EXPECTED_UNIQUE[db], module.EXPECTED_ROWS[db])
            for db in module.TARGET_DATABASES
        }
        self.support_sizes = {
            db: (module.EXPECTED_UNIQUE[db] + 100, module.EXPECTED_ROWS[db] + 100)
            for db in module.SUPPORT_DATABASES
        }
        self.maxima = maxima or {
            "imdb": (1, 1, 1, 1),
            "stats": (12, 14, 8, 10),
            "ergastf1": (1, 1, 1, 1),
            "genome": (1, 1, 1, 1),
        }
        self.shared_hashes = shared_hashes

    def path_for(self, split, db):
        return self.root / split / db

    def finetune_available(self, db):
        return db in self.finetune_dbs

    def get(self, split, db):
        module = _module()
        if split not in module.ALLOWED_SPLITS or db not in module.TARGET_DATABASES:
            raise ValueError("Database or split is outside the target whitelist")
        if split == "finetune" and db == "imdb":
            raise ValueError("IMDB has no independent finetune cache")
        key = (split, db)
        if key not in self.values:
            if split == "finetune":
                unique, raw = self.support_sizes[db]
            else:
                unique, raw = self.sizes[db]
            self.values[key] = FakeStore(
                db,
                split,
                unique,
                raw,
                self.maxima[db],
                self.shared_hashes if split == "finetune" else (),
            )
        if key not in self.opened:
            self.opened.append(key)
        return self.values[key]

    def close(self):
        self.values.clear()


class Target126PlanGuards(unittest.TestCase):
    def module(self):
        return _module()

    def plan(self, stores=None):
        module = self.module()
        with (
            mock.patch.object(module, "TargetStores", stores or FakeStores),
            mock.patch.object(module, "_store_data_sha256", return_value="b" * 64),
        ):
            return module.build_plan("/unused")

    def test_frozen_target_order_and_populations(self):
        module = self.module()
        self.assertEqual(
            module.TARGET_DATABASES, ("imdb", "stats", "ergastf1", "genome")
        )
        self.assertEqual(
            module.EXPECTED_UNIQUE,
            {"imdb": 521, "stats": 2603, "ergastf1": 1258, "genome": 1699},
        )
        self.assertEqual(
            module.EXPECTED_ROWS,
            {"imdb": 766, "stats": 2749, "ergastf1": 1422, "genome": 2302},
        )
        self.assertEqual(module.TRAIN_LAYOUT, (12, 12, 7, 10))
        self.assertEqual(
            module.EXPECTED_MAXIMA,
            {
                "imdb": (12, 12, 7, 10),
                "stats": (12, 14, 8, 10),
                "ergastf1": (12, 12, 7, 10),
                "genome": (12, 12, 7, 10),
            },
        )

    def test_plan_kind_schema_and_exposure(self):
        module = self.module()
        plan = self.plan()
        self.assertEqual(plan["kind"], "price_meta126_target_plan")
        self.assertEqual(plan["schema"], 1)
        self.assertEqual(tuple(plan["databases"]), module.TARGET_DATABASES)
        self.assertEqual(
            list(plan["target_databases_read"]), list(module.TARGET_DATABASES)
        )
        self.assertNotIn("finetune/imdb", plan["caches_read"])

    def test_test_population_is_complete_and_unique(self):
        module = self.module()
        plan = self.plan()
        for db in module.TARGET_DATABASES:
            entry = plan["databases"][db]["test"]
            self.assertEqual(len(entry["indices"]), module.EXPECTED_UNIQUE[db])
            self.assertEqual(len(entry["ids"]), len(entry["indices"]))
            self.assertEqual(len(entry["hashes"]), len(entry["indices"]))
            self.assertEqual(len(set(entry["hashes"])), len(entry["hashes"]))
            self.assertEqual(len(set(entry["indices"])), len(entry["indices"]))
            self.assertEqual(entry["rows"], module.EXPECTED_ROWS[db])

    def test_imdb_support_is_absent_with_a_stated_reason(self):
        plan = self.plan()
        support = plan["databases"]["imdb"]["support"]
        self.assertFalse(support["available"])
        self.assertEqual(support["reason"], "missing_independent_finetune_cache")
        self.assertEqual(support["blocks"], [])
        self.assertFalse(support["cache_present"])
        self.assertTrue(support["cache_path"].endswith("finetune/imdb"))

    def test_support_blocks_shape_and_disjointness(self):
        plan = self.plan()
        for db in ("stats", "ergastf1", "genome"):
            support = plan["databases"][db]["support"]
            self.assertTrue(support["available"])
            self.assertIsNone(support["reason"])
            self.assertEqual(len(support["blocks"]), 5)
            flat = []
            for block in support["blocks"]:
                self.assertEqual(len(block["indices"]), 20)
                self.assertEqual(len(block["ids"]), 20)
                self.assertEqual(len(block["hashes"]), 20)
                flat.extend(block["hashes"])
            self.assertEqual(len(set(flat)), 100)

    def test_support_never_shares_a_query_hash_with_test(self):
        plan = self.plan()
        for db in ("stats", "ergastf1", "genome"):
            test_hashes = set(plan["databases"][db]["test"]["hashes"])
            for block in plan["databases"][db]["support"]["blocks"]:
                self.assertFalse(test_hashes & set(block["hashes"]))

    def test_layout_is_the_elementwise_maximum(self):
        plan = self.plan()
        self.assertEqual(plan["databases"]["stats"]["max_counts"], [12, 14, 8, 10])
        for db in ("imdb", "ergastf1", "genome"):
            self.assertEqual(plan["databases"][db]["max_counts"], [12, 12, 7, 10])

    def test_plan_is_deterministic_and_uses_the_frozen_rng(self):
        module = self.module()
        first = self.plan()
        second = self.plan()
        self.assertEqual(first, second)
        self.assertEqual(first["plan_sha256"], second["plan_sha256"])
        stores = FakeStores("/unused")
        plan = self.plan()
        for db_index, db in enumerate(module.TARGET_DATABASES):
            if db == "imdb":
                continue
            # Protocol: one representative per query hash, ordered by
            # (query_hash, id, source_line); support keeps the representatives
            # whose hash never occurs in test.
            finetune_rows = stores.get("finetune", db).rows
            test_hashes = {row["query_hash"] for row in stores.get("test", db).rows}
            groups = {}
            for index, row in enumerate(finetune_rows):
                groups.setdefault(row["query_hash"], []).append(index)
            representatives = []
            for query_hash in sorted(groups):
                members = sorted(
                    groups[query_hash],
                    key=lambda i: (
                        finetune_rows[i]["id"],
                        finetune_rows[i]["source_line"],
                    ),
                )
                if query_hash not in test_hashes:
                    representatives.append(members[0])
            expected = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([20261118, 126, db_index]))
            )
            pool = np.asarray(representatives, dtype=np.int64)
            chosen = expected.choice(pool, size=100, replace=False)
            blocks = plan["databases"][db]["support"]["blocks"]
            observed = [i for block in blocks for i in block["indices"]]
            self.assertEqual(observed, [int(i) for i in chosen])

    def test_unauthorized_database_and_split_are_refused_before_any_read(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            stores = module.TargetStores(directory)
            for db in ("accidents", "chembl", "sakila"):
                with self.assertRaises(ValueError):
                    stores.get("test", db)
            for split in ("pretrain", "train", "validation"):
                with self.assertRaises(ValueError):
                    stores.get(split, "stats")
            self.assertEqual(stores.opened, [])
            self.assertFalse((Path(directory) / "test").exists())
            stores.close()

    def test_imdb_finetune_is_refused_even_if_the_cache_appears(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finetune" / "imdb"
            path.mkdir(parents=True)
            (path / "manifest.json").write_text("{}")
            stores = module.TargetStores(directory)
            with self.assertRaises(ValueError):
                stores.get("finetune", "imdb")
            self.assertEqual(stores.opened, [])
            stores.close()

    def test_an_imdb_finetune_cache_stops_the_plan_instead_of_adding_a_shot(self):
        class WithImdb(FakeStores):
            def __init__(self, cache_root, max_open=4):
                super().__init__(
                    cache_root,
                    max_open,
                    finetune_dbs=("imdb", "stats", "ergastf1", "genome"),
                )

        with self.assertRaises(ValueError):
            self.plan(WithImdb)

    def test_provenance_source_split_prefix_and_exclusions(self):
        module = self.module()
        store = mock.Mock()
        store.manifest = {
            "provenance": {"source": {"database": "stats", "split": "test"}},
            "rows": [{"id": "stats/test/1"}],
        }
        module._check_provenance(store, "stats", "test")
        store.manifest["provenance"]["source"]["database"] = "imdb"
        with self.assertRaises(ValueError):
            module._check_provenance(store, "stats", "test")
        store.manifest["provenance"]["source"]["database"] = "stats"
        store.manifest["provenance"]["source"]["split"] = "finetune"
        with self.assertRaises(ValueError):
            module._check_provenance(store, "stats", "test")
        store.manifest["provenance"]["source"]["split"] = "test"
        store.manifest["rows"] = [{"id": "imdb/test/1"}]
        with self.assertRaises(ValueError):
            module._check_provenance(store, "stats", "test")
        store.manifest["rows"] = [{"id": "stats/test/1"}]
        store.manifest["provenance"]["excluded"] = [{"id": "stats/test/7"}]
        with self.assertRaises(ValueError):
            module._check_provenance(store, "stats", "test")

    def test_conflicting_duplicate_query_hash_is_rejected(self):
        module = self.module()
        store = mock.Mock()
        store.path = Path("stats")
        store.rows = [
            {"id": "stats/test/1", "query_hash": "a" * 64, "source_line": 1},
            {"id": "stats/test/2", "query_hash": "a" * 64, "source_line": 2},
        ]
        store.cards = np.array([[1.0, 1.0], [2.0, 2.0]])
        store.offsets = np.zeros((3, 4), dtype=np.int64)
        store.groups = [np.zeros((0, 1)) for _ in range(4)]
        with self.assertRaises(ValueError):
            module._canonical(store)

    def test_tampering_is_rejected_even_with_a_recomputed_digest(self):
        module = self.module()
        plan = self.plan()
        cases = []
        changed = copy.deepcopy(plan)
        changed["databases"]["stats"]["test"]["hashes"][0] = "e" * 64
        cases.append(changed)
        changed = copy.deepcopy(plan)
        changed["databases"]["stats"]["test"]["unique_queries"] -= 1
        cases.append(changed)
        changed = copy.deepcopy(plan)
        changed["databases"]["stats"]["test"]["rows"] += 5
        cases.append(changed)
        changed = copy.deepcopy(plan)
        changed["databases"]["stats"]["support"]["blocks"][0]["indices"][0] = 999
        cases.append(changed)
        changed = copy.deepcopy(plan)
        changed["databases"]["stats"]["support"]["available"] = False
        cases.append(changed)
        changed = copy.deepcopy(plan)
        changed["target_databases_read"] = ["imdb"]
        cases.append(changed)
        for candidate in cases:
            candidate["plan_sha256"] = module.plan_digest(candidate)
            with self.assertRaises(ValueError):
                module.validate_plan(candidate)
        stale = copy.deepcopy(plan)
        stale["databases"]["stats"]["test"]["rows"] += 5
        with self.assertRaises(ValueError):
            module.validate_plan(stale)

    def test_verify_plan_detects_cache_drift(self):
        module = self.module()
        plan = self.plan()

        class Drifted(FakeStores):
            def __init__(self, root, mode, **kwargs):
                super().__init__(root, **kwargs)
                self.mode = mode

            def get(self, split, db):
                store = super().get(split, db)
                if self.mode == "content" and split == "test" and db == "stats":
                    store.manifest["content_hash"] = "d" * 64
                elif self.mode == "identity" and split == "test" and db == "stats":
                    store.rows[0]["query_hash"] = "e" * 64
                elif self.mode == "support" and split == "finetune" and db == "stats":
                    store.rows[0]["query_hash"] = "e" * 64
                return store

        for mode in ("content", "identity", "support"):
            with mock.patch.object(
                module,
                "TargetStores",
                lambda root, _mode=mode: Drifted(root, _mode),
            ):
                with mock.patch.object(
                    module, "_store_data_sha256", return_value="b" * 64
                ):
                    with self.assertRaises(ValueError):
                        module.verify_plan(plan, "/unused")

    def test_verify_plan_accepts_an_unchanged_cache(self):
        module = self.module()
        with (
            mock.patch.object(module, "TargetStores", FakeStores),
            mock.patch.object(module, "_store_data_sha256", return_value="b" * 64),
        ):
            plan = module.build_plan("/unused")
            self.assertTrue(module.verify_plan(plan, "/unused"))

    def test_the_real_frozen_denominators_are_enforced(self):
        """The shipped constants, not a fixture, must stop a wrong population."""
        module = self.module()
        self.assertEqual(
            module.EXPECTED_UNIQUE,
            {"imdb": 521, "stats": 2603, "ergastf1": 1258, "genome": 1699},
        )
        self.assertEqual(
            module.EXPECTED_ROWS,
            {"imdb": 766, "stats": 2749, "ergastf1": 1422, "genome": 2302},
        )
        self.assertEqual(module.SUPPORT_DATABASES, ("stats", "ergastf1", "genome"))
        self.assertEqual(module.ALLOWED_SPLITS, ("test", "finetune"))

        class WrongDenominator(FakeStores):
            def __init__(self, cache_root, max_open=4):
                super().__init__(cache_root, max_open)
                self.sizes = {
                    "imdb": (520, 766),
                    "stats": (2603, 2749),
                    "ergastf1": (1258, 1422),
                    "genome": (1699, 2302),
                }

        with self.assertRaises(ValueError):
            self.plan(WrongDenominator)

    def test_the_real_layouts_are_enforced(self):
        module = self.module()
        self.assertEqual(
            module.EXPECTED_MAXIMA,
            {
                "imdb": (12, 12, 7, 10),
                "stats": (12, 14, 8, 10),
                "ergastf1": (12, 12, 7, 10),
                "genome": (12, 12, 7, 10),
            },
        )

        class WrongLayout(FakeStores):
            def __init__(self, cache_root, max_open=4):
                super().__init__(cache_root, max_open)
                self.maxima = {
                    "imdb": (1, 1, 1, 1),
                    "stats": (12, 20, 8, 10),
                    "ergastf1": (1, 1, 1, 1),
                    "genome": (1, 1, 1, 1),
                }

        with self.assertRaises(ValueError):
            self.plan(WrongLayout)

    def test_a_wrong_unique_denominator_is_refused(self):
        """The cache must match the frozen unique-query denominator exactly."""
        module = self.module()

        class WrongUnique(FakeStores):
            def __init__(self, cache_root, max_open=4):
                super().__init__(cache_root, max_open)
                self.sizes = {
                    db: (module.EXPECTED_UNIQUE[db] - 1, module.EXPECTED_ROWS[db])
                    for db in module.TARGET_DATABASES
                }

        with self.assertRaisesRegex(ValueError, "denominator"):
            self.plan(WrongUnique)

    def test_a_cache_that_drifts_from_the_frozen_constants_is_refused(self):
        """Guard against the frozen checks being weakened, not just wrong data.

        A cache whose population no longer matches the shipped constants must
        be refused even when the numbers differ by one row, because that is
        exactly the state a truncated or re-sampled cache would produce.
        """
        module = self.module()
        for delta in (1, -1):
            sizes = {
                db: (
                    module.EXPECTED_UNIQUE[db] + delta,
                    module.EXPECTED_ROWS[db] + delta,
                )
                for db in module.TARGET_DATABASES
            }

            class Drifted(FakeStores):
                def __init__(self, cache_root, max_open=4):
                    super().__init__(cache_root, max_open, sizes=sizes)

            with self.assertRaises(ValueError):
                self.plan(Drifted)

    def test_a_wrong_row_count_is_refused(self):
        module = self.module()

        class WrongRows(FakeStores):
            def __init__(self, cache_root, max_open=4):
                super().__init__(cache_root, max_open)
                self.sizes = {
                    db: (module.EXPECTED_UNIQUE[db], module.EXPECTED_ROWS[db] - 1)
                    for db in module.TARGET_DATABASES
                }

        with self.assertRaisesRegex(ValueError, "row count"):
            self.plan(WrongRows)

    def test_support_rows_sharing_a_test_hash_are_removed_from_the_pool(self):
        """A finetune query that also occurs in test may not become support."""
        stores = FakeStores("/unused")
        test_hash = stores.get("test", "stats").rows[0]["query_hash"]

        class SharedHash(FakeStores):
            def __init__(self, cache_root, max_open=4):
                super().__init__(cache_root, max_open, shared_hashes=(test_hash,))

        plan = self.plan(SharedHash)
        support = plan["databases"]["stats"]["support"]
        self.assertEqual(support["metadata"]["overlap_removed"], 1)
        test_hashes = set(plan["databases"]["stats"]["test"]["hashes"])
        for block in support["blocks"]:
            self.assertFalse(test_hashes & set(block["hashes"]))


if __name__ == "__main__":
    unittest.main()
