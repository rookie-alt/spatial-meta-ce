import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts import s2_source_replay as core
from spatial_meta_ce.manifests import digest, episode_indices


class S2ReplayTests(unittest.TestCase):
    def fixture(self, directory, db="accidents", conflict=False):
        path = directory / "artifacts/cache/pretrain" / db
        path.mkdir(parents=True)
        rows = [
            {"id": f"{db}/pretrain/{i + 1}", "query_hash": h}
            for i, h in enumerate(["z", "a", "z", "b"])
        ]
        manifest = {
            "schema": 1,
            "transform": "price40-group-logfanout-v1",
            "rows": rows,
            "unique_queries": 3,
            "provenance": {"excluded": []},
        }
        manifest = {**manifest, "content_hash": digest(manifest)}
        (path / "manifest.json").write_text(json.dumps(manifest))
        cards = np.array(
            [
                [10.0, 11.0],
                [0.0, 0.0],
                [10.0 if not conflict else 12.0, 11.0],
                [100.0, 100.0],
            ]
        )
        np.save(path / "cards.npy", cards)
        np.save(
            path / "offsets.npy",
            np.array(
                [[0, 0, 0, 0], [1, 1, 2, 1], [2, 2, 5, 2], [3, 3, 7, 3], [4, 4, 11, 4]]
            ),
        )
        frozen = {
            "manifest_hash": manifest["content_hash"],
            "rows": 4,
            "file_sha256": {n: core.file_hash(path / n) for n in core.CACHE_FILES},
        }
        return path, frozen

    def test_cache_first_identity_and_no_features(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            path, frozen = self.fixture(root)
            data = core.read_cache(root, "accidents", frozen)
            self.assertEqual(data["indices"].tolist(), [0, 1, 3])
            self.assertEqual(data["query_hash"], ["z", "a", "b"])
            self.assertEqual(data["n_tables"].tolist(), [2, 3, 4])
            self.assertEqual(data["raw_rows"], 4)
            (path / "cards.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "bytes"):
                core.read_cache(root, "accidents", frozen)

    def test_conflicting_duplicates_stop_without_filter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            _, frozen = self.fixture(root, conflict=True)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                core.read_cache(root, "accidents", frozen)

    def test_target_and_symlink_refused_before_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            for db in ("imdb", "../accidents", "stats", "genome", "ergastf1"):
                with self.assertRaises(ValueError):
                    core.read_cache(root, db, {})
            (root / "link").symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Symlink"):
                core.safe(root, root / "link/file")
            with self.assertRaises(ValueError):
                core.safe(root, root / "../escape")

    def test_manifest_identity_shape_offsets_and_values(self):
        for defect in (
            "identity",
            "shape",
            "offsets",
            "nan",
            "negative",
            "hash",
            "empty",
        ):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as d:
                root = Path(d).resolve()
                path, frozen = self.fixture(root)
                m = json.loads((path / "manifest.json").read_text())
                if defect in ("identity", "hash", "empty"):
                    if defect == "identity":
                        m["rows"][0]["id"] = "imdb/pretrain/1"
                    elif defect == "hash":
                        m["content_hash"] = "wrong"
                    else:
                        m["rows"][0]["query_hash"] = ""
                    if defect != "hash":
                        m["content_hash"] = digest(
                            {k: v for k, v in m.items() if k != "content_hash"}
                        )
                        frozen = {**frozen, "manifest_hash": m["content_hash"]}
                    (path / "manifest.json").write_text(json.dumps(m))
                elif defect == "offsets":
                    np.save(
                        path / "offsets.npy",
                        np.array([[0] * 4, [1] * 4, [0] * 4, [3] * 4, [4] * 4]),
                    )
                else:
                    values = np.array(
                        [[10.0, 11.0], [0.0, 0.0], [10.0, 11.0], [100.0, 100.0]]
                    )
                    if defect == "shape":
                        values = values[:, :1]
                    else:
                        values[0, 0] = np.nan if defect == "nan" else -1
                    np.save(path / "cards.npy", values)
                frozen = {
                    **frozen,
                    "file_sha256": {
                        n: core.file_hash(path / n) for n in core.CACHE_FILES
                    },
                }
                with self.assertRaises(ValueError):
                    core.read_cache(root, "accidents", frozen)

    def caches(self):
        return {
            db: {
                "indices": np.arange(0, 60, 2),
                "query_hash": [f"h{i}" for i in range(30)],
                "true": np.ones(30),
                "pg": np.ones(30),
                "n_tables": np.ones(30, dtype=int),
                "raw_rows": 60,
            }
            for db in ("accidents", "airline")
        }

    def test_replay_matches_independent_original_calls_and_chain(self):
        caches = self.caches()
        config = {
            "seed": 42,
            "steps": 4,
            "meta_batch": 4,
            "query_size": 6,
            "support_sizes": [2, 5],
            "accumulation": 1,
        }
        result = core.replay(caches, list(caches), config, prefixes=(1, 2, 4))
        rng = np.random.default_rng(43)
        previous = hashlib.sha256(b"").hexdigest()
        presentations = 0
        for _ in range(16):
            db = str(rng.choice(list(caches)))
            k = int(rng.choice([2, 5]))
            si, qi = episode_indices(caches[db]["indices"], k, 6, rng)
            episode = {
                "db": db,
                "support": [caches[db]["query_hash"][int(i) // 2] for i in si],
                "query": [caches[db]["query_hash"][int(i) // 2] for i in qi],
            }
            previous = digest({"previous": previous, "episode": episode})
            presentations += k + 6
        self.assertEqual(result["digest"], previous)
        final = result["prefixes"][4]
        self.assertEqual(final["presentations"], presentations)
        self.assertEqual(final["episodes"], 16)
        self.assertAlmostEqual(
            sum(d["mass"].sum() for d in final["databases"].values()), 4, places=12
        )
        self.assertEqual(
            sum(d["query"].sum() for d in final["databases"].values()), 16 * 6
        )
        for db, data in final["databases"].items():
            self.assertEqual(data["episodes"], sum(data["k_counts"].values()))
            self.assertTrue(
                np.all(
                    data["support"] >= result["prefixes"][2]["databases"][db]["support"]
                )
            )
        self.assertNotEqual(
            core.replay(caches, list(reversed(caches)), config, prefixes=(4,))[
                "digest"
            ],
            previous,
        )
        self.assertTrue(np.all(caches["accidents"]["true"] == 1))

    def test_invalid_sampling_and_digest_block(self):
        c = {
            "seed": 42,
            "steps": 1,
            "meta_batch": 4,
            "query_size": 64,
            "support_sizes": [5],
            "accumulation": 1,
        }
        with self.assertRaises(ValueError):
            core.replay(self.caches(), list(self.caches()), c, prefixes=(1,))
        with self.assertRaises(core.ReplayMismatch):
            core.certify(
                {
                    "digest": "wrong",
                    "prefixes": {4000: {"episodes": 16000, "presentations": 1621915}},
                }
            )

    def test_frozen_contract_and_hashes(self):
        root = Path(__file__).resolve().parents[1]
        contract = core.load_contract(root)
        self.assertEqual(len(contract["train"]), 23)
        self.assertEqual(contract["validation"], ["chembl", "legalacts", "telstra"])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises((ValueError, FileNotFoundError)):
                core.load_contract(Path(d))

    def test_fresh_process_never_imports_torch_or_models(self):
        process = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import sys; from scripts import s2_source_replay; assert not any(x == 'torch' or x.startswith('spatial_meta_ce') for x in sys.modules)",
            ],
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(process.returncode, 0, process.stderr.decode())


if __name__ == "__main__":
    unittest.main()
