"""Independent synthetic evidence; no production loss or training imports."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.s2_source_replay import TRAIN
from scripts.s6_loss_verify import verify_loss_evidence


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def fixture(steps=2, mode="mean_tail10", predictions=None, formal=False):
    config = dict(
        steps=steps,
        meta_batch=4,
        accumulation=1,
        seed=42,
        query_size=8,
        support_sizes=[5],
        tail_risk_mode=mode,
    )
    if formal:
        config = {
            **config,
            "steps": 4000,
            "query_size": 64,
            "support_sizes": [5, 10, 20, 50, 100],
        }
    pool = 200 if formal else 30
    # All 23 sources; include a duplicate whose first index must be preserved.
    stores = {
        db: dict(
            query_hashes=[f"{db}-{i}" for i in range(pool)] + [f"{db}-0"],
            unique=np.arange(pool),
            tables=np.array([1, 4, 2, 5, 3] * (pool // 5) + [1]),
            labels=np.zeros(pool + 1, dtype=np.float64),
        )
        for db in TRAIN
    }
    rng = np.random.default_rng(43)
    previous = hashlib.sha256(b"").hexdigest()
    rows, snapshots = [], []
    for episode in range(1, config["steps"] * 4 + 1):
        db = str(rng.choice(TRAIN))
        k = int(rng.choice(config["support_sizes"]))
        chosen = rng.choice(
            stores[db]["unique"], k + config["query_size"], replace=False
        )
        hashes = [stores[db]["query_hashes"][i] for i in chosen]
        previous = digest(
            dict(
                previous=previous,
                episode=dict(db=db, support=hashes[:k], query=hashes[k:]),
            )
        )
        pred = (
            np.zeros(k + config["query_size"], dtype=np.float32)
            if formal
            else np.array(
                predictions
                if predictions is not None
                else [-3, 0, 1, 2, 3, 3, 0, -1, 0, 1, 0, 2, 0],
                dtype=np.float32,
            )
        )
        loss = np.where(
            np.abs(pred) < 1,
            np.float32(0.5) * pred * pred,
            np.abs(pred) - np.float32(0.5),
        ).astype(np.float32)
        n, m = len(loss), (len(loss) + 9) // 10
        tau = np.sort(loss)[-m]
        h, e = int(np.sum(loss > tau)), int(np.sum(loss == tau))
        a = np.where(
            loss > tau,
            np.float32(1 / m),
            np.where(loss == tau, np.float32((m - h) / (m * e)), np.float32(0)),
        ).astype(np.float32)
        c = (
            (np.float32(0.5 / n) + np.float32(0.5) * a)
            if mode == "mean_tail10"
            else np.full(n, 1 / n, dtype=np.float32)
        )
        cd = c.astype(np.float64)
        tables = stores[db]["tables"][chosen]
        row = dict(
            step=(episode - 1) // 4 + 1,
            episode_index=episode,
            database=db,
            k=k,
            n=n,
            m=m,
            tau=float(tau),
            h=h,
            e=e,
            mode=mode,
            mean_loss=float(np.mean(loss)),
            tail_mean=float(np.sum(a * loss)),
            mixed_loss=float(np.sum(c * loss))
            if mode == "mean_tail10"
            else float(np.mean(loss)),
            coefficient_sum=float(cd.sum()),
            coefficient_square_sum=float((cd * cd).sum()),
            low_mass=float(cd[tables < 4].sum()),
            high_mass=float(cd[tables >= 4].sum()),
            under_mass=float(cd[pred < 0].sum()),
            over_mass=float(cd[pred > 0].sum()),
            exact_mass=float(cd[pred == 0].sum()),
            sample_plan_digest=previous,
        )
        samples = [
            dict(
                query_hash=hashes[j],
                role="support" if j < k else "query",
                log_pred=float(pred[j]),
                label=0.0,
                loss=float(loss[j]),
                coefficient=float(c[j]),
                n_tables=int(tables[j]),
            )
            for j in range(n)
        ]
        rows.append(row)
        if not formal or row["step"] in (1, 1000, 2000, 4000):
            snapshots.append(dict(audit=copy.deepcopy(row), samples=samples))
    return config, stores, rows, snapshots


class LossEvidenceTests(unittest.TestCase):
    def verify(self, data):
        config, stores, rows, snapshots = data
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audit = "".join(json.dumps(row) + "\n" for row in rows).encode()
            snap = "".join(json.dumps(row) + "\n" for row in snapshots).encode()
            (root / "loss-audit.jsonl").write_bytes(audit)
            (root / "loss-snapshots.jsonl").write_bytes(snap)
            result = verify_loss_evidence(root, config, stores)
            self.assertEqual(result["audit_sha256"], hashlib.sha256(audit).hexdigest())
            self.assertEqual(
                result["snapshots_sha256"], hashlib.sha256(snap).hexdigest()
            )
            return result

    def rejects(self, data):
        with self.assertRaises(ValueError):
            self.verify(data)

    def test_smoke_and_mass_conservation(self):
        for mode in ("mean", "mean_tail10"):
            result = self.verify(fixture(mode=mode))
            self.assertTrue(result["passed"])
            self.assertEqual(
                (result["episodes"], result["presentations"], result["snapshot_rows"]),
                (8, 104, 8),
            )
            self.assertEqual(result["mode"], mode)
            self.assertAlmostEqual(
                sum(v["coefficient_sum"] for v in result["per_database"].values()),
                2,
                places=6,
            )
            self.assertIn("16000", result["verification_scope"])
            self.assertNotIn("execution_verified", result)

    def test_small_configs_not_bound_to_formal_budget(self):
        for steps in (3, 5):
            self.assertEqual(self.verify(fixture(steps))["episodes"], steps * 4)

    def test_actual_fp32_zero_ties_and_threshold_neighbors(self):
        for predictions in (
            [0.0] * 13,
            [1e-30] * 13,
            [1.0, np.nextafter(np.float32(1), np.float32(2)), 1.0] + [0.0] * 10,
        ):
            self.assertTrue(self.verify(fixture(predictions=predictions))["passed"])

    def test_audit_and_snapshot_order_coverage(self):
        for which in (2, 3):
            for operation in ("reverse", "missing", "duplicate", "extra"):
                data = list(fixture())
                if operation == "reverse":
                    data[which].reverse()
                elif operation == "missing":
                    data[which].pop()
                elif operation == "duplicate":
                    data[which][1] = copy.deepcopy(data[which][0])
                else:
                    data[which].append(copy.deepcopy(data[which][-1]))
                self.rejects(data)

    def test_snapshot_identity_roles_and_missing_support(self):
        for operation in ("hash", "role", "missing", "reverse", "table", "identity"):
            data = list(fixture())
            samples = data[3][0]["samples"]
            if operation == "hash":
                samples[0]["query_hash"] = "wrong"
            elif operation == "role":
                samples[0]["role"] = "query"
            elif operation == "missing":
                del samples[:5]
            elif operation == "reverse":
                samples.reverse()
            elif operation == "table":
                samples[0]["n_tables"] += 1
            else:
                del samples[0]["query_hash"]
            self.rejects(data)

    def test_cache_label_and_unique_drift(self):
        for change in (
            "label",
            "duplicate_label",
            "unique",
            "missing_source",
            "target",
            "dtype",
        ):
            data = list(fixture())
            db = data[2][0]["database"]
            store = data[1][db]
            if change == "label":
                store["labels"] = np.ones(31, dtype=np.float64)
            elif change == "duplicate_label":
                store["labels"][-1] = 1
            elif change == "unique":
                store["unique"] = np.arange(1, 31)
            elif change == "missing_source":
                del data[1][TRAIN[0]]
            elif change == "target":
                data[1]["imdb"] = store
            else:
                store["labels"] = store["labels"].astype(np.float32)
            self.rejects(data)

    def test_parent_passed_and_unknown_schema_not_trusted(self):
        for which in (2, 3):
            data = list(fixture())
            data[which][0]["passed"] = True
            self.rejects(data)

    def test_exact_replay_and_audit_invariants(self):
        changes = dict(
            step=2,
            episode_index=2,
            database=TRAIN[0],
            k=10,
            n=8,
            m=3,
            h=2,
            e=0,
            tau=-1.0,
            mode="mean",
            sample_plan_digest="0" * 64,
            coefficient_sum=0.9,
            coefficient_square_sum=0.5,
            low_mass=0.99,
            under_mass=0.99,
            mean_loss=-1.0,
            tail_mean=0.0,
            mixed_loss=99.0,
        )
        for field, value in changes.items():
            data = list(fixture())
            data[2][0][field] = value
            data[3][0]["audit"][field] = value
            self.rejects(data)

    def test_snapshot_numeric_tampering(self):
        for field, value in dict(
            log_pred=100.0, label=1.0, loss=100.0, coefficient=0.4
        ).items():
            data = list(fixture())
            data[3][0]["samples"][0][field] = value
            self.rejects(data)
        for field in ("log_pred", "label", "loss", "coefficient"):
            data = list(fixture())
            data[3][0]["samples"][0][field] = 0.12345678912345
            self.rejects(data)

    def test_formal_cannot_claim_historical_digest_from_small_pool(self):
        data = list(fixture())
        data[0] = {
            **data[0],
            "steps": 4000,
            "query_size": 64,
            "support_sizes": [5, 10, 20, 50, 100],
        }
        self.rejects(data)

    def test_complete_synthetic_formal_replay_rejects_historical_claim(self):
        # 16000 internally valid rows + 16 fixed snapshots, never fake the digest.
        data = fixture(formal=True)
        self.assertEqual((len(data[2]), len(data[3])), (16000, 16))
        with self.assertRaisesRegex(
            ValueError, "Historical formal sampling budget/digest"
        ):
            self.verify(data)

    def test_recorded_reduction_must_be_actual_fp32(self):
        for field in ("mean_loss", "tail_mean", "mixed_loss"):
            data = list(fixture())
            # Within numeric comparison tolerance, but not an FP32 reduction.
            value = data[2][0][field] + 1e-10
            data[2][0][field] = value
            data[3][0]["audit"][field] = value
            self.rejects(data)

    def test_snapshot_audit_boolean_alias_cannot_match_integer(self):
        data = list(fixture())
        data[3][0]["audit"]["step"] = True
        self.rejects(data)

    def test_invalid_config_types_and_nonfinite_evidence(self):
        for field, value in dict(
            seed=43,
            accumulation=2,
            meta_batch=1,
            steps=True,
            query_size=0,
            support_sizes=[5, 5],
            tail_risk_mode="bogus",
        ).items():
            data = list(fixture())
            data[0][field] = value
            self.rejects(data)
        for value in (float("nan"), float("inf"), True):
            data = list(fixture())
            data[2][0]["tau"] = value
            data[3][0]["audit"]["tau"] = value
            self.rejects(data)


if __name__ == "__main__":
    unittest.main()
