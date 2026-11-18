import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from spatial_meta_ce import price_meta127_native as native


def native_row(n_join, n_fanout, n_table, n_filter, offset=0.0):
    """Build one official-style feature row (flat groups, filter may be None)."""
    join = torch.arange(n_join * 40, dtype=torch.float32) + offset
    fanout = torch.arange(n_fanout * 40, dtype=torch.float32) + offset
    table = torch.arange(n_table * 4, dtype=torch.float32) + offset
    if n_filter == 0:
        filter_group = None
    else:
        filter_group = torch.arange(n_filter * 43, dtype=torch.float32) + offset
    return (join, fanout, table, filter_group)


def write_pickle(directory, rows, counts, truths=None, pgs=None):
    path = Path(directory) / "features40.pkl"
    data = {
        "data_features": rows,
        "true_cards": (
            list(truths)
            if truths is not None
            else [10.0 * (i + 1) for i in range(len(rows))]
        ),
        "pg_est_cards": (
            list(pgs) if pgs is not None else [5.0 * (i + 1) for i in range(len(rows))]
        ),
        "n_join_cols": [c[0] for c in counts],
        "n_fanouts": [c[1] for c in counts],
        "n_tables": [c[2] for c in counts],
        "n_filter_cols": [c[3] for c in counts],
    }
    with path.open("wb") as stream:
        pickle.dump(data, stream)
    return path


def literal_onecycle_lr(
    step,
    max_lr=1e-5,
    div_factor=2,
    final_div_factor=1e4,
    pct_start=0.3,
    total_steps=100,
):
    """Literal transcription of the official OneCycleLR schedule."""
    initial_lr = max_lr / div_factor
    min_lr = initial_lr / final_div_factor
    phase_one_end = pct_start * total_steps - 1
    if step <= phase_one_end:
        pct = step / phase_one_end
        start, end = initial_lr, max_lr
    else:
        pct = (step - phase_one_end) / ((total_steps - 1) - phase_one_end)
        start, end = max_lr, min_lr
    return end + (start - end) / 2.0 * (math.cos(math.pi * pct) + 1)


class TinyLinear(torch.nn.Module):
    """Minimal stand-in for the PRICE encoder (single linear map)."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x, pg=None, mask=None, *rest):
        return self.linear(x[:, :1])


class RecordingAdam(torch.optim.Adam):
    """Adam that snapshots gradients right before each parameter update."""

    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.snapshots = []

    def step(self, closure=None):
        self.snapshots.append(
            [
                None
                if group["params"][0].grad is None
                else group["params"][0].grad.detach().clone()
                for group in self.param_groups
            ]
        )
        return super().step(closure)


def tiny_dataset(n_rows):
    features = torch.arange(n_rows, dtype=torch.float32).reshape(n_rows, 1) / 10.0
    truth = torch.arange(1, n_rows + 1, dtype=torch.float32)
    pg = torch.arange(1, n_rows + 1, dtype=torch.float32) * 2.0
    mask = torch.ones((n_rows, 1), dtype=torch.int64)
    counts = torch.ones((n_rows, 4), dtype=torch.float32)
    return torch.utils.data.TensorDataset(features, truth, pg, mask, counts)


def two_row_dataset():
    rows = [
        native_row(2, 2, 2, 2, offset=0.0),
        native_row(1, 1, 1, 0, offset=100.0),
    ]
    counts = [(2, 2, 2, 2), (1, 1, 1, 0)]
    return rows, counts


class RawLoaderTests(unittest.TestCase):
    def test_load_raw_exposes_counts_rows_and_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(
                directory, rows, counts, truths=[12.0, 30.0], pgs=[3.0, 7.0]
            )
            dataset = native.load_raw(path)
        self.assertEqual(dataset.counts, (2, 2, 2, 2))
        self.assertEqual(dataset.rows, 2)
        self.assertEqual(dataset.truth.dtype, torch.float32)
        self.assertEqual(dataset.pg.dtype, torch.float32)
        self.assertEqual(tuple(dataset.truth.shape), (2,))
        self.assertEqual(tuple(dataset.pg.shape), (2,))
        self.assertEqual(dataset.truth.tolist(), [12.0, 30.0])
        self.assertEqual(dataset.pg.tolist(), [3.0, 7.0])
        self.assertEqual(dataset.pg.device.type, "cpu")

    def test_load_raw_rejects_misaligned_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            with self.assertRaises(ValueError):
                native.load_raw(path, expected_rows=3)

    def test_load_raw_rejects_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            data = pickle.loads(path.read_bytes())
            data["n_tables"] = [2, 2]
            broken = Path(directory) / "broken.pkl"
            broken.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                native.load_raw(broken)

    def test_load_raw_rejects_non_finite_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            data = pickle.loads(path.read_bytes())
            data["true_cards"] = [float("nan"), 1.0]
            broken = Path(directory) / "nan.pkl"
            broken.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                native.load_raw(broken)

    def test_load_raw_rejects_non_positive_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            data = pickle.loads(path.read_bytes())
            data["pg_est_cards"] = [0.0, 1.0]
            broken = Path(directory) / "zero.pkl"
            broken.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                native.load_raw(broken)

    def test_load_raw_rejects_short_feature_tuple(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            data = pickle.loads(path.read_bytes())
            data["data_features"] = [rows[0][:3]]
            broken = Path(directory) / "short.pkl"
            broken.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                native.load_raw(broken)

    def test_load_raw_rejects_wrong_group_width(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            data = pickle.loads(path.read_bytes())
            broken_join = native_row(1, 1, 1, 1)[0][:39]
            data["data_features"] = [
                (
                    broken_join,
                    native_row(1, 1, 1, 1)[1],
                    native_row(1, 1, 1, 1)[2],
                    native_row(1, 1, 1, 1)[3],
                ),
                rows[1],
            ]
            broken = Path(directory) / "width.pkl"
            broken.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                native.load_raw(broken)

    def test_load_raw_rejects_count_feature_disagreement(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, counts = two_row_dataset()
            path = write_pickle(directory, rows, counts)
            data = pickle.loads(path.read_bytes())
            data["n_join_cols"] = [1, 1]
            broken = Path(directory) / "disagree.pkl"
            broken.write_bytes(pickle.dumps(data))
            with self.assertRaises(ValueError):
                native.load_raw(broken)


class PaddingTests(unittest.TestCase):
    def padded_dataset(self, directory):
        rows, counts = two_row_dataset()
        path = write_pickle(
            directory, rows, counts, truths=[12.0, 30.0], pgs=[3.0, 7.0]
        )
        return native.load_raw(path)

    def test_padded_matches_official_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.padded_dataset(directory)
            padded = dataset.padded(dataset.counts)
        self.assertIsInstance(padded, torch.utils.data.TensorDataset)
        features, truth, pg, mask, counts = padded.tensors
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.int64)
        self.assertEqual(counts.dtype, torch.float32)
        self.assertEqual(tuple(counts.shape), (2, 4))
        self.assertEqual(tuple(features.shape), (2, 254))
        self.assertEqual(tuple(mask.shape), (2, 9))

        row = native_row(1, 1, 1, 0, 100.0)
        join_row = row[0].reshape(1, 40)
        fanout_row = row[1].reshape(1, 40)
        table_row = row[2].reshape(1, 4)
        join_block = features[1, 0:80]
        self.assertTrue(torch.equal(join_block[0:40], join_row[0]))
        self.assertTrue((join_block[40:80] == -1000.0).all())
        fanout_block = features[1, 80:160]
        self.assertTrue(torch.equal(fanout_block[0:40], fanout_row[0]))
        self.assertTrue((fanout_block[40:80] == -1000.0).all())
        table_block = features[1, 160:168]
        self.assertTrue(torch.equal(table_block[0:4], table_row[0]))
        self.assertTrue(
            torch.equal(
                table_block[4:8], torch.tensor([0.0, -1000.0, -1000.0, -1000.0])
            )
        )
        self.assertTrue((features[1, 168:254] == -1000.0).all())

        self.assertEqual(mask[0].tolist(), [1] * 9)
        self.assertEqual(mask[1].tolist(), [1, 1, 0, 1, 0, 1, 0, 0, 0])
        self.assertEqual(counts[0].tolist(), [2.0, 2.0, 2.0, 2.0])
        self.assertEqual(counts[1].tolist(), [1.0, 1.0, 1.0, 0.0])
        self.assertEqual(truth.tolist(), [12.0, 30.0])
        self.assertEqual(pg.tolist(), [3.0, 7.0])

    def test_padded_keeps_native_values_without_adapter_transforms(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.padded_dataset(directory)
            padded = dataset.padded(dataset.counts)
        features = padded.tensors[0]
        fanout = features[0, 80:120]
        expected = native_row(2, 2, 2, 2, 0.0)[1][:40]
        self.assertTrue(torch.equal(fanout, expected))
        self.assertFalse(torch.allclose(fanout, torch.log1p(fanout)))

    def test_padded_supports_none_filter_group(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.padded_dataset(directory)
            padded = dataset.padded(dataset.counts)
        self.assertEqual(tuple(padded.tensors[0].shape), (2, 254))
        self.assertTrue((padded.tensors[0][1, 168:254] == -1000.0).all())

    def test_padded_feeds_data_loader_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.padded_dataset(directory)
            padded = dataset.padded(dataset.counts)
            loader = native.make_loader(padded, batch_size=1)
            batches = list(loader)
        self.assertEqual(len(batches), 2)
        features, truth, pg, mask, counts = batches[0]
        self.assertEqual(tuple(features.shape), (1, 254))
        self.assertEqual(tuple(truth.shape), (1,))
        self.assertEqual(tuple(pg.shape), (1,))
        self.assertEqual(tuple(mask.shape), (1, 9))
        self.assertEqual(tuple(counts.shape), (1, 4))

    def test_padded_rejects_capacity_below_data(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.padded_dataset(directory)
            with self.assertRaises(ValueError):
                dataset.padded((1, 2, 2, 2))


class ForwardBatchTests(unittest.TestCase):
    def test_forward_batch_uses_official_log_transforms(self):
        captured = {}

        class Recorder(torch.nn.Module):
            def forward(self, x, pg=None, mask=None, *rest):
                captured["x"] = x
                captured["pg"] = pg
                captured["mask"] = mask
                captured["rest"] = rest
                return torch.zeros((x.shape[0], 1), dtype=torch.float32)

        features = torch.zeros((2, 4), dtype=torch.float32)
        truth = torch.tensor([9.0, 99.0], dtype=torch.float32)
        pg = torch.tensor([1.0, 4.0], dtype=torch.float32)
        mask = torch.ones((2, 3), dtype=torch.int64)
        counts = torch.tensor([[2.0, 1.0, 3.0, 0.0], [1.0, 2.0, 0.0, 4.0]])
        output, labels = native.forward_batch(
            Recorder(), (features, truth, pg, mask, counts), "cpu"
        )
        self.assertEqual(tuple(output.shape), (2,))
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertTrue(torch.allclose(labels, torch.log(truth + 1.0) + 1.0))
        self.assertTrue(
            torch.allclose(captured["pg"].reshape(-1), torch.log(pg + 1.0) + 1.0)
        )
        self.assertEqual(tuple(captured["pg"].shape), (2, 1))
        self.assertEqual(
            [tuple(part.shape) for part in captured["rest"]],
            [(2, 1), (2, 1), (2, 1), (2, 1)],
        )
        self.assertEqual(
            [float(part.reshape(-1)[0]) for part in captured["rest"]],
            [2.0, 1.0, 3.0, 0.0],
        )


class OptimizerTests(unittest.TestCase):
    def test_optimizer_matches_official_hyperparameters(self):
        model = torch.nn.Linear(1, 1)
        optimizer, scheduler = native.create_optimizer(model)
        group = optimizer.param_groups[0]
        # OneCycleLR sets the live lr to max_lr / div_factor right after
        # construction, exactly like the official finetune script.
        self.assertAlmostEqual(group["lr"], 5e-6)
        self.assertAlmostEqual(group["weight_decay"], 5e-5)
        self.assertEqual(scheduler.total_steps, 100)
        self.assertAlmostEqual(scheduler.base_lrs[0], 5e-6)
        self.assertEqual(len(scheduler._schedule_phases), 2)
        self.assertAlmostEqual(scheduler._schedule_phases[0]["end_step"], 29.0)
        self.assertAlmostEqual(scheduler._schedule_phases[1]["end_step"], 99.0)

    def test_epoch_scheduler_steps_match_literal_reference(self):
        model = torch.nn.Linear(1, 1)
        optimizer, scheduler = native.create_optimizer(model)
        for step in range(101):
            self.assertAlmostEqual(
                optimizer.param_groups[0]["lr"], literal_onecycle_lr(step), places=12
            )
            if step < 100:
                scheduler.step()


class TrainEpochTests(unittest.TestCase):
    def reference_gradients(self, model, dataset, batch_size):
        """Recompute the official macro-batch gradients without any update."""
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False
        )
        gradients = []
        for features, truth, pg, mask, counts in loader:
            labels = torch.log(truth + 1.0) + 1.0
            output = model(features[:, :1]).reshape(-1)
            loss = torch.mean((output - labels) ** 2)
            model.zero_grad(set_to_none=True)
            loss.backward()
            gradients.append(model.linear.weight.grad.detach().clone())
            model.zero_grad(set_to_none=True)
        return gradients

    def test_micro_accumulation_matches_full_macro_batch(self):
        torch.manual_seed(0)
        dataset = tiny_dataset(17)
        model = TinyLinear()
        reference = self.reference_gradients(model, dataset, 10)
        model.zero_grad(set_to_none=True)
        optimizer = RecordingAdam(model.parameters(), lr=0.0)
        result = native.train_epoch(
            model, dataset, optimizer, batch_size=10, microbatch=4, shuffle=False
        )
        self.assertEqual(result["updates"], 2)
        self.assertEqual(result["examples"], 17)
        self.assertEqual(len(result["step_seconds"]), 2)
        self.assertEqual(len(optimizer.snapshots), 2)
        for snapshot, expected in zip(optimizer.snapshots, reference):
            self.assertTrue(torch.allclose(snapshot[0], expected, atol=1e-6, rtol=1e-5))

    def test_partial_last_macro_batch_is_weighted_by_its_own_size(self):
        torch.manual_seed(0)
        dataset = tiny_dataset(5)
        model = TinyLinear()
        reference = self.reference_gradients(model, dataset, 3)
        model.zero_grad(set_to_none=True)
        optimizer = RecordingAdam(model.parameters(), lr=0.0)
        result = native.train_epoch(
            model, dataset, optimizer, batch_size=3, microbatch=2, shuffle=False
        )
        self.assertEqual(result["updates"], 2)
        self.assertEqual(result["examples"], 5)
        self.assertTrue(
            torch.allclose(
                optimizer.snapshots[1][0], reference[1], atol=1e-6, rtol=1e-5
            )
        )

    def test_macro_step_count_for_fifty_thousand_rows(self):
        dataset = tiny_dataset(50000)
        model = TinyLinear()
        optimizer = RecordingAdam(model.parameters(), lr=1e-3)
        observed = []
        original = native.forward_batch

        def recorder(model_, batch, device):
            observed.append(int(batch[0].shape[0]))
            return original(model_, batch, device)

        with mock.patch.object(native, "forward_batch", side_effect=recorder):
            result = native.train_epoch(model, dataset, optimizer, batch_size=15000)
        self.assertEqual(result["updates"], 4)
        self.assertEqual(result["examples"], 50000)
        self.assertEqual(len(result["step_seconds"]), 4)
        self.assertTrue(all(value >= 0 for value in result["step_seconds"]))
        self.assertEqual(sum(observed), 50000)
        # 15000 = 29 * 512 + 152, and the last 5000-row macro batch splits
        # into 9 * 512 + 392 microbatches.
        self.assertEqual(sorted(set(observed)), [152, 392, 512])
        self.assertEqual(len(observed), 100)
        self.assertEqual(observed[-1], 392)

    def test_train_epoch_rejects_non_finite_loss(self):
        class Exploding(TinyLinear):
            def forward(self, x, pg=None, mask=None, *rest):
                return torch.full((x.shape[0], 1), float("nan"))

        model = Exploding()
        optimizer = RecordingAdam(model.parameters(), lr=1e-3)
        with self.assertRaises(Exception):
            native.train_epoch(model, tiny_dataset(4), optimizer, batch_size=4)


class CheckpointTests(unittest.TestCase):
    def test_load_model_rejects_sha_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"weight": torch.zeros(1)}, path)
            with self.assertRaises(ValueError):
                native.load_model(path, "0" * 64, (1, 1, 1, 1), "cpu")

    def test_load_model_rejects_non_plain_state_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            torch.save({"epoch": 3, "optimizer": {}}, path)
            digest = native.file_sha256(path)
            with self.assertRaises(ValueError):
                native.load_model(path, digest, (1, 1, 1, 1), "cpu")

    def test_state_dict_prefix_stripping(self):
        state = {
            "module.scale_encoder.norm.weight": torch.zeros(1),
            "module.scale_encoder.norm.bias": torch.zeros(1),
        }
        stripped = native.strip_state_dict_prefixes(state)
        self.assertEqual(
            sorted(stripped), ["scale_encoder.norm.bias", "scale_encoder.norm.weight"]
        )
        price_state = {"price.scale_encoder.norm.weight": torch.zeros(1)}
        self.assertEqual(
            sorted(native.strip_state_dict_prefixes(price_state)),
            ["scale_encoder.norm.weight"],
        )

    def test_vendor_verification_matches_provenance(self):
        digests = native.verify_vendor()
        self.assertIn("price_baseline/encoder.py", digests)
        self.assertEqual(len(digests), 3)


if __name__ == "__main__":
    unittest.main()
