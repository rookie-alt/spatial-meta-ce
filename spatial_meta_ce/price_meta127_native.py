"""Official PRICE native runtime: raw loader, padding, full finetune core.

This module reproduces the official PRICE finetuning path
(`PRICE-master/finetune.py` plus `utils/model/{dataset,padding}.py`) without
the `PriceAdapter` feature transform and without any validation or target
database selection.  Group values stay native: fanout features are *not*
log-transformed here, padding uses the official `-1000` fill value, and the
official `log(card + 1) + 1` transforms are applied only where the official
script applies them (PG input and regression label).
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

GROUP_DIMS = (40, 40, 4, 43)
COUNT_KEYS = ("n_join_cols", "n_fanouts", "n_tables", "n_filter_cols")
RAW_KEYS = ("data_features", "true_cards", "pg_est_cards", *COUNT_KEYS)
PADDING_VALUE = -1000.0
VENDOR_DIR = Path(__file__).resolve().parent.parent / "vendor"
VENDOR_PROVENANCE = VENDOR_DIR / "PROVENANCE.json"
VENDOR_FILES = (
    "price_baseline/encoder.py",
    "price_baseline/module.py",
    "price_features/features_tool.py",
)

MACRO_BATCH_SIZE = 15000
MICRO_BATCH_SIZE = 512
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-5
TRAINING_EPOCHS = 100
SCHEDULER_STEPS_PER_EPOCH = 10
SCHEDULER_EPOCHS = 10
SCHEDULER_PCT_START = 0.3
SCHEDULER_DIV_FACTOR = 2
SCHEDULER_ANNEAL = "cos"


def file_sha256(path):
    """Return the SHA256 of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_vendor():
    """Verify the vendored PRICE sources against `vendor/PROVENANCE.json`."""
    provenance = json.loads(VENDOR_PROVENANCE.read_text())
    files = provenance.get("files", {})
    missing = [name for name in VENDOR_FILES if name not in files]
    if missing:
        raise ValueError(f"Vendor provenance is missing files: {missing}")
    digests = {}
    for name in VENDOR_FILES:
        path = VENDOR_DIR / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Vendor source is missing or linked: {path}")
        actual = file_sha256(path)
        expected = files[name].get("adapted_sha256")
        if actual != expected:
            raise ValueError(f"Vendor source changed: {name}")
        digests[name] = actual
    return digests


def strip_state_dict_prefixes(state):
    """Strip the `module.` / `price.` prefixes a wrapped checkpoint may carry."""
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint must be a non-empty plain state_dict")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError(
            "Official PRICE requires a plain state_dict, not a training checkpoint"
        )
    keys = list(state)
    if all(key.startswith("module.") for key in keys):
        return {key[len("module.") :]: value for key, value in state.items()}
    if all(key.startswith("price.") for key in keys):
        return {key[len("price.") :]: value for key, value in state.items()}
    return dict(state)


@dataclass(frozen=True)
class RawDataset:
    """Raw official PRICE features plus aligned labels and PG estimates."""

    rows: int
    counts: tuple
    row_counts: np.ndarray
    truth: torch.Tensor
    pg: torch.Tensor
    features: tuple

    def padded(self, counts):
        """Pad the raw features exactly like the official `features_padding`."""
        return pad_features(self, counts)


def _as_float32(value):
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def load_raw(path, expected_rows=None):
    """Load an official `features40.pkl` payload without any feature transform."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Raw feature file is missing: {path}")
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Raw feature payload must be a mapping")
    missing = [key for key in RAW_KEYS if key not in data]
    if missing:
        raise ValueError(f"Raw feature payload is missing keys: {missing}")
    lengths = {key: len(data[key]) for key in RAW_KEYS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Raw feature fields are not aligned: {lengths}")
    rows = lengths["true_cards"]
    if rows == 0:
        raise ValueError("Raw feature payload has no rows")
    if expected_rows is not None and int(expected_rows) != rows:
        raise ValueError(
            f"Raw feature rows {rows} do not match the expected {expected_rows}"
        )

    counts = tuple(
        int(max(int(data[key][index]) for index in range(rows))) for key in COUNT_KEYS
    )
    per_row = np.asarray(
        [[int(data[key][index]) for key in COUNT_KEYS] for index in range(rows)],
        dtype=np.int64,
    )
    if (per_row < 0).any():
        raise ValueError("Raw feature counts must be nonnegative")

    truth = _as_float32(data["true_cards"])
    pg = _as_float32(data["pg_est_cards"])
    for name, values in (("true_cards", truth), ("pg_est_cards", pg)):
        if not torch.isfinite(values).all():
            raise ValueError(f"Raw feature {name} contains non-finite values")
        if (values <= 0).any():
            raise ValueError(f"Raw feature {name} must be strictly positive")

    features = []
    for index, parts in enumerate(data["data_features"]):
        if not isinstance(parts, (tuple, list)) or len(parts) != 4:
            raise ValueError(f"Row {index} must carry exactly four feature groups")
        row = []
        for group, (part, dim) in enumerate(zip(parts, GROUP_DIMS)):
            expected = int(per_row[index, group])
            if part is None:
                if expected != 0:
                    raise ValueError(f"Row {index} group {group} is missing")
                row.append(None)
                continue
            tensor = torch.as_tensor(part, dtype=torch.float32)
            if tensor.ndim == 1:
                # The official feature generator stores each group flattened.
                if tensor.shape[0] != expected * dim:
                    raise ValueError(
                        f"Row {index} group {group} holds {tensor.shape[0]} values, "
                        f"expected {expected * dim} (count {expected} x dim {dim})"
                    )
                tensor = tensor.reshape(expected, dim)
            elif tensor.ndim == 2:
                if tensor.shape != (expected, dim):
                    raise ValueError(
                        f"Row {index} group {group} has shape {tuple(tensor.shape)}, "
                        f"expected {(expected, dim)}"
                    )
            else:
                raise ValueError(
                    f"Row {index} group {group} must be flat or 2D, got "
                    f"{tuple(tensor.shape)}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError(
                    f"Row {index} group {group} contains non-finite values"
                )
            row.append(tensor)
        features.append(tuple(row))

    return RawDataset(
        rows=rows,
        counts=counts,
        row_counts=per_row,
        truth=truth,
        pg=pg,
        features=tuple(features),
    )


def pad_features(dataset, counts):
    """Reproduce the official `features_padding` layout for one dataset."""
    counts = tuple(int(value) for value in counts)
    if len(counts) != 4:
        raise ValueError("counts must hold four group maxima")
    max_join, max_fanout, max_table, max_filter = counts
    for name, observed, allowed in (
        ("n_join_cols", dataset.counts[0], max_join),
        ("n_fanouts", dataset.counts[1], max_fanout),
        ("n_tables", dataset.counts[2], max_table),
        ("n_filter_cols", dataset.counts[3], max_filter),
    ):
        if observed > allowed:
            raise ValueError(
                f"Padding capacity for {name} ({allowed}) is below the data ({observed})"
            )

    width = 40 * (max_join + max_fanout) + 4 * max_table + 43 * max_filter
    mask_width = 1 + max_join + max_fanout + max_table + max_filter
    features = torch.full((dataset.rows, width), PADDING_VALUE, dtype=torch.float32)
    masks = torch.zeros((dataset.rows, mask_width), dtype=torch.int64)
    per_row = dataset.row_counts

    join_offset = 0
    fanout_offset = 40 * max_join
    table_offset = 40 * (max_join + max_fanout)
    filter_offset = 40 * (max_join + max_fanout) + 4 * max_table
    for index, row in enumerate(dataset.features):
        n_join, n_fanout, n_table, n_filter = (int(value) for value in per_row[index])
        join, fanout, table, filter_group = row
        if n_join:
            features[index, join_offset : join_offset + 40 * n_join] = join.reshape(-1)
        if n_fanout:
            features[index, fanout_offset : fanout_offset + 40 * n_fanout] = (
                fanout.reshape(-1)
            )
        if n_table:
            features[index, table_offset : table_offset + 4 * n_table] = table.reshape(
                -1
            )
        if filter_group is not None and n_filter:
            features[index, filter_offset : filter_offset + 43 * n_filter] = (
                filter_group.reshape(-1)
            )

        masks[index, 0] = 1
        cursor = 1
        for size, capacity in (
            (n_join, max_join),
            (n_fanout, max_fanout),
            (n_table, max_table),
            (n_filter, max_filter),
        ):
            masks[index, cursor : cursor + size] = 1
            cursor += capacity

    if max_table:
        table_pad = features[:, table_offset : table_offset + 4 * max_table].reshape(
            dataset.rows, max_table, 4
        )
        for index in range(dataset.rows):
            n_table = int(per_row[index, 2])
            table_pad[index, n_table:, 0] = 0.0
        features[:, table_offset : table_offset + 4 * max_table] = table_pad.reshape(
            dataset.rows, 4 * max_table
        )

    counts_tensor = torch.as_tensor(per_row, dtype=torch.float32)
    return TensorDataset(
        features,
        dataset.truth,
        dataset.pg,
        masks,
        counts_tensor,
    )


def make_loader(dataset, batch_size=MACRO_BATCH_SIZE, shuffle=True):
    """Build the official `DataLoader(dataset, batch_size=..., shuffle=True)`."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_model(checkpoint, expected_sha, counts, device="cpu"):
    """Load the official PRICE encoder from a plain checkpoint state_dict."""
    verify_vendor()
    checkpoint = Path(checkpoint)
    actual = file_sha256(checkpoint)
    if actual != expected_sha:
        raise ValueError(
            f"Checkpoint SHA256 mismatch: expected {expected_sha}, got {actual}"
        )
    from vendor.price_baseline.encoder import RegressionModel

    counts = tuple(int(value) for value in counts)
    with contextlib.redirect_stdout(io.StringIO()):
        model = RegressionModel(*counts, 40, 4, 43, 512, 1024, 1, 256, 6, 8, 0.2)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stripped = strip_state_dict_prefixes(state)
    for name, value in stripped.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor is not finite: {name}")
    model.load_state_dict(stripped, strict=True)
    model.to(device)
    model.eval()
    return model


def forward_batch(model, batch, device):
    """Run the official encoder forward pass on one padded batch."""
    features, truth, pg, mask, counts = batch
    features = features.to(device=device, dtype=torch.float32)
    mask = mask.to(device=device, dtype=torch.int64)
    counts = counts.to(device=device, dtype=torch.float32)
    pg = pg.to(device=device, dtype=torch.float32).reshape(-1, 1)
    pg = torch.log(pg + 1.0) + 1.0
    labels = truth.to(device=device, dtype=torch.float32).reshape(-1)
    labels = torch.log(labels + 1.0) + 1.0
    group_counts = [counts[:, index : index + 1] for index in range(4)]
    output = model(features, pg, mask, *group_counts)
    return output.reshape(-1), labels


def create_optimizer(model):
    """Build the official Adam + OneCycleLR pair."""
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=SCHEDULER_STEPS_PER_EPOCH,
        epochs=SCHEDULER_EPOCHS,
        pct_start=SCHEDULER_PCT_START,
        div_factor=SCHEDULER_DIV_FACTOR,
        anneal_strategy=SCHEDULER_ANNEAL,
    )
    return optimizer, scheduler


def _require_finite(tensor, message):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(message)


def train_epoch(
    model,
    dataset,
    optimizer,
    batch_size=MACRO_BATCH_SIZE,
    microbatch=MICRO_BATCH_SIZE,
    device=None,
    shuffle=True,
):
    """Train one epoch with macro batches of `batch_size` split into microbatches."""
    if device is None:
        device = next(model.parameters()).device
    if isinstance(dataset, DataLoader):
        loader = dataset
    else:
        loader = make_loader(dataset, batch_size=batch_size, shuffle=shuffle)
    use_cuda = torch.device(device).type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
    model.train()
    total_loss = 0.0
    total_examples = 0
    updates = 0
    update_seconds = []
    started = time.perf_counter()
    for batch in loader:
        features, truth, pg, mask, counts = batch
        macro_size = int(features.shape[0])
        if macro_size == 0:
            continue
        if use_cuda:
            torch.cuda.synchronize(device)
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for start in range(0, macro_size, microbatch):
            stop = min(start + microbatch, macro_size)
            micro = (
                features[start:stop],
                truth[start:stop],
                pg[start:stop],
                mask[start:stop],
                counts[start:stop],
            )
            output, labels = forward_batch(model, micro, device)
            _require_finite(output, "Non-finite model output during training")
            micro_loss = torch.mean((output - labels) ** 2)
            _require_finite(micro_loss, "Non-finite training loss")
            (micro_loss * (stop - start) / macro_size).backward()
            loss_sum += float(micro_loss.detach()) * (stop - start)
        for parameter in model.parameters():
            if parameter.grad is not None:
                _require_finite(parameter.grad, "Non-finite gradient during training")
        optimizer.step()
        if use_cuda:
            torch.cuda.synchronize(device)
        updates += 1
        update_seconds.append(time.perf_counter() - update_started)
        total_loss += loss_sum
        total_examples += macro_size
    elapsed = time.perf_counter() - started
    return {
        "train_loss": total_loss / total_examples if total_examples else float("nan"),
        "updates": updates,
        "examples": total_examples,
        "seconds": elapsed,
        "step_seconds": update_seconds,
    }


__all__ = [
    "COUNT_KEYS",
    "GROUP_DIMS",
    "PADDING_VALUE",
    "RawDataset",
    "create_optimizer",
    "file_sha256",
    "forward_batch",
    "load_model",
    "load_raw",
    "make_loader",
    "pad_features",
    "strip_state_dict_prefixes",
    "train_epoch",
    "verify_vendor",
]
