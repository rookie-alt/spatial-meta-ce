from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .mappings import (
    MAPPING_KEYS,
    MAPPING_MODES,
    TARGETS,
    MappingStore,
    validate_mapping,
)

GROUP_DIMS = (40, 40, 4, 43)
TRANSFORM_VERSION = "price40-group-logfanout-v1"


def batch_samples(samples, device="cpu"):
    if not samples:
        raise ValueError("Cannot batch zero samples")
    mapping_presence = [s.get("mapping") is not None for s in samples]
    if any(mapping_presence) and not all(mapping_presence):
        raise ValueError("Cannot mix mapped and unmapped samples")
    mapping_mode = {
        s.get("mapping_mode") for s in samples if s.get("mapping") is not None
    }
    if len(mapping_mode) > 1 or (
        mapping_mode and not mapping_mode <= set(MAPPING_MODES)
    ):
        raise ValueError("Invalid or mixed mapping modes in one batch")
    labels = np.asarray([s["true"] for s in samples], dtype=np.float64)
    pg = np.asarray([s["pg"] for s in samples], dtype=np.float64)
    if (
        not np.isfinite(labels).all()
        or not np.isfinite(pg).all()
        or (labels < 0).any()
        or (pg < 0).any()
    ):
        raise ValueError("Cardinalities must be finite and nonnegative")
    groups, masks = [], []
    counts = []
    mapped = bool(mapping_presence[0])
    if mapped:
        for sample in samples:
            validate_mapping(sample["mapping"], [len(g) for g in sample["groups"]])
    for group_idx, dim in enumerate(GROUP_DIMS):
        seqs = [
            torch.as_tensor(s["groups"][group_idx], dtype=torch.float32)
            for s in samples
        ]
        if any(
            t.ndim != 2 or t.shape[1] != dim or not torch.isfinite(t).all()
            for t in seqs
        ):
            raise ValueError(f"Invalid group {group_idx} shape or nonfinite values")
        if group_idx == 1:
            if any((t < 0).any() for t in seqs):
                raise ValueError("Negative fanout requires investigation")
            seqs = [torch.log1p(t) for t in seqs]
        lengths = [len(t) for t in seqs]
        width = max(2 if mapped and group_idx == 1 else 1, max(lengths))
        padded = torch.zeros(len(samples), width, dim)
        mask = torch.zeros(len(samples), width, dtype=torch.bool)
        for i, t in enumerate(seqs):
            padded[i, : len(t)] = t
            mask[i, : len(t)] = True
        groups.append(padded.to(device))
        masks.append(mask.to(device))
        counts.append(lengths)
    result = {
        "groups": groups,
        "masks": masks,
        "counts": torch.tensor(counts, dtype=torch.float32, device=device).T,
        "label": torch.as_tensor(
            np.log(np.maximum(labels, 1)), dtype=torch.float32, device=device
        ),
        "true": torch.as_tensor(labels, dtype=torch.float64, device=device),
        "pg_log": torch.as_tensor(
            np.log(np.maximum(pg, 1)), dtype=torch.float32, device=device
        ),
        "ids": [s["id"] for s in samples],
    }
    if mapped:
        mapping_values = {}
        for key, group_index in zip(MAPPING_KEYS, (0, 3, 1)):
            width = groups[group_index].shape[1]
            padded = [
                s["mapping"][key] + [-1] * (width - len(s["mapping"][key]))
                for s in samples
            ]
            mapping_values[key] = torch.tensor(padded, dtype=torch.long, device=device)
        result["mapping"] = mapping_values
        result["mapping_mode"] = next(iter(mapping_mode))
    return result


class FeatureStore:
    """Read-only mmap access to a verified, derived database cache."""

    def __init__(self, path, mapping_path=None, mapping_mode="mapping"):
        import json

        self.path = Path(path)
        with (self.path / "manifest.json").open() as f:
            self.manifest = json.load(f)
        if (
            self.manifest["schema"] != 1
            or self.manifest["transform"] != TRANSFORM_VERSION
        ):
            raise ValueError("Unsupported data cache schema/transform")
        from .manifests import digest

        payload = {k: v for k, v in self.manifest.items() if k != "content_hash"}
        if digest(payload) != self.manifest.get("content_hash"):
            raise ValueError("Cache manifest content hash mismatch")
        self.rows = self.manifest["rows"]
        self.groups = [
            np.load(self.path / f"group{i}.npy", mmap_mode="r") for i in range(4)
        ]
        self.offsets = np.load(self.path / "offsets.npy", mmap_mode="r")
        self.cards = np.load(self.path / "cards.npy", mmap_mode="r")
        if self.cards.shape != (len(self.rows), 2) or self.offsets.shape != (
            len(self.rows) + 1,
            4,
        ):
            raise ValueError("Cache arrays and manifest disagree")
        for i, dim in enumerate(GROUP_DIMS):
            if self.groups[i].shape != (int(self.offsets[-1, i]), dim):
                raise ValueError("Invalid cache group dimensions")
        self.mapping = None
        if mapping_path is not None:
            self.mapping = MappingStore(
                mapping_path,
                mode=mapping_mode,
                expected_database=self.path.name,
                expected_split=self.path.parent.name,
                expected_cache_hash=self.manifest.get("content_hash"),
            )
            self.mapping.bind_cache(self)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {
            "groups": [
                torch.from_numpy(
                    np.array(
                        g[self.offsets[index, i] : self.offsets[index + 1, i]],
                        copy=True,
                    )
                )
                for i, g in enumerate(self.groups)
            ],
            "true": float(self.cards[index, 0]),
            "pg": float(self.cards[index, 1]),
            "id": row["id"],
            "hash": row["query_hash"],
            **(
                {"mapping": self._mapping_for(index), "mapping_mode": self.mapping.mode}
                if self.mapping is not None
                else {}
            ),
        }

    def _mapping_for(self, index):
        mapping = self.mapping[self.rows[index]["query_hash"]]
        expected = np.diff(self.offsets[index : index + 2], axis=0)[0].tolist()
        if mapping["counts"] != expected:
            raise ValueError(f"Mapping/cache counts mismatch at row {index}")
        return mapping

    def unique_indices(self, exclude=()):
        seen = set(exclude)
        result = []
        for i, row in enumerate(self.rows):
            if row["query_hash"] not in seen:
                result.append(i)
                seen.add(row["query_hash"])
        return np.array(result, dtype=np.int64)

    def batch(self, indices, device="cpu"):
        return batch_samples([self[int(i)] for i in indices], device)


class StorePool:
    def __init__(
        self, cache_root, max_open=4, mapping_root=None, mapping_mode="mapping"
    ):
        if mapping_mode not in ("mapping", "shuffled_mapping"):
            raise ValueError(f"Unknown mapping mode: {mapping_mode}")
        self.root = Path(cache_root)
        self.mapping_root = Path(mapping_root) if mapping_root is not None else None
        self.mapping_mode = mapping_mode
        self.max_open = max_open
        self.open = OrderedDict()

    def get(self, split, db):
        if self.mapping_root is not None:
            if (
                not isinstance(db, str)
                or not db
                or Path(db).name != db
                or db in (".", "..")
                or split != "pretrain"
                or db in TARGETS
            ):
                raise ValueError(
                    "Mapping sidecars require source pretrain database names"
                )
            for root, path in (
                (self.root, self.root / split / db),
                (self.mapping_root, self.mapping_root / db),
            ):
                if path.is_symlink() or not path.resolve().is_relative_to(
                    root.resolve()
                ):
                    raise ValueError("Mapping cache/sidecar path escapes its root")
        key = (split, db)
        if key not in self.open:
            mapping_path = (
                self.mapping_root / db if self.mapping_root is not None else None
            )
            self.open[key] = FeatureStore(
                self.root / split / db,
                mapping_path=mapping_path,
                mapping_mode=self.mapping_mode,
            )
        self.open.move_to_end(key)
        while len(self.open) > self.max_open:
            self.open.popitem(last=False)
        return self.open[key]
