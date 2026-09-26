import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

GROUP_DIMS = (40, 40, 4, 43)
TRANSFORM_VERSION = 'price40-group-logfanout-v1'


def batch_samples(samples, device='cpu'):
    if not samples:
        raise ValueError('Cannot batch zero samples')
    labels = np.asarray([s['true'] for s in samples], dtype=np.float64)
    pg = np.asarray([s['pg'] for s in samples], dtype=np.float64)
    if not np.isfinite(labels).all() or not np.isfinite(pg).all() or (labels < 0).any() or (pg < 0).any():
        raise ValueError('Cardinalities must be finite and nonnegative')
    groups, masks = [], []
    counts = []
    for group_idx, dim in enumerate(GROUP_DIMS):
        seqs = [torch.as_tensor(s['groups'][group_idx], dtype=torch.float32) for s in samples]
        if any(t.ndim != 2 or t.shape[1] != dim or not torch.isfinite(t).all() for t in seqs):
            raise ValueError(f'Invalid group {group_idx} shape or nonfinite values')
        if group_idx == 1:
            if any((t < 0).any() for t in seqs):
                raise ValueError('Negative fanout requires investigation')
            seqs = [torch.log1p(t) for t in seqs]
        lengths = [len(t) for t in seqs]
        width = max(1, max(lengths))
        padded = torch.zeros(len(samples), width, dim)
        mask = torch.zeros(len(samples), width, dtype=torch.bool)
        for i, t in enumerate(seqs):
            padded[i, :len(t)] = t
            mask[i, :len(t)] = True
        groups.append(padded.to(device))
        masks.append(mask.to(device))
        counts.append(lengths)
    return {
        'groups': groups, 'masks': masks,
        'counts': torch.tensor(counts, dtype=torch.float32, device=device).T,
        'label': torch.as_tensor(np.log(np.maximum(labels, 1)), dtype=torch.float32, device=device),
        'true': torch.as_tensor(labels, dtype=torch.float64, device=device),
        'pg_log': torch.as_tensor(np.log(np.maximum(pg, 1)), dtype=torch.float32, device=device),
        'ids': [s['id'] for s in samples],
    }


class FeatureStore:
    """Read-only mmap access to a verified, derived database cache."""
    def __init__(self, path):
        import json
        self.path = Path(path)
        with (self.path / 'manifest.json').open() as f:
            self.manifest = json.load(f)
        if self.manifest['schema'] != 1 or self.manifest['transform'] != TRANSFORM_VERSION:
            raise ValueError('Unsupported data cache schema/transform')
        from .manifests import digest
        payload = {k:v for k,v in self.manifest.items() if k != 'content_hash'}
        if digest(payload) != self.manifest.get('content_hash'):
            raise ValueError('Cache manifest content hash mismatch')
        self.rows = self.manifest['rows']
        self.groups = [np.load(self.path / f'group{i}.npy', mmap_mode='r') for i in range(4)]
        self.offsets = np.load(self.path / 'offsets.npy', mmap_mode='r')
        self.cards = np.load(self.path / 'cards.npy', mmap_mode='r')
        if self.cards.shape != (len(self.rows), 2) or self.offsets.shape != (len(self.rows)+1, 4):
            raise ValueError('Cache arrays and manifest disagree')
        for i, dim in enumerate(GROUP_DIMS):
            if self.groups[i].shape != (int(self.offsets[-1, i]), dim):
                raise ValueError('Invalid cache group dimensions')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {'groups': [torch.from_numpy(np.array(g[self.offsets[index, i]:self.offsets[index+1, i]], copy=True))
                           for i, g in enumerate(self.groups)],
                'true': float(self.cards[index, 0]), 'pg': float(self.cards[index, 1]),
                'id': row['id'], 'hash': row['query_hash']}

    def unique_indices(self, exclude=()):
        seen = set(exclude)
        result = []
        for i, row in enumerate(self.rows):
            if row['query_hash'] not in seen:
                result.append(i)
                seen.add(row['query_hash'])
        return np.array(result, dtype=np.int64)

    def batch(self, indices, device='cpu'):
        return batch_samples([self[int(i)] for i in indices], device)


class StorePool:
    def __init__(self, cache_root, max_open=4):
        self.root = Path(cache_root)
        self.max_open = max_open
        self.open = OrderedDict()

    def get(self, split, db):
        key = (split, db)
        if key not in self.open:
            self.open[key] = FeatureStore(self.root / split / db)
        self.open.move_to_end(key)
        while len(self.open) > self.max_open:
            self.open.popitem(last=False)
        return self.open[key]
