"""Query-level source holdout for pre-registered all-source training.

Using every source database for gradient updates leaves no separate validation
database.  This module carves a fixed, hash-derived slice of unique queries out
of each database: the remainder trains, the slice selects the checkpoint.
Selection stays source-only and never reads a target database.

The split is deterministic in ``(fraction, seed, database)``, so the same
configuration reproduces the same holdout on any machine.
"""

import hashlib

import numpy as np

from .manifests import digest, episode_indices

HOLDOUT_SALT = 20261118
MIN_HELDOUT_QUERIES = 200


def database_key(database):
    """Stable per-database seed material (``hash()`` is randomized per run)."""
    return int.from_bytes(hashlib.sha256(database.encode()).digest()[:8], "big")


def split_holdout(unique, fraction, seed, database):
    """Return ``(train_indices, held_indices)`` as sorted int64 arrays."""
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must lie strictly between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("holdout seed must be an integer")
    unique = np.asarray(unique, dtype=np.int64)
    if len(unique) != len(np.unique(unique)):
        raise ValueError(f"{database} unique indices repeat")
    if len(unique) < MIN_HELDOUT_QUERIES:
        raise ValueError(f"{database} has too few unique queries for a holdout")
    count = int(round(len(unique) * fraction))
    if count < MIN_HELDOUT_QUERIES:
        raise ValueError(
            f"{database} holdout would be {count} queries, below the "
            f"{MIN_HELDOUT_QUERIES} required for one episode"
        )
    if len(unique) - count < MIN_HELDOUT_QUERIES:
        raise ValueError(f"{database} would keep too few training queries")
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, HOLDOUT_SALT, database_key(database)])
    )
    permutation = rng.permutation(unique)
    held = np.sort(permutation[:count])
    train = np.sort(permutation[count:])
    if np.intersect1d(held, train).size:
        raise ValueError(f"{database} holdout overlaps its training pool")
    if len(held) + len(train) != len(unique):
        raise ValueError(f"{database} holdout lost queries")
    return train, held


def holdout_digest(entries):
    """Fingerprint the held-out query identities, not just their count."""
    return digest(
        {
            db: {
                "content_hash": entry["content_hash"],
                "held_hashes": entry["held_hashes"],
            }
            for db, entry in sorted(entries.items())
        }
    )


def holdout_validation_plan(entries, support_sizes, runs, query_size, seed):
    """Fixed episodes drawn only from held-out queries."""
    if not support_sizes:
        raise ValueError("holdout validation needs at least one support size")
    if runs < 1 or query_size < 1:
        raise ValueError("holdout validation needs positive runs and query size")
    largest = max(support_sizes)
    episodes = []
    for db, entry in entries.items():
        held = np.asarray(entry["held"], dtype=np.int64)
        if len(held) < largest + query_size:
            raise ValueError(
                f"{db} holdout has {len(held)} queries, fewer than "
                f"K={largest} + query={query_size}"
            )
        rng = np.random.default_rng(
            np.random.SeedSequence([seed, HOLDOUT_SALT, database_key(db), 1])
        )
        for k in support_sizes:
            for run in range(runs):
                support, query = episode_indices(held, k, query_size, rng)
                episodes.append(
                    {
                        "db": db,
                        "k": k,
                        "run": run,
                        "support": support.tolist(),
                        "query": query.tolist(),
                    }
                )
    return episodes
