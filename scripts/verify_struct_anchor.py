"""Doc 118 step 3 acceptance: an untrained struct-anchor model equals its anchor.

Loads a prepared feature cache with a `struct_log.npy` sidecar, runs the
zero-initialised model with `anchor="struct"` and requires its output to match
the stored structural log bit for bit.  It also reports the Q-error of the
structural anchor against the cache's true cardinalities, so the number can be
compared with the exact `C_uncond` measured in `struct_summary.py bineval`.

    python scripts/verify_struct_anchor.py \
        --cache-root artifacts/cache --split pretrain --database chembl \
        --arch sum --width 64 --limit 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_meta_ce.data import StorePool  # noqa: E402
from spatial_meta_ce.metrics import q_values  # noqa: E402
from spatial_meta_ce.model import SetEstimator  # noqa: E402

TOLERANCE = 1e-5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--split", default="pretrain")
    parser.add_argument("--database", required=True)
    parser.add_argument("--arch", default="sum")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    store = StorePool(args.cache_root).get(args.split, args.database)
    if store.struct_log is None:
        raise SystemExit(f"{args.database}: no struct_log sidecar")
    model = SetEstimator(
        width=args.width, arch=args.arch, use_pg=False, anchor="struct"
    ).to(args.device)
    model.eval()

    indices = np.arange(min(args.limit, len(store)))
    batch = store.batch(indices, args.device)
    with torch.no_grad():
        output = model(batch)
    anchor = batch["struct_log"]
    difference = (output - anchor).abs().max().item()
    if difference > TOLERANCE:
        raise SystemExit(
            f"Untrained model differs from its struct anchor by {difference}"
        )

    log_prediction = anchor.detach().cpu().numpy().astype(np.float64)
    truth = batch["true"].cpu().numpy()
    errors, _, _ = q_values(log_prediction, truth)
    print(
        {
            "database": args.database,
            "rows_checked": int(len(indices)),
            "max_abs_difference": difference,
            "anchor_q_median": float(np.median(errors)),
            "anchor_q95": float(np.percentile(errors, 95)),
            "tolerance": TOLERANCE,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
