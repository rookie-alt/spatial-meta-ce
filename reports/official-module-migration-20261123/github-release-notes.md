# Official26/4 module training inputs

This release contains the certified inputs for five A1-PG module arms × seeds 42/43/44, 24,000 updates per run. It provides data and deployment tools, not training authorization or model weights.

The original archive is split into three raw byte chunks because GitHub limits each release asset to less than 2 GiB. Download all assets, check each chunk, and concatenate them in the explicit order below. Individual chunks cannot be extracted independently.

Original archive: `module26-official26-4-inputs.tar.gz`, **3,049,224,614 bytes**, containing **634 files / 17,012,110,734 uncompressed bytes**.

Original SHA256: `9077240a3e1b560aed7eb0366049f6195629f26c647ae04224028c2613aaea73`.

## Download and reconstruct

On the new Linux server, create a fresh download directory. If the repository is public, authentication is optional for direct asset URLs; GitHub CLI may ask for login if it is not already configured.

```bash
set -euo pipefail
mkdir -p /gly/weilin
mkdir /gly/weilin/module26-migration-20261123
gh release download module26-migration-20261123 \
  --repo rookie-alt/spatial-meta-ce \
  --dir /gly/weilin/module26-migration-20261123
cd /gly/weilin/module26-migration-20261123
sha256sum -c PARTS-SHA256SUMS
test ! -e module26-official26-4-inputs.tar.gz
cat module26-official26-4-inputs.tar.gz.part01 \
    module26-official26-4-inputs.tar.gz.part02 \
    module26-official26-4-inputs.tar.gz.part03 \
  > module26-official26-4-inputs.tar.gz
sha256sum -c SHA256SUMS
sha256sum -c TOOL-SHA256SUMS
```

Then follow `DEPLOYMENT.md` from section 4 for full archive readback, isolated extraction, restoration of the original real `/gly/weilin` input paths, code checkout, environment setup, CUDA verification and unsigned spec preparation. Do not run the previous server's queue or signed authorizations.

The required 37 training source files match commit `26d7056935e445c5c3d27926fd5e90e6dd5f4e06` on branch `feat/official-module-training-prep-20261118`. The data archive preserves the earlier certification provenance without rewriting it.

This is an offline feature-training bundle; no PostgreSQL database service, raw database table installation or previous checkpoint is required. The original input archive and downloaded local copy were both fully read back and checked, 634/634 files. New-server CUDA availability and deployment must still be verified on that server.

## Historical A1-PG 24k checkpoint supplement

`old-a1-pg24k-reference.tar.gz` adds all three original A1-PG seed42/43/44 `last.pt` checkpoints and their original configs, training/completion records and audits. It is **13,725,084 bytes**, SHA256 `f082cd5a63f3997ffd2bf01afd92234bd205f750866628e585b5c68a5602dda2`.

Download that archive plus `OLD-A1-PG24K-*` assets, check `OLD-A1-PG24K-SHA256SUMS`, and follow `OLD-A1-PG24K-README.md`. Already-downloaded input parts do not need to be downloaded again. The supplement contains 24 original files and two embedded metadata records, preserves original checkpoint bytes, and does not change any of the initial 30 assets.

Each checkpoint strictly loads on CPU as `price_lite_pg_residual`, with `next_step=24000`, `optimizer_steps=24000` and 294,033 parameters. Original metadata `fixed_steps=4000` and selection text are stale legacy fields; actual step counters, configs, preregistered budgets, sealed plans and completion records confirm 24,000. This discrepancy is disclosed rather than rewriting the original files.

These are historical reference checkpoints from the earlier `122-official-26 / 122-A1-PG` exploratory experiment. They are not completion artifacts for the new `132-module26` five-arm matrix and do not change its from-scratch initialization. Migration does not launch training or evaluation.
