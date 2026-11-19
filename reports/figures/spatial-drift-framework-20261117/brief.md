# Spatial Meta-CE Figure Brief

## Request

Research journal conventions for conceptual method figures and create an advanced, editable framework figure about spatial drift in cross-database cardinality estimation.

## Scope and source of truth

- This is a research-framework and protocol schematic, not a results figure.
- Use the newest project route documented by `实验文档/125-官方26库划分的PRICE元学习实验.md` in worktree `price-meta125-official26-20261117`.
- This route starts from the official PRICE checkpoint P and compares unchanged P, ordinary continued training E, and full-parameter FOMAML M.
- Current protocol split: 26 source databases for training, 0 held-out source-validation databases, 4 target databases for later evaluation. The target set is never used for training, hyperparameter/model/strategy/checkpoint selection.
- Treat these as frozen design details, not proof that the planned run or final target evaluation has completed. Do not show performance gains or imply that FOMAML is effective.

## Method facts to show

- Query features are processed by the original PRICE architecture; retain the four feature groups: join-column histograms, fanout, table features, and filter features. The original regression head also consumes the PG estimate and token counts; show these as auxiliary inputs to the head (not as an added residual anchor).
- A source-database task samples non-overlapping support (20) and query (64) queries.
- Meta-training starts each task from the shared parameters: `phi_0 = theta`.
- Inner loop: full-parameter SGD for five support-loss steps; native target `z = log1p(cardinality) + 1`; native MSE; fixed `alpha = 1e-6`.
- Outer loop: compute query loss after adaptation and use its first-order gradient to update shared `theta`; Adam outer learning rate `2.85e-5`; four sampled tasks per outer update; 2,000 outer updates; use the final checkpoint (no validation-based early stopping or best-checkpoint selection).
- Target use: seal the common checkpoint theta_star, create a fresh task copy for each target episode, adapt that copy only from the target support set, then estimate disjoint target queries. The sealed checkpoint itself is not updated. Include generic K-shot and K=0 zero-shot paths rather than asserting an unfrozen future target matrix. Future P/E/M comparisons must share support/query identities and K settings; report unique-query Q95 as a planned evaluation, not an outcome.

## Required boundaries

- Draw source training and target evaluation as separate database-level regions, with a visible no-leakage boundary.
- In meta-training, draw support and query as disjoint sets. Support labels enter the inner loop; query labels enter only the post-adaptation outer loss.
- In target evaluation, draw target support and test queries as disjoint sets. Support labels are used only for adaptation; test labels only for final scoring.
- Label the distinction between protocol design and measured result.
- Do not mix in the superseded 23/3/4 light model, PG-residual light model, structural-anchor route, LoRA candidate, temporal drift, or online context learning.

## Journal-oriented visual direction

- Chinese labels with necessary technical abbreviations and formula notation.
- Shared asset: VA-196 (`architecture.source.mlsys26-charon_fig2`), source `/Users/williamlin/.codex/skills/vivid-illustration-skill/architecture/source-assets/mlsys26-charon_fig2`. Read the actual asset preview and TEMPLATE.md. Repurpose relevant real stack, panel and feedback-vector objects; remove unrelated GPU parallelism, MoE, KV-cache and cost-chart content.
- Shared canvas: 1800 x 1200 SVG viewBox units, equivalent aspect ratio for both candidates. Aim at body text at least 24 units, key labels 28-32 units. Concise figure labels take precedence over cramming every numerical recipe into the graphic; detailed recipe belongs in README/caption.
- Clean multi-panel vector schematic with a clear left-to-right reading order, compact model detail, visible support/query separation, few high-contrast color roles, black text, no decorative icons or fake charts.
- Keep two candidates at the same canvas size and equivalent text scale: A preserves the chosen source asset's main panel logic; B reorganizes it around the spatial-drift source-to-target narrative.
- Target final display width: 180 mm for comparison only; do not claim journal-specific compliance without checking a selected journal's final specifications.
- Deliver editable SVG, rendered PNG and vector PDF for each candidate, with reproducible source/config and genuine semantic part exports.

## Candidate outputs

- `A/`: preserve composition.
- `B/`: reorganize composition.
- `original.png`: preview of the shared source asset.
- `comparison.html`: equal-width comparison of source preview and completed candidates.

## Primary local references

- `/Users/williamlin/Downloads/13/spatial-meta-ce/.claude/worktrees/price-meta125-official26-20261117/实验文档/125-官方26库划分的PRICE元学习实验.md`
- `/Users/williamlin/Downloads/13/spatial-meta-ce/.claude/worktrees/official-price-meta-plan-20261116/实验文档/123-原版PRICE元学习增量实验方案.md` (model/native loss and FOMAML semantics only; its older split/calibration is superseded by 125)
- `/Users/williamlin/Downloads/13/spatial-meta-ce/.claude/worktrees/official-price-meta-impl-20261116/spatial_meta_ce/price_meta123_model.py`

## Formula and diagram cautions

- P is a read-only common initial checkpoint. Clone into theta for meta-training; the outer update changes theta, never P. The final theta_star is a separate sealed artifact.
- During target adaptation, phi_t changes, not sealed theta_star. “Frozen” applies to the common artifact; all parameters of its task copy may adapt.
- First-order approximation detaches support gradients, not the whole fast weight or query loss. Outer gradients update shared theta through phi.
- Source support/query are disjoint; target support/test are disjoint at normalized query_hash level.
- Histograms/schema sketches illustrate cross-database differences; they are not empirical plots or a claim that an extra graph encoder exists.
- The future target matrix is not specified in protocol 125; omit claims of a frozen K0/5/20 target matrix. Mention K-shot generically.
