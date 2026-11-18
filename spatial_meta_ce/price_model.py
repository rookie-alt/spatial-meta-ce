"""PRICE-style two-stage encoder with optional PG residual-head input."""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from vendor.price_baseline.module import Encoder, FilterEmbedding, ScaleEmbedding

from .data import GROUP_DIMS
from .manifests import digest
from .price_capacity import GROUP_NAMES, UnsupportedCapacity, validate_capacity

PRICE_ARCHES = ("price_lite", "price_tiny")
GROUP_POOLING_MODES = ("none", "mean", "attention")
GROUP_POOLING_MODEL_TYPES = (
    "price_lite_pg_group_mean_residual",
    "price_lite_pg_group_attention_residual",
)
LORA_RANK = 4
LORA_ALPHA = 4
LORA_SEED = 42


def price_model_type(
    arch: str, pg_head_input: bool = False, group_pooling: str = "none"
) -> str:
    """Validate opt-in readout controls and return their canonical identity."""
    if type(pg_head_input) is not bool or (pg_head_input and arch != "price_lite"):
        raise ValueError("PG head input requires price_lite and a boolean flag")
    if type(group_pooling) is not str or group_pooling not in GROUP_POOLING_MODES:
        raise ValueError("group_pooling must be 'none', 'mean' or 'attention'")
    if group_pooling != "none":
        if arch != "price_lite" or not pg_head_input:
            raise ValueError(
                "group pooling requires price_lite with pg_head_input=True"
            )
        return f"price_lite_pg_group_{group_pooling}_residual"
    return "price_lite_pg_residual" if pg_head_input else f"{arch}_residual"


class LoRALinear(nn.Linear):
    """Unmerged per-head Q/V projection, with functional_call-visible A/B."""

    def __init__(self, original: nn.Linear, generator: torch.Generator) -> None:
        super().__init__(original.in_features, original.out_features)
        self.weight = original.weight
        self.bias = original.bias
        self.lora_A = nn.Parameter(torch.empty(LORA_RANK, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, LORA_RANK))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=generator)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self.weight, self.bias) + (
            LORA_ALPHA / LORA_RANK
        ) * F.linear(F.linear(inputs, self.lora_A), self.lora_B)


class PriceResidualEstimator(nn.Module):
    def __init__(
        self,
        capacity: dict,
        arch: str = "price_lite",
        d_model: int | None = None,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
        lora: bool = False,
        lora_seed: int = LORA_SEED,
        pg_head_input: bool = False,
        group_pooling: str = "none",
    ) -> None:
        super().__init__()
        model_type = price_model_type(arch, pg_head_input, group_pooling)
        expected_dim = 64 if arch == "price_lite" else 32
        if arch not in PRICE_ARCHES or d_model not in (None, expected_dim):
            raise ValueError("PRICE model_type/arch/d_model configuration mismatch")
        if layers != 2 or heads != 4 or dropout != 0.0:
            raise ValueError("122 freezes layers=2, heads=4, dropout=0")
        if (
            type(lora) is not bool
            or type(lora_seed) is not int
            or lora_seed != LORA_SEED
        ):
            raise ValueError("122 freezes LoRA A initialization seed=42")
        self.capacity = copy.deepcopy(validate_capacity(capacity))
        self.counts = tuple(self.capacity["max_counts"])
        self.config = {
            "arch": arch,
            "d_model": expected_dim,
            "layers": layers,
            "heads": heads,
            "dropout": float(dropout),
            "capacity": copy.deepcopy(self.capacity),
            "lora": lora,
            "lora_seed": lora_seed,
            **({"pg_head_input": True} if pg_head_input else {}),
            **({"group_pooling": group_pooling} if group_pooling != "none" else {}),
        }
        self.pg_head_input = pg_head_input
        self.group_pooling = group_pooling
        self.model_type = model_type
        self.scale_embedding = ScaleEmbedding(*self.counts[:2], 40, expected_dim)
        self.filter_embedding = FilterEmbedding(*self.counts, 40, 4, 43, expected_dim)
        self.scale_encoder = Encoder(expected_dim, layers, heads, dropout)
        self.filter_encoder = Encoder(expected_dim, layers, heads, dropout)
        self.len_net = nn.Linear(4, 16)
        self.head = nn.Sequential(
            nn.Linear(expected_dim + 16 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        if pg_head_input:
            original = self.head[0]
            with torch.no_grad():
                weight = torch.cat(
                    [original.weight, original.weight.new_zeros(128, 1)], dim=1
                )
            self.head[0] = nn.Linear(original.in_features + 1, 128)
            with torch.no_grad():
                self.head[0].weight.copy_(weight)
                self.head[0].bias.copy_(original.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if lora:
            generator = torch.Generator().manual_seed(lora_seed)
            for encoder in (self.scale_encoder, self.filter_encoder):
                for block in encoder.blocks:
                    for attention_head in block.attention.heads:
                        attention_head.Query = LoRALinear(
                            attention_head.Query, generator
                        )
                        attention_head.Value = LoRALinear(
                            attention_head.Value, generator
                        )
        if group_pooling != "none":
            # Build the unchanged baseline first, preserving all shared initial
            # parameters and RNG consumption when pooling is disabled. Append
            # new columns so [CLS, lengths, Cartesian size, PG] keeps its order.
            original = self.head[0]
            self.head[0] = nn.Linear(original.in_features + 4 * expected_dim, 128)
            with torch.no_grad():
                self.head[0].weight[:, : original.in_features].copy_(
                    original.weight
                )
                self.head[0].bias.copy_(original.bias)
            if group_pooling == "attention":
                self.group_scorers = nn.ModuleList(
                    nn.Sequential(nn.Linear(64, 16), nn.Tanh(), nn.Linear(16, 1))
                    for _ in GROUP_NAMES
                )
            self._pooling_config = copy.deepcopy(self.config)

    def validate_pooling_identity(self) -> None:
        """Fail closed on a mutated candidate identity/config before saving."""
        expected = price_model_type("price_lite", True, self.group_pooling)
        if (
            self.model_type != expected
            or self.group_pooling != self._pooling_config["group_pooling"]
            or self.pg_head_input is not True
            or digest(self.config) != digest(self._pooling_config)
            or digest(self.capacity) != digest(self._pooling_config["capacity"])
            or self.counts != tuple(self._pooling_config["capacity"]["max_counts"])
            or self.head[0].in_features != 338
            or (self.group_pooling == "attention") != hasattr(self, "group_scorers")
        ):
            raise ValueError("Noncanonical group pooling candidate identity/config")

    def _pool_groups(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Read four final token groups at fixed capacity offsets, excluding CLS."""
        if self.group_pooling == "none":
            raise ValueError("Group pooling is disabled")
        pooled = []
        offset = 1
        for group, width in enumerate(self.counts):
            valid = mask[:, offset : offset + width]
            # Mask before score/value arithmetic: padding can be nonzero, inf
            # or NaN, so multiplying by a boolean is insufficient.
            values = tokens[:, offset : offset + width].masked_fill(
                ~valid[..., None], 0
            )
            if self.group_pooling == "mean":
                summary = values.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
            elif width == 0:
                summary = values.sum(1)
                # With no tokens this scorer is unused; expose its zero
                # derivative to strict functional SGD without dummy tokens.
                for parameter in self.group_scorers[group].parameters():
                    summary = summary + 0 * parameter.sum()
            else:
                scores = self.group_scorers[group](values).squeeze(-1)
                scores = scores.masked_fill(~valid, -torch.inf)
                # Avoid softmax(-inf, ... -inf) on empty rows, then force
                # every invalid weight to exact zero.
                scores = torch.where(valid.any(1, keepdim=True), scores, 0)
                weights = scores.softmax(1).masked_fill(~valid, 0)
                summary = (values * weights[..., None]).sum(1)
            pooled.append(summary)
            offset += width
        return torch.cat(pooled, dim=1)

    def _check_counts(self, batch: dict) -> None:
        counts = batch["counts"]
        if counts.ndim != 2 or counts.shape[1] != 4 or not torch.isfinite(counts).all():
            raise ValueError("Invalid PRICE batch counts")
        if (counts < 0).any() or not torch.equal(counts, counts.round()):
            raise ValueError("PRICE batch counts must be nonnegative integers")
        overflow = counts > counts.new_tensor(self.counts)
        if overflow.any():
            records = []
            for row, group in overflow.nonzero().tolist():
                records.append(
                    {
                        "database": batch["databases"][row],
                        "query_hash": batch["hashes"][row],
                        "group": GROUP_NAMES[group],
                        "actual": int(counts[row, group]),
                        "capacity": self.counts[group],
                    }
                )
            raise UnsupportedCapacity(records)

    def price_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore raw fanout, then pad to the checkpoint's fixed capacities."""
        self._check_counts(batch)
        if "mapping" in batch or len(batch["groups"]) != 4 or len(batch["masks"]) != 4:
            raise ValueError(
                "PRICE candidates require the four unmapped feature groups"
            )
        vectors = []
        masks = [
            torch.ones(
                len(batch["counts"]), 1, dtype=torch.bool, device=batch["counts"].device
            )
        ]
        for i, (group, mask, n, dim) in enumerate(
            zip(batch["groups"], batch["masks"], self.counts, GROUP_DIMS)
        ):
            if (
                group.ndim != 3
                or group.shape[2] != dim
                or mask.shape != group.shape[:2]
            ):
                raise ValueError("Invalid PRICE group/mask shape")
            expected = (
                torch.arange(group.shape[1], device=group.device)[None, :]
                < batch["counts"][:, i, None]
            )
            if (
                mask.dtype != torch.bool
                or not torch.equal(mask, expected)
                or not torch.equal(
                    mask.sum(1).to(batch["counts"].dtype), batch["counts"][:, i]
                )
            ):
                raise ValueError("PRICE mask/counts mismatch (prefix padding required)")
            raw = group.masked_fill(~mask[..., None], 0)
            raw = torch.expm1(raw) if i == 1 else raw
            if not torch.isfinite(raw).all():
                raise FloatingPointError("Nonfinite PRICE input/fanout")
            padded = group.new_full((len(group), n, dim), -1000.0)
            if i == 2:
                padded[:, :, 0] = 0
            # Counts were checked above; this only discards empty batch padding.
            width = min(group.shape[1], n)
            padded[:, :width] = torch.where(
                mask[:, :width, None], raw[:, :width], padded[:, :width]
            )
            valid = torch.zeros(len(group), n, dtype=torch.bool, device=group.device)
            valid[:, :width] = mask[:, :width]
            vectors.append(padded.flatten(1))
            masks.append(valid)
        return torch.cat(vectors, dim=1), torch.cat(masks, dim=1)

    def _head_input(self, batch: dict) -> torch.Tensor:
        x, mask = self.price_inputs(batch)
        scale = self.scale_embedding(x)
        scale = self.scale_encoder(scale, mask[:, : 1 + sum(self.counts[:2])])
        filtered = self.filter_embedding(scale, x)
        tokens = self.filter_encoder(filtered, mask)
        cls = tokens[:, 0, :]
        # A zero-capacity group has no vendor embedding call. Keep an explicit
        # zero derivative so strict functional SGD/audits work without dummy tokens.
        embeddings = (
            self.scale_embedding.join_hist_embeddings,
            self.scale_embedding.fanout_embeddings,
            self.filter_embedding.table_embeddings,
            self.filter_embedding.filter_embeddings,
        )
        for count, embedding in zip(self.counts, embeddings):
            if count == 0:
                cls = cls + 0 * (embedding.weight.sum() + embedding.bias.sum())
        lengths = self.len_net(torch.log1p(batch["counts"]))
        tables = batch["groups"][2].masked_fill(~batch["masks"][2][..., None], 0)
        log_cartesian = tables[:, :, 0].sum(1, keepdim=True)
        features = [cls, lengths, log_cartesian]
        if self.pg_head_input:
            features.append(batch["pg_log"][:, None])
        if self.group_pooling != "none":
            features.append(self._pool_groups(tokens, mask))
        return torch.cat(features, dim=1)

    def head_features(self, batch: dict) -> torch.Tensor:
        return self.head[:-1](self._head_input(batch))

    def output_anchor(self, batch: dict) -> torch.Tensor:
        return batch["pg_log"]

    def forward(self, batch: dict) -> torch.Tensor:
        return self.output_anchor(batch) + self.head(self._head_input(batch)).squeeze(
            -1
        )
