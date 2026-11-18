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
LORA_RANK = 4
LORA_ALPHA = 4
LORA_SEED = 42


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
        relation_attention: bool = False,
    ) -> None:
        super().__init__()
        if type(pg_head_input) is not bool or (pg_head_input and arch != "price_lite"):
            raise ValueError("PG head input requires price_lite and a boolean flag")
        if type(relation_attention) is not bool or (
            relation_attention and (arch != "price_lite" or not pg_head_input)
        ):
            raise ValueError("Relation attention requires price_lite, PG head input and a boolean flag")
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
            **({"relation_attention": True} if relation_attention else {}),
        }
        self.pg_head_input = pg_head_input
        self.relation_attention = relation_attention
        self.model_type = (
            "price_lite_pg_residual" if pg_head_input else f"{arch}_residual"
        )
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
        if relation_attention:
            from .price_relations import RELATION_MODEL_TYPE, RelationEncoder

            self.model_type = RELATION_MODEL_TYPE
            # Wrap only after base/LoRA initialization, preserving RNG and state keys.
            self.scale_encoder = RelationEncoder(self.scale_encoder)
            self.filter_encoder = RelationEncoder(self.filter_encoder)
            self._relation_config = copy.deepcopy(self.config)
            self._relation_state_shapes = {
                name: tuple(value.shape) for name, value in self.state_dict().items()
            }

    def validate_relation_identity(self) -> None:
        """Validate construction identity without constraining learned weight values."""
        from .price_relations import RELATION_MODEL_TYPE

        if (
            self.model_type != RELATION_MODEL_TYPE
            or self.relation_attention is not True
            or self.pg_head_input is not True
            or self.config != self._relation_config
            or digest(self.config) != digest(self._relation_config)
            or self.capacity != self._relation_config["capacity"]
            or digest(self.capacity) != digest(self._relation_config["capacity"])
            or self.counts != tuple(self._relation_config["capacity"]["max_counts"])
            or {
                name: tuple(value.shape) for name, value in self.state_dict().items()
            } != self._relation_state_shapes
        ):
            raise ValueError("Noncanonical relation candidate identity/config/state")

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
        if self.relation_attention:
            from .price_relations import validate_relation_batch

            validate_relation_batch(batch, self.counts)
        self._check_counts(batch)
        if (
            ("mapping" in batch and not self.relation_attention)
            or len(batch["groups"]) != 4
            or len(batch["masks"]) != 4
        ):
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
        scale_mask = mask[:, : 1 + sum(self.counts[:2])]
        if self.relation_attention:
            from .price_relations import build_relation_matrices

            scale_relations, full_relations = build_relation_matrices(batch, self.counts)
            scale = self.scale_encoder(scale, scale_mask, scale_relations)
            filtered = self.filter_embedding(scale, x)
            cls = self.filter_encoder(filtered, mask, full_relations)[:, 0, :]
        else:
            scale = self.scale_encoder(scale, scale_mask)
            filtered = self.filter_embedding(scale, x)
            cls = self.filter_encoder(filtered, mask)[:, 0, :]
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
        return torch.cat(features, dim=1)

    def head_features(self, batch: dict) -> torch.Tensor:
        return self.head[:-1](self._head_input(batch))

    def output_anchor(self, batch: dict) -> torch.Tensor:
        return batch["pg_log"]

    def forward(self, batch: dict) -> torch.Tensor:
        return self.output_anchor(batch) + self.head(self._head_input(batch)).squeeze(
            -1
        )
