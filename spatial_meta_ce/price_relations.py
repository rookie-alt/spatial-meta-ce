"""Fixed query-local relation bias for the opt-in A1-PG candidate.

Rows are attention queries and columns are attention keys. Relation IDs:
0 none (also CLS edges/padding), 1 self, 2 column->table, 3 table->column,
4 fanout->origin join column, 5 join column->origin fanout, 6 paired fanouts,
7 same local table. Columns in 2/3 are join or filter columns. A fanout's
table is inherited from its origin join column. Precedence, highest first:
self > paired fanouts > fanout/join > column/table > same table > none.

All associations use query-local indices only. Mapping mode selects input
sidecar indices, never weights or architecture. No labels or PG values enter
relation construction; source-side mapping certification is a separate gate.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import GROUP_DIMS
from .mappings import MAPPING_KEYS, MAPPING_MODES, TARGETS

RELATION_NAMES = (
    "none",
    "self",
    "column_to_table",
    "table_to_column",
    "fanout_to_join",
    "join_to_fanout",
    "paired_fanout",
    "same_table",
)
RELATION_MODEL_TYPE = "price_lite_pg_relation_residual"


def validate_relation_batch(batch: dict, capacities: tuple[int, ...]) -> None:
    """Fail closed for direct batch calls as well as MappingStore callers."""
    counts = batch.get("counts")
    if (
        len(capacities) != 4
        or any(type(n) is not int or n < 0 for n in capacities)
        or capacities[2] < 1
    ):
        raise ValueError("Invalid relation token capacities")
    if (
        not isinstance(counts, torch.Tensor)
        or counts.ndim != 2
        or counts.shape[1] != 4
        or counts.shape[0] < 1
        or counts.dtype == torch.bool
        or counts.is_complex()
        or counts.device.type == "meta"
        or not torch.isfinite(counts).all()
        or (counts < 0).any()
        or not torch.equal(counts, counts.round())
        or (counts > counts.new_tensor(capacities)).any()
        or (counts[:, 2] < 1).any()
        or (counts[:, 1] % 2 != 0).any()
    ):
        raise ValueError("Invalid relation counts or incomplete directed fanout pairs")
    size = len(counts)
    databases = batch.get("databases")
    if (
        not isinstance(databases, (list, tuple))
        or len(databases) != size
        or any(not isinstance(db, str) or not db.strip() for db in databases)
    ):
        raise ValueError(
            "Relation attention requires explicit source database identities"
        )
    if any(db.strip().lower() in TARGETS for db in databases):
        raise ValueError(
            "Relation attention is source-only; target databases are rejected"
        )
    mode = batch.get("mapping_mode")
    if not isinstance(mode, str) or mode not in MAPPING_MODES:
        raise ValueError("Relation attention requires a single valid mapping mode")
    groups, masks = batch.get("groups"), batch.get("masks")
    if (
        not isinstance(groups, (list, tuple))
        or not isinstance(masks, (list, tuple))
        or len(groups) != 4
        or len(masks) != 4
    ):
        raise ValueError("Relation attention requires four feature groups and masks")
    for i, (group, mask, dim) in enumerate(zip(groups, masks, GROUP_DIMS)):
        if (
            not isinstance(group, torch.Tensor)
            or not isinstance(mask, torch.Tensor)
            or group.ndim != 3
            or group.shape[0] != size
            or group.shape[2] != dim
            or not group.is_floating_point()
            or mask.shape != group.shape[:2]
            or mask.dtype != torch.bool
            or group.device != counts.device
            or mask.device != counts.device
        ):
            raise ValueError(
                "Relation group/mask batch, shape, dtype or device mismatch"
            )
        expected = (
            torch.arange(group.shape[1], device=counts.device)[None, :]
            < counts[:, i, None]
        )
        if not torch.equal(mask, expected) or not torch.equal(
            mask.sum(1), counts[:, i]
        ):
            raise ValueError("Relation mask/counts mismatch (prefix padding required)")
    mapping = batch.get("mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(MAPPING_KEYS):
        raise ValueError(
            "Relation attention requires exactly the three mapping tensors"
        )
    for key, source, destination in zip(MAPPING_KEYS, (0, 3, 1), (2, 2, 0)):
        indices = mapping[key]
        mask = masks[source]
        if (
            not isinstance(indices, torch.Tensor)
            or indices.dtype != torch.long
            or indices.shape != mask.shape
            or indices.device != counts.device
        ):
            raise ValueError(
                f"Relation mapping shape, long dtype or device mismatch: {key}"
            )
        if (
            (indices[~mask] != -1).any()
            or (indices[mask] < 0).any()
            or (mask & (indices >= counts[:, destination, None])).any()
        ):
            raise ValueError(
                f"Relation mapping presence, -1 padding or real-token range mismatch: {key}"
            )


def build_relation_matrices(
    batch: dict, capacities: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return scale/full matrices using capacity offsets, independent of padding width."""
    validate_relation_batch(batch, capacities)
    counts = batch["counts"]
    device = counts.device
    size = len(counts)
    nj, nf, nt, nc = capacities
    join_start, fanout_start = 1, 1 + nj
    table_start, filter_start = 1 + nj + nf, 1 + nj + nf + nt
    length = 1 + sum(capacities)

    def pad_mapping(key, width):
        values = batch["mapping"][key]
        padded = torch.full((size, width), -1, dtype=torch.long, device=device)
        copied = min(width, values.shape[1])
        padded[:, :copied] = values[:, :copied]
        return padded

    joins = pad_mapping("join_column_tables", nj)
    filters = pad_mapping("filter_column_tables", nc)
    fanouts = pad_mapping("fanout_join_columns", nf)
    tables = torch.arange(nt, device=device)[None, :].expand(size, -1)
    tables = tables.masked_fill(tables >= counts[:, 2, None], -1)
    # Zero join capacity implies no real fanout; avoid gather into a zero axis.
    fanout_tables = joins.gather(1, fanouts.clamp_min(0)) if nj else fanouts.clone()
    fanout_tables = fanout_tables.masked_fill(fanouts < 0, -1)
    owners = torch.cat(
        [
            torch.full((size, 1), -1, dtype=torch.long, device=device),
            joins,
            fanout_tables,
            tables,
            filters,
        ],
        dim=1,
    )
    real = owners >= 0
    real[:, 0] = True
    same_table = (owners[:, :, None] == owners[:, None, :]) & (owners[:, :, None] >= 0)
    relations = torch.zeros(size, length, length, dtype=torch.long, device=device)
    relations.masked_fill_(same_table, 7)

    def edges(indices, source_start, destination_start, forward_id, reverse_id):
        rows, positions = (indices >= 0).nonzero(as_tuple=True)
        source = positions + source_start
        destination = indices[rows, positions] + destination_start
        relations[rows, source, destination] = forward_id
        relations[rows, destination, source] = reverse_id

    edges(joins, join_start, table_start, 2, 3)
    edges(filters, filter_start, table_start, 2, 3)
    edges(fanouts, fanout_start, join_start, 4, 5)
    rows, positions = (fanouts >= 0).nonzero(as_tuple=True)
    relations[rows, positions + fanout_start, (positions ^ 1) + fanout_start] = 6
    rows, positions = real.nonzero(as_tuple=True)
    relations[rows, positions, positions] = 1
    scale_length = 1 + nj + nf
    return relations[:, :scale_length, :scale_length], relations


class RelationHead(nn.Module):
    """Reuse vendor projections and mask/dropout behavior, with zero-init bias."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.head_size = original.head_size
        self.n_embd = original.n_embd
        self.dropout_rate = original.dropout_rate
        self.Key, self.Query, self.Value = original.Key, original.Query, original.Value
        self.relation_bias = nn.Parameter(
            original.Query.weight.new_zeros(len(RELATION_NAMES))
        )

    def attention_weights(self, inputs, padding_mask, relations):
        keys, queries = self.Key(inputs), self.Query(inputs)
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(
            keys.size(-1)
        )
        scores = scores + self.relation_bias[relations].to(scores.dtype)
        if padding_mask is not None:
            min_value = -65504 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(padding_mask.unsqueeze(1) == 0, min_value)
        weights = F.softmax(scores, dim=-1)
        return F.dropout(weights, p=self.dropout_rate, training=self.training)

    def forward(self, inputs, padding_mask, relations):
        weights = self.attention_weights(inputs, padding_mask, relations)
        return torch.matmul(weights, self.Value(inputs))


class RelationMultiHeadAttention(nn.Module):
    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.heads = nn.ModuleList([RelationHead(head) for head in original.heads])
        self.projection = original.projection

    def forward(self, inputs, padding_mask, relations):
        outputs = [head(inputs, padding_mask, relations) for head in self.heads]
        return self.projection(torch.cat(outputs, dim=-1))


class RelationBlock(nn.Module):
    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.attention = RelationMultiHeadAttention(original.attention)
        self.norm1, self.norm2 = original.norm1, original.norm2
        self.feed_forward = original.feed_forward
        self.dropout_rate = original.dropout_rate

    def forward(self, inputs, padding_mask, relations):
        normalized = self.norm1(inputs)
        attention = normalized + self.attention(normalized, padding_mask, relations)
        attention = F.dropout(attention, p=self.dropout_rate, training=self.training)
        normalized = self.norm2(attention)
        feed_forward = self.feed_forward(normalized)
        feed_forward = F.dropout(
            feed_forward, p=self.dropout_rate, training=self.training
        )
        return normalized + feed_forward


class RelationEncoder(nn.Module):
    """Wrap an initialized vendor encoder without consuming RNG or renaming base weights."""

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([RelationBlock(block) for block in original.blocks])
        self.norm, self.linear, self.elu = original.norm, original.linear, original.elu
        self.output = original.output
        self.dropout_rate = original.dropout_rate

    def forward(self, inputs, padding_mask, relations):
        output = inputs
        for block in self.blocks:
            output = block(output, padding_mask, relations)
        output = self.elu(self.linear(self.norm(output)))
        output = F.dropout(output, p=self.dropout_rate, training=self.training)
        return self.output(output)
