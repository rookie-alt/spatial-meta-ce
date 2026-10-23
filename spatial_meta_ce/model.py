import torch
from torch import nn

from .data import GROUP_DIMS

# arch: 'set'       original model (per-group mean+max pooling)
#       'sum'       variant A: + masked sum pooling (log-cardinality is additive)
#       'attn'      variant B: A + token self-attention across all groups
#       'attn_pair' variant C: B + join-index embedding on fanout tokens
ARCHES = ("set", "sum", "sum_map", "attn", "attn_pair")
MAX_JOINS = 16


class SetEstimator(nn.Module):
    def __init__(
        self,
        width=64,
        use_pg=True,
        arch="set",
        layers=2,
        heads=4,
        pg_output_scale=1.0,
        gate=None,
    ):
        super().__init__()
        if (
            isinstance(pg_output_scale, bool)
            or not isinstance(pg_output_scale, (int, float))
            or pg_output_scale not in (0.0, 1.0)
        ):
            raise ValueError("pg_output_scale must be 0 or 1")
        if gate is not None and gate != "sigmoid":
            raise ValueError("gate must be None or 'sigmoid'")
        if width < 1:
            raise ValueError("width must be positive")
        if arch not in ARCHES:
            raise ValueError(f"Unknown arch {arch!r}")
        # Keep the original config shape for arch='set' so old checkpoints stay byte-identical.
        self.config = {"width": width, "use_pg": use_pg}
        if arch != "set":
            self.config.update({"arch": arch, "layers": layers, "heads": heads})
        self.pg_output_scale = float(pg_output_scale)
        if self.pg_output_scale != 1.0:
            self.config = {**self.config, "pg_output_scale": self.pg_output_scale}
        self.gate_mode = gate
        if gate is not None:
            self.config = {**self.config, "gate": gate}
        self.use_pg = use_pg
        self.arch = arch
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()
                )
                for dim in GROUP_DIMS
            ]
        )
        if arch in ("attn", "attn_pair"):
            self.type_embedding = nn.Embedding(len(GROUP_DIMS), width)
            layer = nn.TransformerEncoderLayer(
                width,
                heads,
                dim_feedforward=2 * width,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
            self.mixer = nn.TransformerEncoder(
                layer, layers, enable_nested_tensor=False
            )
        if arch == "attn_pair":
            self.join_embedding = nn.Embedding(MAX_JOINS, width)
        if arch == "sum_map":
            self.mapping_fusion = nn.Sequential(
                nn.Linear(2 * width, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
            )
            self.fanout_fusion = nn.Sequential(
                nn.Linear(5 * width, width),
                nn.ReLU(),
                nn.Linear(width, width),
                nn.ReLU(),
            )
        per_group = 2 if arch == "set" else 3
        self.head = nn.Sequential(
            nn.Linear(per_group * len(GROUP_DIMS) * width + 6, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if gate is not None:
            # Gate reads the 64-d trunk output that also feeds the final linear, so
            # the zero-initialised residual head still predicts exactly the PG anchor.
            self.gate = nn.Sequential(nn.Linear(64, 1))
            with torch.no_grad():
                # Start near g≈0.5 (residual fully live) rather than exp(-b)≈0.
                self.gate[-1].bias.fill_(0.0)
                self.gate[-1].weight.normal_(std=0.01)

    def _mix(self, tokens, masks):
        # Concatenate all groups into one sequence, run self-attention, split back.
        device = tokens[0].device
        seq = [t + self.type_embedding.weight[i] for i, t in enumerate(tokens)]
        if self.arch == "attn_pair":
            # Fanout tokens come in pairs: positions 2j and 2j+1 belong to join j.
            n = seq[1].shape[1]
            if n > 2 * MAX_JOINS:
                raise ValueError(f"More than {MAX_JOINS} joins; raise MAX_JOINS")
            join_idx = torch.arange(n, device=device) // 2
            seq[1] = seq[1] + self.join_embedding(join_idx)
        lengths = [t.shape[1] for t in seq]
        x = torch.cat(seq, dim=1)
        valid = torch.cat(masks, dim=1)
        x = self.mixer(x, src_key_padding_mask=~valid)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        return list(torch.split(x, lengths, dim=1))

    @staticmethod
    def _gather(tokens, indices, target_mask):
        if indices.ndim != 2 or indices.dtype != torch.long:
            raise ValueError("Mapping indices must be a two-dimensional LongTensor")
        valid = indices >= 0
        if valid.any() and indices[valid].max() >= tokens.shape[1]:
            raise ValueError("Mapping index exceeds token range")
        safe = indices.clamp_min(0)
        gathered_mask = target_mask.gather(1, safe)
        if (valid & ~gathered_mask).any():
            raise ValueError("Mapping points to a padded token")
        result = tokens.gather(1, safe.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))
        return result.masked_fill(~valid.unsqueeze(-1), 0)

    def _map_tokens(self, tokens, masks, mapping):
        table_tokens = tokens[2]
        join_indices = mapping["join_column_tables"]
        filter_indices = mapping["filter_column_tables"]
        fanout_indices = mapping["fanout_join_columns"]
        if (
            join_indices.shape != masks[0].shape
            or filter_indices.shape != masks[3].shape
        ):
            raise ValueError("Mapping/token shape mismatch")
        if fanout_indices.shape != masks[1].shape or fanout_indices.shape[1] % 2:
            raise ValueError("Fanout mapping must contain complete directed pairs")
        for indices, mask in zip(
            (join_indices, fanout_indices, filter_indices),
            (masks[0], masks[1], masks[3]),
        ):
            if not torch.equal(indices >= 0, mask):
                raise ValueError("Mapping presence disagrees with token mask")
        join_tables = self._gather(table_tokens, join_indices, masks[2])
        filter_tables = self._gather(table_tokens, filter_indices, masks[2])
        joined = self.mapping_fusion(torch.cat([tokens[0], join_tables], dim=-1))
        filtered = self.mapping_fusion(torch.cat([tokens[3], filter_tables], dim=-1))
        fanout_join = self._gather(tokens[0], fanout_indices, masks[0])
        opposite = (
            fanout_indices.reshape(fanout_indices.shape[0], -1, 2)
            .flip(-1)
            .reshape_as(fanout_indices)
        )
        opposite_join = self._gather(tokens[0], opposite, masks[0])
        fanout_tables = self._gather(join_tables, fanout_indices, masks[0])
        opposite_tables = self._gather(join_tables, opposite, masks[0])
        fanout = self.fanout_fusion(
            torch.cat(
                [tokens[1], fanout_join, opposite_join, fanout_tables, opposite_tables],
                dim=-1,
            )
        )
        return [
            joined.masked_fill(~masks[0].unsqueeze(-1), 0),
            fanout.masked_fill(~masks[1].unsqueeze(-1), 0),
            table_tokens,
            filtered.masked_fill(~masks[3].unsqueeze(-1), 0),
        ]

    def _head_input(self, batch):
        masks = batch["masks"]
        if self.arch == "sum_map" and "mapping" not in batch:
            raise ValueError("sum_map requires mapping indices")
        if self.arch != "sum_map" and "mapping" in batch:
            raise ValueError("Mapping indices supplied to an unmapped architecture")
        # Mask BEFORE the MLP as well: even arbitrary padding must not overflow.
        tokens = [
            encoder(x.masked_fill(~mask.unsqueeze(-1), 0))
            for encoder, x, mask in zip(self.encoders, batch["groups"], masks)
        ]
        if self.arch == "sum_map":
            tokens = self._map_tokens(tokens, masks, batch["mapping"])
        elif self.arch in ("attn", "attn_pair"):
            tokens = self._mix(tokens, masks)
        pooled = []
        for t, mask in zip(tokens, masks):
            valid = mask.unsqueeze(-1)
            total = (t * valid).sum(1)
            mean = total / valid.sum(1).clamp_min(1)
            largest = t.masked_fill(~valid, -torch.inf).amax(1)
            largest = torch.where(
                mask.any(1, keepdim=True), largest, torch.zeros_like(largest)
            )
            pooled.extend(
                [mean, largest] if self.arch == "set" else [mean, largest, total]
            )
        pg = batch["pg_log"] if self.use_pg else torch.zeros_like(batch["pg_log"])
        log_cartesian = (batch["groups"][2][:, :, 0] * masks[2]).sum(1)
        return torch.cat(
            [
                *pooled,
                torch.log1p(batch["counts"]),
                pg[:, None],
                log_cartesian[:, None],
            ],
            dim=1,
        )

    def output_anchor(self, batch):
        """Return the zero-shot anchor used by the residual output head."""
        if self.use_pg and self.pg_output_scale == 1.0:
            return batch["pg_log"]
        return torch.zeros_like(batch["pg_log"])

    def head_features(self, batch):
        """Return the read-only feature vector entering the final output linear."""
        return self.head[:-1](self._head_input(batch))

    def gate_values(self, batch):
        """Return the P1-2 residual gate g(q) in [0, 1]; requires gate='sigmoid'."""
        if self.gate_mode is None:
            raise ValueError("gate_values requires a gated model")
        return torch.sigmoid(self.gate(self.head_features(batch)).squeeze(-1))

    def residual_values(self, batch):
        x = self._head_input(batch)
        return self.head(x).squeeze(-1)

    def forward(self, batch):
        anchor = self.output_anchor(batch)
        residual = self.residual_values(batch)
        if self.gate_mode is None:
            return anchor + residual
        return anchor + self.gate_values(batch) * residual
