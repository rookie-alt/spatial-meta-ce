import torch
from torch import nn

from .data import GROUP_DIMS

# arch: 'set'       original model (per-group mean+max pooling)
#       'sum'       variant A: + masked sum pooling (log-cardinality is additive)
#       'attn'      variant B: A + token self-attention across all groups
#       'attn_pair' variant C: B + join-index embedding on fanout tokens
ARCHES = ('set', 'sum', 'attn', 'attn_pair')
MAX_JOINS = 16


class SetEstimator(nn.Module):
    def __init__(self, width=64, use_pg=True, arch='set', layers=2, heads=4):
        super().__init__()
        if width < 1:
            raise ValueError('width must be positive')
        if arch not in ARCHES:
            raise ValueError(f'Unknown arch {arch!r}')
        # Keep the original config shape for arch='set' so old checkpoints stay byte-identical.
        self.config = {'width': width, 'use_pg': use_pg}
        if arch != 'set':
            self.config.update({'arch': arch, 'layers': layers, 'heads': heads})
        self.use_pg = use_pg
        self.arch = arch
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            for dim in GROUP_DIMS
        ])
        if arch in ('attn', 'attn_pair'):
            self.type_embedding = nn.Embedding(len(GROUP_DIMS), width)
            layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, dropout=0.0,
                                               batch_first=True, norm_first=True)
            self.mixer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        if arch == 'attn_pair':
            self.join_embedding = nn.Embedding(MAX_JOINS, width)
        per_group = 2 if arch == 'set' else 3
        self.head = nn.Sequential(nn.Linear(per_group * len(GROUP_DIMS) * width + 6, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def _mix(self, tokens, masks):
        # Concatenate all groups into one sequence, run self-attention, split back.
        device = tokens[0].device
        seq = [t + self.type_embedding.weight[i] for i, t in enumerate(tokens)]
        if self.arch == 'attn_pair':
            # Fanout tokens come in pairs: positions 2j and 2j+1 belong to join j.
            n = seq[1].shape[1]
            if n > 2 * MAX_JOINS:
                raise ValueError(f'More than {MAX_JOINS} joins; raise MAX_JOINS')
            join_idx = torch.arange(n, device=device) // 2
            seq[1] = seq[1] + self.join_embedding(join_idx)
        lengths = [t.shape[1] for t in seq]
        x = torch.cat(seq, dim=1)
        valid = torch.cat(masks, dim=1)
        x = self.mixer(x, src_key_padding_mask=~valid)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        return list(torch.split(x, lengths, dim=1))

    def forward(self, batch):
        masks = batch['masks']
        # Mask BEFORE the MLP as well: even arbitrary padding must not overflow.
        tokens = [encoder(x.masked_fill(~mask.unsqueeze(-1), 0))
                  for encoder, x, mask in zip(self.encoders, batch['groups'], masks)]
        if self.arch in ('attn', 'attn_pair'):
            tokens = self._mix(tokens, masks)
        pooled = []
        for t, mask in zip(tokens, masks):
            valid = mask.unsqueeze(-1)
            total = (t * valid).sum(1)
            mean = total / valid.sum(1).clamp_min(1)
            largest = t.masked_fill(~valid, -torch.inf).amax(1)
            largest = torch.where(mask.any(1, keepdim=True), largest, torch.zeros_like(largest))
            pooled.extend([mean, largest] if self.arch == 'set' else [mean, largest, total])
        pg = batch['pg_log'] if self.use_pg else torch.zeros_like(batch['pg_log'])
        log_cartesian = (batch['groups'][2][:, :, 0] * masks[2]).sum(1)
        x = torch.cat([*pooled, torch.log1p(batch['counts']), pg[:, None], log_cartesian[:, None]], dim=1)
        return pg + self.head(x).squeeze(-1)
