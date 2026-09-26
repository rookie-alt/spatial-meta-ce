import torch
from torch import nn

from .data import GROUP_DIMS


class SetEstimator(nn.Module):
    def __init__(self, width=64, use_pg=True):
        super().__init__()
        if width < 1:
            raise ValueError('width must be positive')
        self.config = {'width': width, 'use_pg': use_pg}
        self.use_pg = use_pg
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
            for dim in GROUP_DIMS
        ])
        self.head = nn.Sequential(nn.Linear(8 * width + 6, 128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        pooled = []
        for encoder, x, mask in zip(self.encoders, batch['groups'], batch['masks']):
            # Mask BEFORE the MLP as well: even arbitrary padding must not overflow.
            tokens = encoder(x.masked_fill(~mask.unsqueeze(-1), 0))
            valid = mask.unsqueeze(-1)
            mean = (tokens * valid).sum(1) / valid.sum(1).clamp_min(1)
            largest = tokens.masked_fill(~valid, -torch.inf).amax(1)
            largest = torch.where(mask.any(1, keepdim=True), largest, torch.zeros_like(largest))
            pooled.extend([mean, largest])
        pg = batch['pg_log'] if self.use_pg else torch.zeros_like(batch['pg_log'])
        log_cartesian = (batch['groups'][2][:, :, 0] * batch['masks'][2]).sum(1)
        x = torch.cat([*pooled, torch.log1p(batch['counts']), pg[:, None], log_cartesian[:, None]], dim=1)
        return pg + self.head(x).squeeze(-1)
