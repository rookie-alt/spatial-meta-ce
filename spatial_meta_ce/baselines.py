import contextlib
import io
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PriceAdapter(nn.Module):
    """Isolated PRICE baseline with its original log1p(card)+1 output."""
    def __init__(self, checkpoint, counts, device='cpu'):
        super().__init__()
        from vendor.price_baseline.encoder import RegressionModel
        self.counts = tuple(int(x) for x in counts)
        with contextlib.redirect_stdout(io.StringIO()):
            self.price = RegressionModel(*self.counts, 40, 4, 43, 512, 1024, 1, 256, 6, 8, 0.2)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
            raise ValueError('PRICE reference requires a plain state_dict, not a training checkpoint')
        prefix = all(k.startswith('module.') for k in state)
        if prefix:
            state = {k[7:]:v for k,v in state.items()}
        self.price.load_state_dict(state, strict=True)
        self.to(device)
        self.eval()
        self.reference_path = str(checkpoint)

    def forward(self, batch):
        vectors, masks = [], [torch.ones(len(batch['pg_log']),1,device=batch['pg_log'].device)]
        for j,(g,mask,n,dim) in enumerate(zip(batch['groups'],batch['masks'],self.counts,(40,40,4,43))):
            counts = mask.sum(1)
            if (counts > n).any():
                raise ValueError('PRICE padding capacity too small')
            raw = torch.expm1(g) if j==1 else g
            padded = torch.full((len(g),n,dim),-1000.,device=g.device)
            if j==2:
                padded[:,:,0]=0
            width = min(g.shape[1],n)
            padded[:,:width] = torch.where(mask[:,:width,None],raw[:,:width],padded[:,:width])
            vectors.append(padded.flatten(1))
            m = torch.zeros(len(g),n,device=g.device)
            m[:,:width] = mask[:,:width]
            masks.append(m)
        pg = torch.logaddexp(batch['pg_log'],torch.zeros_like(batch['pg_log']))+1
        output = self.price(torch.cat(vectors,1),pg[:,None],torch.cat(masks,1),
                            *[batch['counts'][:,i:i+1] for i in range(4)])
        return output.flatten()


def price_log_predictions(encoded):
    a = np.asarray(encoded,dtype=np.float64)-1
    # Decode max(exp(a)-1,1) without overflowing or log of a negative value.
    result = np.zeros_like(a)
    positive = a > np.log(2)
    result[positive] = a[positive]+np.log1p(-np.exp(-a[positive]))
    return result


def price_support(batch):
    result = dict(batch)
    result['label'] = (torch.log1p(batch['true'])+1).to(torch.float32)
    return result
