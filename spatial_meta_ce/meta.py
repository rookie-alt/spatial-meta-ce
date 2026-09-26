from collections import OrderedDict

import torch
from torch.func import functional_call
from torch.nn import functional as F


def loss_value(pred, label, loss='smooth_l1'):
    if pred.shape != label.shape:
        raise ValueError('Prediction/label shape mismatch')
    if loss not in ('smooth_l1', 'mse'):
        raise ValueError('Unknown loss')
    value = F.smooth_l1_loss(pred, label) if loss == 'smooth_l1' else F.mse_loss(pred, label)
    if not torch.isfinite(value):
        raise FloatingPointError('Nonfinite loss; refusing to silently skip task')
    return value


def forward(model, params, batch):
    return functional_call(model, params, (batch,), strict=True)


def adapt(model, support, lr=0.001, steps=5, scope='all', loss='smooth_l1', training=True):
    if steps < 0 or lr <= 0 or scope not in ('all', 'head'):
        raise ValueError('Invalid adaptation configuration')
    params = OrderedDict(model.named_parameters())
    if steps == 0 or support is None:
        return params
    names = [n for n in params if scope == 'all' or n.startswith('head.')]
    if not names:
        raise ValueError('No adaptation parameters selected')
    with torch.enable_grad():
        for _ in range(steps):
            objective = loss_value(forward(model, params, support), support['label'], loss)
            grads = torch.autograd.grad(objective, [params[n] for n in names], create_graph=False)
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite inner gradient')
            updated = dict(zip(names, grads))
            params = OrderedDict((n, p - lr * updated[n] if n in updated else p) for n, p in params.items())
            if not training:
                params = OrderedDict((n, p.detach().requires_grad_(True)) for n, p in params.items())
    return params


def meta_loss(model, support, query, lr, steps, scope='all', loss='smooth_l1'):
    params = adapt(model, support, lr, steps, scope, loss)
    return loss_value(forward(model, params, query), query['label'], loss)
