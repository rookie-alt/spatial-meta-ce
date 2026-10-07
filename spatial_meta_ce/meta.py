from collections import OrderedDict

import torch
from torch.func import functional_call
from torch.nn import functional as F


def loss_value(pred, label, loss='smooth_l1', weights=None):
    if pred.shape != label.shape:
        raise ValueError('Prediction/label shape mismatch')
    if loss not in ('smooth_l1', 'mse'):
        raise ValueError('Unknown loss')
    if weights is None:
        value = F.smooth_l1_loss(pred, label) if loss == 'smooth_l1' else F.mse_loss(pred, label)
    else:
        if loss != 'smooth_l1':
            raise ValueError('Weighted loss requires smooth_l1')
        if not isinstance(weights, torch.Tensor) or weights.shape != pred.shape:
            raise ValueError('Weights must be a tensor matching prediction shape')
        if weights.numel() == 0 or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError('Weights must be finite and positive')
        losses = F.smooth_l1_loss(pred, label, reduction='none', beta=1)
        value = (weights * losses).sum() / weights.sum()
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


def adapt_trajectory(model, support, lr, record_steps, scope='all', loss='smooth_l1'):
    """Evaluation-only adaptation that returns detached params after each step in
    record_steps. Identical update rule to adapt(training=False), so one pass covers
    every step count of a search grid."""
    record_steps = sorted({int(s) for s in record_steps})
    if not record_steps or record_steps[0] < 0 or lr <= 0 or scope not in ('all', 'head'):
        raise ValueError('Invalid adaptation configuration')
    params = OrderedDict(model.named_parameters())
    out = {}
    if 0 in record_steps:
        out[0] = params
    names = [n for n in params if scope == 'all' or n.startswith('head.')]
    if not names:
        raise ValueError('No adaptation parameters selected')
    with torch.enable_grad():
        for step in range(1, record_steps[-1] + 1):
            objective = loss_value(forward(model, params, support), support['label'], loss)
            grads = torch.autograd.grad(objective, [params[n] for n in names], create_graph=False)
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError('Nonfinite inner gradient')
            updated = dict(zip(names, grads))
            params = OrderedDict((n, (p - lr * updated[n] if n in updated else p).detach().requires_grad_(True))
                                 for n, p in params.items())
            if step in record_steps:
                out[step] = params
    return out


def meta_loss(model, support, query, lr, steps, scope='all', loss='smooth_l1'):
    params = adapt(model, support, lr, steps, scope, loss)
    return loss_value(forward(model, params, query), query['label'], loss)
