from collections import OrderedDict

import torch
from torch import nn


def _positive_finite(value, name):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not torch.isfinite(torch.tensor(float(value))) or value <= 0):
        raise ValueError(f'{name} must be a finite positive number')
    return float(value)


def ridge_last(model, support, weight_decay=1.0, bias_decay=None,
               training=True, return_audit=False):
    """Solve the final linear layer with a proximal ridge objective.

    The solver preserves the existing last-layer parameter family. It shrinks
    toward the checkpoint's final-layer weights instead of toward zero.
    """
    weight_decay = _positive_finite(weight_decay, 'weight_decay')
    bias_decay = (weight_decay if bias_decay is None
                  else _positive_finite(bias_decay, 'bias_decay'))
    head = getattr(model, 'head', None)
    if (not isinstance(head, nn.Sequential) or not len(head)
            or not isinstance(head[-1], nn.Linear)
            or head[-1].bias is None or head[-1].out_features != 1):
        raise ValueError(
            'ridge_last requires a Sequential head ending in scalar Linear with bias')
    features = model.head_features(support)
    anchor = model.output_anchor(support)
    target = support['label'] - anchor
    if features.ndim != 2 or target.ndim != 1:
        raise ValueError('Support features/target must be two-/one-dimensional')
    if features.shape[0] != target.shape[0] or features.shape[0] < 1:
        raise ValueError('Support feature/target sizes disagree')
    if features.shape[1] != head[-1].in_features:
        raise ValueError('Support feature width does not match the final linear')
    if not torch.isfinite(features).all() or not torch.isfinite(target).all():
        raise FloatingPointError('Nonfinite support features or residual target')

    if training:
        solve_features = features.to(torch.float64)
        solve_target = target.to(torch.float64)
        prior_weight = head[-1].weight.reshape(-1).to(torch.float64)
        prior_bias = head[-1].bias.reshape(-1)[0].to(torch.float64)
    else:
        solve_features = features.detach().to(torch.float64)
        solve_target = target.detach().to(torch.float64)
        prior_weight = head[-1].weight.detach().reshape(-1).to(torch.float64)
        prior_bias = head[-1].bias.detach().reshape(-1)[0].to(torch.float64)

    ones = torch.ones(solve_features.shape[0], 1, dtype=torch.float64,
                      device=solve_features.device)
    design = torch.cat([solve_features, ones], dim=1)
    penalty = torch.cat([
        torch.full((solve_features.shape[1],), weight_decay, dtype=torch.float64,
                   device=solve_features.device),
        torch.tensor([bias_decay], dtype=torch.float64, device=solve_features.device),
    ])
    prior = torch.cat([prior_weight, prior_bias.reshape(1)])
    gram = design.T @ design + torch.diag(penalty)
    rhs = design.T @ solve_target + penalty * prior
    try:
        solution = torch.linalg.solve(gram, rhs)
    except RuntimeError as exc:
        raise ValueError('Proximal ridge solve failed') from exc
    if not torch.isfinite(solution).all():
        raise FloatingPointError('Nonfinite proximal ridge solution')
    condition_number = torch.linalg.cond(gram)
    if not torch.isfinite(condition_number):
        raise FloatingPointError('Nonfinite proximal ridge condition number')

    new_weight = solution[:-1].reshape_as(head[-1].weight).to(
        device=head[-1].weight.device, dtype=head[-1].weight.dtype)
    new_bias = solution[-1:].reshape_as(head[-1].bias).to(
        device=head[-1].bias.device, dtype=head[-1].bias.dtype)
    params = OrderedDict(model.named_parameters())
    weight_name = next(name for name, value in params.items()
                       if value is head[-1].weight)
    bias_name = next(name for name, value in params.items()
                     if value is head[-1].bias)
    params[weight_name] = new_weight
    params[bias_name] = new_bias
    if not training:
        params = OrderedDict((name, value.detach()) for name, value in params.items())

    if not return_audit:
        return params
    with torch.no_grad():
        base = solve_features @ prior_weight + prior_bias
        fitted = solve_features @ solution[:-1] + solution[-1]
        audit = {
            'method': 'proximal_ridge_last',
            'support_count': int(solve_features.shape[0]),
            'feature_dim': int(solve_features.shape[1]),
            'weight_decay': weight_decay,
            'bias_decay': bias_decay,
            'support_mse_before': float(torch.mean((base - solve_target).square())),
            'support_mse_after': float(torch.mean((fitted - solve_target).square())),
            'update_norm': float(torch.linalg.vector_norm(solution - prior)),
            'condition_number': float(condition_number),
            'changed_parameters': [weight_name, bias_name],
            'training': bool(training),
            'finite': True,
        }
    return params, audit
