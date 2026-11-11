from collections import OrderedDict
from contextlib import contextmanager

import torch
from torch.func import functional_call
from torch.nn import functional as F


LOSSES = ("smooth_l1", "mse", "log_l1", "pinball_tau")
JOINT_PRE_WEIGHT = 0.5
OBJECTIVES = ("post", "joint")


def _check_tau(tau):
    """Pinball tau is pre-registered (doc 93 §3 P1-1); no on-the-fly grid."""
    if isinstance(tau, bool) or not isinstance(tau, (int, float)):
        raise ValueError("pinball_tau requires a numeric tau in (0, 1)")
    tau = float(tau)
    if not 0.0 < tau < 1.0:
        raise ValueError("pinball_tau requires a numeric tau in (0, 1)")
    return tau


def tail_aligned_loss(pred, label, loss, tau=None):
    """Log-domain losses that track the log-Q95 selection metric.

    Predictions and labels are already log-cardinalities, so `log_l1` is the
    mean absolute log error and `pinball_tau` its asymmetric quantile version.
    Neither thresholds nor clamps the output; the residual form is untouched.
    """
    if loss == "log_l1":
        return (pred - label).abs().mean()
    tau = _check_tau(tau)
    residual = label - pred
    return torch.maximum(tau * residual, (tau - 1.0) * residual).mean()


def loss_value(pred, label, loss="smooth_l1", weights=None, tau=None):
    if pred.shape != label.shape:
        raise ValueError("Prediction/label shape mismatch")
    if loss not in LOSSES:
        raise ValueError("Unknown loss")
    if loss in ("log_l1", "pinball_tau"):
        if weights is not None:
            raise ValueError("Weighted loss requires smooth_l1")
        value = tail_aligned_loss(pred, label, loss, tau)
    elif weights is None:
        value = (
            F.smooth_l1_loss(pred, label)
            if loss == "smooth_l1"
            else F.mse_loss(pred, label)
        )
    else:
        if loss != "smooth_l1":
            raise ValueError("Weighted loss requires smooth_l1")
        if not isinstance(weights, torch.Tensor) or weights.shape != pred.shape:
            raise ValueError("Weights must be a tensor matching prediction shape")
        if (
            weights.numel() == 0
            or not torch.isfinite(weights).all()
            or not (weights > 0).all()
        ):
            raise ValueError("Weights must be finite and positive")
        losses = F.smooth_l1_loss(pred, label, reduction="none", beta=1)
        value = (weights * losses).sum() / weights.sum()
    if not torch.isfinite(value):
        raise FloatingPointError("Nonfinite loss; refusing to silently skip task")
    return value


def forward(model, params, batch):
    return functional_call(model, params, (batch,), strict=True)


@contextmanager
def adaptation_mode(model, training):
    modes = [(module, module.training) for module in model.modules()]
    model.train(training)
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def _adaptation_names(model, params, scope):
    if scope not in ("all", "full", "head", "last", "lora"):
        raise ValueError("Invalid adaptation scope")
    if scope == "lora":
        names = [n for n in params if n.endswith((".lora_A", ".lora_B"))]
        if not names:
            raise ValueError("LoRA scope requires registered lora parameters")
    elif scope == "last":
        head = getattr(model, "head", None)
        if (
            not isinstance(head, torch.nn.Sequential)
            or not len(head)
            or not isinstance(head[-1], torch.nn.Linear)
        ):
            raise ValueError(
                "Last-layer adaptation requires a Sequential head ending in Linear"
            )
        prefix = f"head.{len(head) - 1}."
        names = [n for n in params if n.startswith(prefix)]
    else:
        names = [n for n in params if scope in ("all", "full") or n.startswith("head.")]
    if not names:
        raise ValueError("No adaptation parameters selected")
    return names


def adapt(
    model,
    support,
    lr=0.001,
    steps=5,
    scope="all",
    loss="smooth_l1",
    training=True,
    tau=None,
):
    if steps < 0 or lr <= 0 or scope not in ("all", "full", "head", "last", "lora"):
        raise ValueError("Invalid adaptation configuration")
    params = OrderedDict(model.named_parameters())
    names = _adaptation_names(model, params, scope)
    if steps == 0 or support is None:
        return params
    extra = {} if tau is None else {"tau": tau}
    with torch.enable_grad(), adaptation_mode(model, training):
        for _ in range(steps):
            objective = loss_value(
                forward(model, params, support), support["label"], loss, **extra
            )
            grads = torch.autograd.grad(
                objective, [params[n] for n in names], create_graph=False
            )
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Nonfinite inner gradient")
            updated = dict(zip(names, grads))
            params = OrderedDict(
                (n, p - lr * updated[n] if n in updated else p)
                for n, p in params.items()
            )
            if not training:
                params = OrderedDict(
                    (n, p.detach().requires_grad_(True)) for n, p in params.items()
                )
    return params


def adapt_trajectory(
    model, support, lr, record_steps, scope="all", loss="smooth_l1", tau=None
):
    """Evaluation-only adaptation that returns detached params after each step in
    record_steps. Identical update rule to adapt(training=False), so one pass covers
    every step count of a search grid."""
    record_steps = sorted({int(s) for s in record_steps})
    if (
        not record_steps
        or record_steps[0] < 0
        or lr <= 0
        or scope not in ("all", "full", "head", "last", "lora")
    ):
        raise ValueError("Invalid adaptation configuration")
    params = OrderedDict(model.named_parameters())
    out = {}
    if 0 in record_steps:
        out[0] = params
    names = _adaptation_names(model, params, scope)
    extra = {} if tau is None else {"tau": tau}
    with torch.enable_grad(), adaptation_mode(model, False):
        for step in range(1, record_steps[-1] + 1):
            objective = loss_value(
                forward(model, params, support), support["label"], loss, **extra
            )
            grads = torch.autograd.grad(
                objective, [params[n] for n in names], create_graph=False
            )
            if any(not torch.isfinite(g).all() for g in grads):
                raise FloatingPointError("Nonfinite inner gradient")
            updated = dict(zip(names, grads))
            params = OrderedDict(
                (
                    n,
                    (p - lr * updated[n] if n in updated else p)
                    .detach()
                    .requires_grad_(True),
                )
                for n, p in params.items()
            )
            if step in record_steps:
                out[step] = params
    return out


def meta_loss_terms(
    model, support, query, lr, steps, scope="all", loss="smooth_l1", tau=None,
    objective="post",
):
    with adaptation_mode(model, True):
        return _meta_loss_terms(model, support, query, lr, steps, scope, loss, tau, objective)


def _meta_loss_terms(
    model,
    support,
    query,
    lr,
    steps,
    scope="all",
    loss="smooth_l1",
    tau=None,
    objective="post",
):
    """Return pre/post/total losses for the FOMAML objective.

    ``post`` is the adapted-query loss used by the original FOMAML.  ``pre`` is
    the unadapted-query loss used by the joint objective; for post-only it is a
    detached diagnostic and does not enter the graph.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown meta objective {objective!r}")
    extra = {} if tau is None else {"tau": tau}
    base = OrderedDict(model.named_parameters())
    if objective == "joint":
        pre = loss_value(forward(model, base, query), query["label"], loss, **extra)
    else:
        with torch.no_grad():
            pre = loss_value(
                forward(model, base, query), query["label"], loss, **extra
            ).detach()
    params = adapt(model, support, lr, steps, scope, loss, tau=tau)
    post = loss_value(forward(model, params, query), query["label"], loss, **extra)
    if objective == "joint":
        total = JOINT_PRE_WEIGHT * pre + (1.0 - JOINT_PRE_WEIGHT) * post
    else:
        total = post
    return {"pre": pre, "post": post, "total": total}


def meta_loss(
    model, support, query, lr, steps, scope="all", loss="smooth_l1", tau=None
):
    with adaptation_mode(model, True):
        params = adapt(model, support, lr, steps, scope, loss, tau=tau)
        extra = {} if tau is None else {"tau": tau}
        return loss_value(forward(model, params, query), query["label"], loss, **extra)
