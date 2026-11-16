"""PRICE-native outputs and first-order functional adaptation for experiment 123."""

from collections import OrderedDict
from contextlib import contextmanager, redirect_stdout
import io
import json
import math
from pathlib import Path
import re

import torch
from torch import nn
from torch.func import functional_call

from .baselines import PriceAdapter
from .manifests import file_hash

CAPACITY = (12, 12, 7, 10)
INITIAL_SHA = "dee83c462944db8c5901ac32da9c844c61df6175dcb04bc50923f96babff070f"
ENCODER_SHA = "94f3060996cd9a9b3d60da08d525cc4ab8e1666c1eb4d53b3f610f5c248c6bce"
MODULE_SHA = "15d98eb2f1aed939dc53002302016098394e0602719e0d56df6c351a29f4f66e"


def verify_vendor():
    root = Path(__file__).resolve().parents[1] / "vendor"
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    for filename, expected in (("encoder.py", ENCODER_SHA), ("module.py", MODULE_SHA)):
        rel = "price_baseline/" + filename
        if (
            file_hash(root / rel) != expected
            or provenance["files"][rel]["adapted_sha256"] != expected
        ):
            raise ValueError("Frozen PRICE vendor SHA changed")


def _finite(value, what):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {what}")


class NativePriceModel(nn.Module):
    def __init__(self, counts=CAPACITY):
        super().__init__()
        if tuple(counts) != CAPACITY:
            raise ValueError("Experiment 123 requires the frozen PRICE capacity")
        verify_vendor()
        from vendor.price_baseline.encoder import RegressionModel

        self.counts = CAPACITY
        with redirect_stdout(io.StringIO()):
            self.price = RegressionModel(
                *CAPACITY, 40, 4, 43, 512, 1024, 1, 256, 6, 8, 0.2
            )

    def forward(self, batch):
        for group, mask, capacity in zip(batch["groups"], batch["masks"], CAPACITY):
            _finite(group, "PRICE input")
            if (mask.sum(1) > capacity).any():
                raise ValueError("PRICE padding capacity too small")
        _finite(batch["pg_log"], "PG input")
        _finite(batch["counts"], "token counts")
        output = PriceAdapter.forward(self, batch)
        _finite(output, "native PRICE prediction")
        return output


def native_target(batch):
    truth = batch["true"]
    _finite(truth, "true cardinality")
    if (truth < 0).any():
        raise ValueError("True cardinality must be nonnegative")
    return (torch.log1p(truth) + 1).to(torch.float32).flatten()


def build_model(checkpoint, device="cpu", expected_sha=INITIAL_SHA):
    if not isinstance(expected_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha
    ):
        raise ValueError("Expected checkpoint SHA must be a complete SHA256")
    if checkpoint is None or file_hash(checkpoint) != expected_sha:
        raise ValueError("PRICE checkpoint SHA mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        not isinstance(state, dict)
        or not state
        or any(not torch.is_tensor(v) for v in state.values())
    ):
        raise ValueError("PRICE checkpoint must be a plain state_dict")
    for prefix in ("module.", "price."):
        if all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    for value in state.values():
        _finite(value, "checkpoint weight")
    model = NativePriceModel()
    model.price.load_state_dict(state, strict=True)
    return model.to(device).eval()


@contextmanager
def _mode(model, training):
    previous = [(module, module.training) for module in model.modules()]
    model.train(training)
    try:
        yield
    finally:
        for module, enabled in previous:
            module.training = enabled


def predict_native(model, batch, params=None):
    result = (
        model(batch)
        if params is None
        else functional_call(model, params, (batch,), strict=True)
    )
    _finite(result, "native prediction")
    if result.ndim != 1 or result.shape != batch["true"].flatten().shape:
        raise ValueError("Native prediction/true cardinality shape mismatch")
    return result


def _loss(model, batch, params=None):
    target = native_target(batch)
    prediction = predict_native(model, batch, params)
    value = torch.nn.functional.mse_loss(prediction, target)
    _finite(value, "MSE loss")
    return value


def adapt_native(model, support, alpha, steps=5, meta=False):
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
        or type(steps) is not int
        or steps < 0
    ):
        raise ValueError("Invalid native adaptation configuration")
    params = OrderedDict(model.named_parameters())
    audit = {
        "losses": [],
        "gradient_norms": [],
        "steps": steps,
        "audit_extra_forward_count": 1,
        "support_size": len(support["true"]),
    }
    with torch.enable_grad(), _mode(model, False):
        for _ in range(steps):
            loss = _loss(model, support, params)
            grads = torch.autograd.grad(
                loss, tuple(params.values()), create_graph=False
            )
            for grad in grads:
                _finite(grad, "inner gradient")
            norm = torch.sqrt(
                sum(grad.detach().double().square().sum() for grad in grads)
            )
            audit["losses"].append(float(loss.detach()))
            audit["gradient_norms"].append(float(norm))
            updated = OrderedDict()
            for (name, parameter), grad in zip(params.items(), grads):
                value = parameter - alpha * grad
                _finite(value, "fast weight")
                updated[name] = value if meta else value.detach().requires_grad_(True)
            params = updated
        with torch.no_grad():
            final_loss = float(_loss(model, support, params))
    audit["support_mse_before"] = audit["losses"][0] if steps else final_loss
    audit["support_mse_after"] = final_loss
    return params, audit


def erm_loss(model, union_batch):
    with _mode(model, True):
        return _loss(model, union_batch)


def fomaml_loss(model, support, query, alpha):
    params, audit = adapt_native(model, support, alpha, meta=True)
    with _mode(model, True):
        result = _loss(model, query, params)
    return result, audit
