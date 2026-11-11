"""CPU behavior baselines for existing Meta65 all/head three-step FOMAML.

No new implementation and no expected RED: exercise the existing core directly.
Reuse test_arch.sample and test_protocol.fixtures with the real sum64 estimator.
The integration baseline is two outer steps per arm, never the 4000-step run.
Synthetic a/v/t fixture names are not experimental source/target evidence.
Zero-initialized final weights need a signal before asserting encoder gradients.
"""

import contextlib
import copy
import io
import os
import random
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_arch import sample
from test_protocol import fixtures

from scripts.mapping_smoke import state_hash
from spatial_meta_ce.checkpoints import load_checkpoint
from spatial_meta_ce.data import batch_samples
from spatial_meta_ce.meta import adapt, adapt_trajectory, forward, loss_value, meta_loss
from spatial_meta_ce.model import SetEstimator
from spatial_meta_ce.train import episode_indices, train

MODEL_CONFIG = {"width": 64, "arch": "sum", "use_pg": True, "layers": 2, "heads": 4}
SCOPES = ("head", "all")
INNER_STEPS = 3
INNER_LR = 0.001


def signaled_model() -> SetEstimator:
    """Create a separate model with nonzero final weights, not a trained claim."""
    model = SetEstimator(**MODEL_CONFIG)
    original = model.state_dict()
    final_weight = f"head.{len(model.head) - 1}.weight"
    state = {
        name: torch.full_like(value, 0.01) if name == final_weight else value.clone()
        for name, value in original.items()
    }
    model.load_state_dict(state)
    return model


def snapshot(model: SetEstimator) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class Meta65ScopeGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        torch.set_num_threads(1)
        rng_context = torch.random.fork_rng(devices=[])
        rng_context.__enter__()
        self.addCleanup(rng_context.__exit__, None, None, None)
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        self.addCleanup(np.random.set_state, numpy_state)
        self.addCleanup(random.setstate, python_state)
        torch.manual_seed(42)
        self.support = batch_samples([sample(2, y=200.0), sample(4, y=800.0)], "cpu")
        self.query = batch_samples([sample(3, y=300.0), sample(5, y=900.0)], "cpu")

    def assert_parameters_equal(self, actual: dict, expected: dict) -> None:
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(
                actual[name], expected[name], rtol=0, atol=0, msg=name
            )

    def assert_encoder_gradient(self, model: SetEstimator) -> None:
        gradients = [parameter.grad for parameter in model.encoders.parameters()]
        self.assertTrue(gradients)
        self.assertTrue(
            all(g is not None and bool(torch.isfinite(g).all()) for g in gradients)
        )
        self.assertTrue(any(bool(torch.count_nonzero(g)) for g in gradients))
        # Every group receives signal, not just one fortuitous encoder parameter.
        for encoder in model.encoders:
            self.assertTrue(
                any(
                    parameter.grad is not None
                    and bool(torch.count_nonzero(parameter.grad))
                    for parameter in encoder.parameters()
                )
            )

    def test_head_three_inner_steps_leave_encoders_unchanged_but_update_head(
        self,
    ) -> None:
        model = signaled_model()
        before = snapshot(model)
        params = adapt(
            model, self.support, INNER_LR, INNER_STEPS, "head", training=True
        )
        encoder_names = [name for name in before if name.startswith("encoders.")]
        self.assertTrue(encoder_names)
        for name in encoder_names:
            torch.testing.assert_close(params[name], before[name], rtol=0, atol=0)
        self.assertTrue(
            any(
                not torch.equal(params[name], before[name])
                for name in before
                if name.startswith("head.")
            )
        )
        self.assert_parameters_equal(model.state_dict(), before)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_all_three_inner_steps_update_encoders_without_changing_original(
        self,
    ) -> None:
        model = signaled_model()
        before = snapshot(model)
        params = adapt(model, self.support, INNER_LR, INNER_STEPS, "all", training=True)
        for group in range(len(model.encoders)):
            self.assertTrue(
                any(
                    not torch.equal(params[name], before[name])
                    for name in before
                    if name.startswith(f"encoders.{group}.")
                )
            )
        self.assert_parameters_equal(model.state_dict(), before)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_both_scopes_keep_finite_nonzero_outer_encoder_gradients(self) -> None:
        initial = signaled_model()
        initial_state = snapshot(initial)
        initial_digest = state_hash(initial)
        for scope in SCOPES:
            with self.subTest(scope=scope):
                model = SetEstimator(**MODEL_CONFIG)
                model.load_state_dict(initial_state)
                self.assertEqual(state_hash(model), initial_digest)
                objective = meta_loss(
                    model, self.support, self.query, INNER_LR, INNER_STEPS, scope
                )
                self.assertTrue(bool(torch.isfinite(objective)))
                objective.backward()
                self.assert_encoder_gradient(model)
                self.assert_parameters_equal(model.state_dict(), initial_state)

    def test_zero_final_layer_gets_encoder_signal_after_one_outer_update(self) -> None:
        initial = SetEstimator(**MODEL_CONFIG)
        initial_state = snapshot(initial)
        self.assertEqual(torch.count_nonzero(initial.head[-1].weight).item(), 0)
        for scope in SCOPES:
            with self.subTest(scope=scope):
                model = SetEstimator(**MODEL_CONFIG)
                model.load_state_dict(initial_state)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
                # Do not require nonzero encoder gradients at zero-init step one.
                first_loss = meta_loss(
                    model, self.support, self.query, INNER_LR, INNER_STEPS, scope
                )
                first_loss.backward()
                optimizer.step()
                self.assertGreater(torch.count_nonzero(model.head[-1].weight).item(), 0)
                optimizer.zero_grad(set_to_none=True)
                after_update = snapshot(model)
                second_loss = meta_loss(
                    model, self.support, self.query, INNER_LR, INNER_STEPS, scope
                )
                second_loss.backward()
                self.assert_encoder_gradient(model)
                self.assert_parameters_equal(model.state_dict(), after_update)

    def test_three_step_result_matches_explicit_fomaml_updates_not_one_or_two(
        self,
    ) -> None:
        for scope in SCOPES:
            with self.subTest(scope=scope):
                model = signaled_model()
                before = snapshot(model)
                expected = OrderedDict(model.named_parameters())
                names = [
                    name
                    for name in expected
                    if scope == "all" or name.startswith("head.")
                ]
                for _ in range(INNER_STEPS):
                    loss = loss_value(
                        forward(model, expected, self.support), self.support["label"]
                    )
                    grads = torch.autograd.grad(
                        loss, [expected[name] for name in names]
                    )
                    updates = dict(zip(names, grads))
                    expected = OrderedDict(
                        (
                            name,
                            value - INNER_LR * updates[name].detach()
                            if name in updates
                            else value,
                        )
                        for name, value in expected.items()
                    )
                actual = adapt(model, self.support, INNER_LR, INNER_STEPS, scope)
                self.assert_parameters_equal(actual, expected)
                prediction = forward(model, actual, self.query)
                for shorter in (1, 2):
                    shorter_params = adapt(
                        model, self.support, INNER_LR, shorter, scope
                    )
                    self.assertFalse(
                        torch.equal(
                            prediction, forward(model, shorter_params, self.query)
                        )
                    )
                self.assert_parameters_equal(model.state_dict(), before)

    def test_training_and_detached_trajectories_agree_at_zero_one_two_three(
        self,
    ) -> None:
        for scope in SCOPES:
            with self.subTest(scope=scope):
                model = signaled_model()
                trajectory = adapt_trajectory(
                    model, self.support, INNER_LR, [0, 1, 2, 3], scope
                )
                self.assertEqual(set(trajectory), {0, 1, 2, 3})
                for steps in range(4):
                    trained = adapt(
                        model, self.support, INNER_LR, steps, scope, training=True
                    )
                    detached = adapt(
                        model, self.support, INNER_LR, steps, scope, training=False
                    )
                    self.assert_parameters_equal(trained, detached)
                    self.assert_parameters_equal(detached, trajectory[steps])

    def test_independent_episodes_reset_and_none_keeps_initial_predictions(
        self,
    ) -> None:
        alternate = {
            **self.support,
            # Opposite residual signs avoid identical saturated SmoothL1 gradients.
            "label": torch.zeros_like(self.support["label"]),
            "true": torch.ones_like(self.support["true"]),
        }
        for scope in SCOPES:
            with self.subTest(scope=scope):
                model = signaled_model()
                before = snapshot(model)
                none_before = model(self.query).detach().clone()
                episode_a = adapt(model, self.support, INNER_LR, INNER_STEPS, scope)
                prediction_a = forward(model, episode_a, self.query).detach().clone()
                episode_b = adapt(model, alternate, INNER_LR, INNER_STEPS, scope)
                self.assertFalse(
                    torch.equal(prediction_a, forward(model, episode_b, self.query))
                )
                repeated_a = adapt(model, self.support, INNER_LR, INNER_STEPS, scope)
                self.assert_parameters_equal(repeated_a, episode_a)
                torch.testing.assert_close(
                    forward(model, repeated_a, self.query), prediction_a, rtol=0, atol=0
                )
                none_params = adapt(model, None, INNER_LR, 0, scope)
                torch.testing.assert_close(
                    forward(model, none_params, self.query), none_before, rtol=0, atol=0
                )
                self.assert_parameters_equal(model.state_dict(), before)

    def test_adaptation_and_outer_gradient_leave_batch_inputs_unchanged(self) -> None:
        support_before = copy.deepcopy(self.support)
        query_before = copy.deepcopy(self.query)
        for scope in SCOPES:
            model = signaled_model()
            meta_loss(
                model, self.support, self.query, INNER_LR, INNER_STEPS, scope
            ).backward()
        for actual, expected in (
            (self.support, support_before),
            (self.query, query_before),
        ):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                if isinstance(actual[key], list):
                    self.assertEqual(len(actual[key]), len(expected[key]))
                    for left, right in zip(actual[key], expected[key]):
                        if isinstance(left, torch.Tensor):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                        else:
                            self.assertEqual(left, right)
                elif isinstance(actual[key], torch.Tensor):
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0
                    )
                else:
                    self.assertEqual(actual[key], expected[key])

    def test_two_cpu_outer_steps_pair_initial_hash_task_flow_and_counts(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"]) as directory:
            root = Path(directory)
            base, data = fixtures(root)
            config = {
                **base,
                "model": dict(MODEL_CONFIG),
                "seed": 42,
                "steps": 2,
                "inner_steps": INNER_STEPS,
                "inner_lr": INNER_LR,
                "outer_lr": 0.0003,
                "purpose": "synthetic two-step behavior baseline; not experimental evidence",
            }
            records = []
            for scope in SCOPES:
                arm_config = {**config, "adapt_scope": scope}
                with (
                    patch(
                        "spatial_meta_ce.train.create_model", wraps=lambda config: SetEstimator(**config)
                    ) as constructor,
                    patch(
                        "spatial_meta_ce.train.episode_indices", wraps=episode_indices
                    ) as sampler,
                    patch(
                        "spatial_meta_ce.train.meta_loss", wraps=meta_loss
                    ) as objective,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    # Capture the actual training model before its first update.
                    initial_hashes = []
                    task_flow = []

                    def build_model(config: dict) -> SetEstimator:
                        model = SetEstimator(**config)
                        initial_hashes.append(state_hash(model))
                        return model

                    def sample_episode(*args: object, **kwargs: object) -> tuple:
                        support, query = episode_indices(*args, **kwargs)
                        task_flow.append((support.tolist(), query.tolist()))
                        return support, query

                    constructor.side_effect = build_model
                    sampler.side_effect = sample_episode
                    output = root / scope
                    result = train(arm_config, data, output, device="cpu")
                model, checkpoint = load_checkpoint(output / "last.pt", "cpu")
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(len(initial_hashes), 1)
                self.assertEqual(result["last_step"], 2)
                self.assertEqual(checkpoint["next_step"], 2)
                self.assertEqual(result["parameters"], 132545)
                self.assertEqual(objective.call_count, 2)
                for call in objective.call_args_list:
                    self.assertEqual(call.args[3:6], (INNER_LR, INNER_STEPS, scope))
                counts = {
                    "support_presentations": 30,
                    "query_presentations": 8,
                    "backward_calls": 8,
                    "optimizer_steps": 2,
                }
                self.assertEqual(result["counts"], counts)
                self.assertEqual(checkpoint["extra"]["counts"], counts)
                self.assertEqual(checkpoint["extra"]["sample_plan_episodes"], 2)
                self.assertNotEqual(state_hash(model), initial_hashes[0])
                self.assertTrue(
                    all(bool(torch.isfinite(p).all()) for p in model.parameters())
                )
                records.append(
                    {
                        "initial_hash": initial_hashes[0],
                        "task_flow": task_flow,
                        "sample_plan_digest": checkpoint["extra"]["sample_plan_digest"],
                        "counts": counts,
                        "validation_plan": (
                            output / "validation_manifest.json"
                        ).read_bytes(),
                    }
                )
            # One validation-plan episode precedes the two training episodes.
            self.assertEqual(len(records[0]["task_flow"]), 3)
            for support, query in records[0]["task_flow"]:
                self.assertEqual(len(support), 5)
                self.assertEqual(len(query), 4)
                self.assertFalse(set(support) & set(query))
            self.assertEqual(records[0], records[1])


if __name__ == "__main__":
    unittest.main()
