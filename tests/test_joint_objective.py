import unittest

import torch

from spatial_meta_ce.meta import adapt, meta_loss, meta_loss_terms


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, batch):
        return self.scale * batch["x"] + self.bias


def batch():
    return {
        "x": torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32),
        "label": torch.tensor([0.2, 0.8, 1.4], dtype=torch.float32),
    }


def gradients(objective, term=None, weight=1.0):
    model = TinyModel()
    terms = meta_loss_terms(
        model, batch(), batch(), lr=0.1, steps=2, loss="mse", objective=objective
    )
    loss = terms["total"] if term is None else terms[term] * weight
    loss.backward()
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


class JointObjectiveTests(unittest.TestCase):
    def test_joint_total_is_half_pre_plus_half_post(self):
        terms = meta_loss_terms(
            TinyModel(),
            batch(),
            batch(),
            lr=0.1,
            steps=2,
            loss="mse",
            objective="joint",
        )
        self.assertAlmostEqual(
            terms["total"].item(),
            0.5 * terms["pre"].item() + 0.5 * terms["post"].item(),
            places=6,
        )

    def test_joint_gradient_is_the_sum_of_the_two_first_order_paths(self):
        joint = gradients("joint")
        pre = gradients("joint", "pre", 0.5)
        post = gradients("joint", "post", 0.5)
        for name in joint:
            self.assertTrue(
                torch.allclose(joint[name], pre[name] + post[name], atol=1e-6),
                msg=name,
            )

    def test_both_joint_terms_change_the_common_parameters(self):
        for term in ("pre", "post"):
            grads = gradients("joint", term)
            self.assertTrue(
                any(value.abs().sum() > 0 for value in grads.values()), msg=term
            )

    def test_post_objective_keeps_pre_as_a_detached_diagnostic(self):
        terms = meta_loss_terms(
            TinyModel(),
            batch(),
            batch(),
            lr=0.1,
            steps=2,
            loss="mse",
            objective="post",
        )
        self.assertFalse(terms["pre"].requires_grad)
        self.assertIs(terms["total"], terms["post"])

    def test_post_total_matches_the_legacy_meta_loss(self):
        model = TinyModel()
        support, query = batch(), batch()
        terms = meta_loss_terms(
            model,
            support,
            query,
            lr=0.1,
            steps=2,
            loss="mse",
            objective="post",
        )
        legacy = meta_loss(model, support, query, 0.1, 2, loss="mse")
        self.assertTrue(torch.equal(terms["total"], legacy))

    def test_adapt_does_not_mutate_the_common_model(self):
        model = TinyModel()
        before = {
            name: value.detach().clone() for name, value in model.named_parameters()
        }
        adapt(model, batch(), lr=0.1, steps=2, loss="mse")
        for name, value in model.named_parameters():
            self.assertTrue(torch.equal(value, before[name]), msg=name)

    def test_unknown_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            meta_loss_terms(
                TinyModel(),
                batch(),
                batch(),
                lr=0.1,
                steps=2,
                loss="mse",
                objective="pre_only",
            )


if __name__ == "__main__":
    unittest.main()
