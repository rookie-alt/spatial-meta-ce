import unittest

import torch

from spatial_meta_ce.meta import loss_value


class TailAlignedLossTests(unittest.TestCase):
    """P1-1: losses aligned with the log-Q95 selection metric (doc 93 §3)."""

    def test_log_l1_is_mean_absolute_log_error(self):
        pred = torch.tensor([0.0, 1.0, -1.0], requires_grad=True)
        label = torch.tensor([0.0, 3.0, 1.0])
        expected = (0.0 + 2.0 + 2.0) / 3
        value = loss_value(pred, label, 'log_l1')
        self.assertAlmostEqual(value.item(), expected, places=6)
        value.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([0.0, -1 / 3, -1 / 3]))

    def test_log_l1_matches_identity_at_zero_residual(self):
        pred = torch.zeros(4)
        self.assertAlmostEqual(loss_value(pred, pred, 'log_l1').item(), 0.0, places=7)

    def test_pinball_tau_is_asymmetric(self):
        # Residual label - pred = +2 (under-prediction) is penalised by tau;
        # residual -2 (over-prediction) by (1 - tau). P1-1 registers tau in (0, 1).
        over = loss_value(torch.tensor([2.0]), torch.tensor([0.0]), 'pinball_tau', tau=0.9)
        under = loss_value(torch.tensor([0.0]), torch.tensor([2.0]), 'pinball_tau', tau=0.9)
        self.assertAlmostEqual(over.item(), 0.1 * 2.0, places=6)
        self.assertAlmostEqual(under.item(), 0.9 * 2.0, places=6)

    def test_pinball_tau_half_is_half_mae(self):
        pred = torch.tensor([0.5, -2.0])
        label = torch.zeros(2)
        value = loss_value(pred, label, 'pinball_tau', tau=0.5)
        self.assertAlmostEqual(value.item(), 0.5 * (0.5 + 2.0) / 2, places=6)

    def test_pinball_tau_requires_registered_tau(self):
        pred = torch.zeros(2)
        for tau in (None, 0.0, 1.0, -0.1, 1.5, True, '0.9', float('nan')):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                loss_value(pred, pred, 'pinball_tau', tau=tau)

    def test_unknown_loss_and_shape_mismatch_still_rejected(self):
        with self.assertRaises(ValueError):
            loss_value(torch.zeros(2), torch.zeros(2), 'huber')
        with self.assertRaises(ValueError):
            loss_value(torch.zeros(2), torch.zeros(3), 'log_l1')

    def test_nonfinite_terms_raise(self):
        for loss in ('log_l1', 'pinball_tau'):
            with self.subTest(loss=loss), self.assertRaises(FloatingPointError):
                loss_value(torch.tensor([float('nan'), 0.0]), torch.zeros(2), loss, tau=0.9)

    def test_legacy_losses_unchanged(self):
        pred = torch.tensor([0.3, -1.2])
        label = torch.tensor([0.0, 0.5])
        smooth = loss_value(pred, label, 'smooth_l1')
        mse = loss_value(pred, label, 'mse')
        torch.testing.assert_close(smooth, torch.nn.functional.smooth_l1_loss(pred, label))
        torch.testing.assert_close(mse, torch.nn.functional.mse_loss(pred, label))


if __name__ == '__main__':
    unittest.main()
