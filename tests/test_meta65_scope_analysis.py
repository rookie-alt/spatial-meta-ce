"""Synthetic regression cases for descriptive scope contrasts, not experiments."""

import unittest

from scripts.meta65_scope_analysis import summarize_runs


def rows(ratios: tuple = (1.2, 1.0, 1.0, 1.0)) -> list[dict]:
    names = ("low4", "low13", "nonlow4", "nonlow13")
    return [
        {
            "run": run,
            "buckets": {
                name: {"count": 40, "none_q95": 10.0, "head_q95": 10.0 * ratio}
                for name, ratio in zip(names, ratios)
            },
            "mean_delta_logq": 0.1,
        }
        for run in range(5)
    ]


class ScopeAnalysisTests(unittest.TestCase):
    def test_stable_and_interaction_require_full_contract(self) -> None:
        result = summarize_runs(rows())
        self.assertTrue(result["stable_aggravation_descriptive"])
        self.assertTrue(result["positive_interaction_descriptive"])

    def test_joint_stronger_cannot_be_omitted_from_interaction(self) -> None:
        result = summarize_runs(rows((1.2, 1.3, 0.5, 1.0)))
        self.assertEqual(result["positive_interaction_runs"], 5)
        self.assertEqual(result["joint_stronger_runs"], 0)
        self.assertFalse(result["positive_interaction_descriptive"])

    def test_exact_ten_percent_not_stable_aggravation(self) -> None:
        self.assertFalse(
            summarize_runs(rows((1.1, 1.0, 1.0, 1.0)))["stable_aggravation_descriptive"]
        )

    def test_too_few_samples_or_empty_buckets_are_unknown(self) -> None:
        base = rows()
        missing = [
            {
                **r,
                "buckets": {
                    **r["buckets"],
                    "low13": {"count": 0, "none_q95": None, "head_q95": None},
                },
            }
            for r in base
        ]
        result = summarize_runs(missing)
        self.assertIsNone(result["positive_interaction_descriptive"])
        self.assertTrue(result["stable_aggravation_descriptive"])
        small = [
            {
                **r,
                "buckets": {
                    **r["buckets"],
                    "low4": {"count": 29, "none_q95": 10.0, "head_q95": 12.0},
                },
            }
            for r in base
        ]
        self.assertIsNone(summarize_runs(small)["stable_aggravation_descriptive"])

    def test_missing_runs_and_nonfinite_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            summarize_runs(rows()[:-1])
        bad = [{**r, "mean_delta_logq": float("nan")} for r in rows()]
        with self.assertRaises(ValueError):
            summarize_runs(bad)

    def test_negative_direction_is_not_stable_harm(self) -> None:
        result = summarize_runs(
            [{**r, "mean_delta_logq": -0.1} for r in rows((0.8, 1.0, 1.0, 1.0))]
        )
        self.assertFalse(result["stable_aggravation_descriptive"])
        self.assertFalse(result["positive_interaction_descriptive"])


if __name__ == "__main__":
    unittest.main()
