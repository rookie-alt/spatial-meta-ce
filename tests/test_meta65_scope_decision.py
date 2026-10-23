"""Test-first contract for the source-only Meta65 scope decision (plan 65).

Frozen normalized schema, deliberately not the raw joint-runner evidence schema:
* cells is a list of exactly 60 dictionaries with exactly the keys
  arm/operator/database/run/k/metrics; database follows existing evidence naming.
* arm is R_head3 or T_all3, operator is none or head3, database is one of
  chembl/legalacts/telstra, run is an exact int in 0..4, k is an exact int 20.
* metrics has exactly all/4+/low4, each with exactly count/q50/q95. low4 is
  the independently certified true<100 AND tables>=4 quantile, not an average
  of the <100|4 and <100|5+ quantiles. No raw truths or deployment routing.
* Counts are positive exact ints with low4<=4+<=all, paired across all four
  arm/operator cells for each database/run. Every low4 count must be >=30.
* Quantiles are finite built-in int/float (not bool), >=1, and q50<=q95.
* Verification flags are actual bools. Invalid input returns blocked, not an
  exception. A valid decision dict exposes at least status and selected_arm;
  only promising_single_seed_K20 selects the global T_all3 arm.
* Five run quantiles are averaged first, then three databases equally. Only
  head3 4+ Q95 enters the main arithmetic macro; guards compare means, not
  per-run ratios. All thresholds are inclusive and combined with AND.

All values below are synthetic unit-test fixtures, not experimental evidence.
The direct import intentionally gives ModuleNotFoundError until implementation.
"""

import copy
import json
import math
import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from scripts.meta65_scope_decision import decide

ARMS = ("R_head3", "T_all3")
OPERATORS = ("none", "head3")
DATABASES = ("chembl", "legalacts", "telstra")
BUCKETS = ("all", "4+", "low4")
PROMISING = "promising_single_seed_K20"


def make_cells(treatment_q95: float = 90.0) -> list[dict]:
    """Build independent, complete, paired cells with all gates satisfied."""
    return [
        {
            "arm": arm,
            "operator": operator,
            "database": database,
            "run": run,
            "k": 20,
            "metrics": {
                "all": {"count": 1000, "q50": 10.0, "q95": value},
                "4+": {"count": 500, "q50": 10.0, "q95": value},
                "low4": {"count": 40, "q50": 5.0, "q95": 20.0},
            },
        }
        for arm in ARMS
        for operator in OPERATORS
        for database in DATABASES
        for run in range(5)
        for value in (
            treatment_q95 if arm == "T_all3" and operator == "head3" else 100.0,
        )
    ]


def select(
    cell: dict,
    *,
    arm: str | None = None,
    operator: str | None = None,
    database: str | None = None,
    run: int | None = None,
) -> bool:
    return all(
        expected is None or cell[key] == expected
        for key, expected in (
            ("arm", arm),
            ("operator", operator),
            ("database", database),
            ("run", run),
        )
    )


def update_metric(
    cells: list[dict],
    bucket: str,
    field: str,
    value: object,
    **selector: object,
) -> list[dict]:
    """Return new fixture dictionaries without modifying the input."""
    return [
        {
            **cell,
            "metrics": {
                **cell["metrics"],
                bucket: {**cell["metrics"][bucket], field: value},
            },
        }
        if select(cell, **selector)
        else cell
        for cell in cells
    ]


def replace_first(cells: list[dict], transform: Callable[[dict], dict]) -> list[dict]:
    return [transform(cells[0]), *cells[1:]]


class Meta65ScopeDecisionTests(unittest.TestCase):
    def assert_decision(
        self,
        cells: object,
        status: str,
        *,
        reproduction_verified: object = True,
        engineering_verified: object = True,
    ) -> dict:
        result = decide(
            cells,
            reproduction_verified=reproduction_verified,
            engineering_verified=engineering_verified,
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["status"], status)
        self.assertEqual(
            result["selected_arm"], "T_all3" if status == PROMISING else None
        )
        # Reports must remain serializable; nonfinite values must not leak out.
        json.dumps(result, allow_nan=False)
        return result

    def test_complete_paired_sixty_cell_matrix_selects_one_global_arm(self) -> None:
        cells = make_cells()
        self.assertEqual(len(cells), 60)
        self.assert_decision(cells, PROMISING)

    def test_main_macro_ten_percent_boundary_is_inclusive(self) -> None:
        for q95, expected in (
            (89.999999, PROMISING),
            (90.0, PROMISING),
            (90.000001, "no_candidate"),
            (100.0, "no_candidate"),
            (110.0, "no_candidate"),
        ):
            with self.subTest(q95=q95):
                self.assert_decision(make_cells(q95), expected)

    def test_all_and_four_plus_q50_q95_guards_are_independent_and_inclusive(
        self,
    ) -> None:
        for database in DATABASES:
            for bucket in ("all", "4+"):
                for field, boundary in (("q50", 11.0), ("q95", 110.0)):
                    for value, expected in (
                        (boundary, PROMISING),
                        (boundary + 0.000001, "no_candidate"),
                    ):
                        with self.subTest(
                            db=database, bucket=bucket, field=field, value=value
                        ):
                            cells = update_metric(
                                make_cells(70.0),
                                bucket,
                                field,
                                value,
                                arm="T_all3",
                                operator="head3",
                                database=database,
                            )
                            self.assert_decision(cells, expected)

    def test_low4_reference_guard_is_per_database_and_inclusive(self) -> None:
        for database in DATABASES:
            for value, expected in ((22.0, PROMISING), (22.000001, "no_candidate")):
                with self.subTest(db=database, value=value):
                    cells = update_metric(
                        make_cells(),
                        "low4",
                        "q95",
                        value,
                        arm="T_all3",
                        operator="head3",
                        database=database,
                    )
                    # Keep T-none safely above T-head to isolate the R-head guard.
                    cells = update_metric(
                        cells,
                        "low4",
                        "q95",
                        30.0,
                        arm="T_all3",
                        operator="none",
                        database=database,
                    )
                    self.assert_decision(cells, expected)

    def test_low4_self_adaptation_guard_is_per_database_and_inclusive(self) -> None:
        for database in DATABASES:
            for value, expected in ((11.0, PROMISING), (11.000001, "no_candidate")):
                with self.subTest(db=database, value=value):
                    cells = update_metric(
                        make_cells(),
                        "low4",
                        "q95",
                        10.0,
                        arm="T_all3",
                        operator="none",
                        database=database,
                    )
                    cells = update_metric(
                        cells,
                        "low4",
                        "q95",
                        value,
                        arm="T_all3",
                        operator="head3",
                        database=database,
                    )
                    self.assert_decision(cells, expected)

    def test_macro_gain_cannot_override_one_local_guard_failure(self) -> None:
        cells = update_metric(
            make_cells(10.0),
            "all",
            "q95",
            111.0,
            arm="T_all3",
            operator="head3",
            database="telstra",
        )
        self.assert_decision(cells, "no_candidate")

    def test_guard_failure_is_not_engineering_blocked(self) -> None:
        cells = update_metric(
            make_cells(),
            "low4",
            "q95",
            23.0,
            arm="T_all3",
            operator="head3",
        )
        self.assert_decision(cells, "no_candidate")

    def test_low4_q50_is_validated_but_is_not_an_extra_candidate_gate(self) -> None:
        cells = update_metric(
            make_cells(),
            "low4",
            "q50",
            19.0,
            arm="T_all3",
            operator="head3",
        )
        self.assert_decision(cells, PROMISING)

    def test_none_never_contributes_to_main_macro_or_all_four_plus_guards(self) -> None:
        for q95, expected in ((90.0, PROMISING), (95.0, "no_candidate")):
            for arm in ARMS:
                for value in (1.0, 1e9):
                    with self.subTest(q95=q95, arm=arm, value=value):
                        cells = make_cells(q95)
                        for bucket in ("all", "4+"):
                            cells = update_metric(
                                cells,
                                bucket,
                                "q50",
                                1.0,
                                arm=arm,
                                operator="none",
                            )
                            cells = update_metric(
                                cells,
                                bucket,
                                "q95",
                                value,
                                arm=arm,
                                operator="none",
                            )
                        self.assert_decision(cells, expected)

    def test_low4_not_four_plus_is_the_self_adaptation_guard(self) -> None:
        cells = update_metric(
            make_cells(),
            "low4",
            "q95",
            10.0,
            arm="T_all3",
            operator="none",
        )
        self.assert_decision(cells, "no_candidate")

    def test_arithmetic_macro_not_geometric_or_macro_log(self) -> None:
        cells = make_cells()
        for database, reference, treatment in (
            ("chembl", 100.0, 100.0),
            ("legalacts", 100.0, 100.0),
            ("telstra", 1000.0, 850.0),
        ):
            cells = update_metric(
                cells,
                "4+",
                "q95",
                reference,
                arm="R_head3",
                operator="head3",
                database=database,
            )
            cells = update_metric(
                cells,
                "4+",
                "q95",
                treatment,
                arm="T_all3",
                operator="head3",
                database=database,
            )
        # Ratio of arithmetic macros = 1050/1200; geometric gain is <10%.
        self.assert_decision(cells, PROMISING)

    def test_database_macro_is_equal_weight_not_query_count_weighted(self) -> None:
        cells = make_cells()
        for database, q95, count in (
            ("chembl", 100.0, 100000),
            ("legalacts", 85.0, 1000),
            ("telstra", 85.0, 1000),
        ):
            cells = update_metric(
                cells,
                "4+",
                "q95",
                q95,
                arm="T_all3",
                operator="head3",
                database=database,
            )
            cells = update_metric(cells, "all", "count", count, database=database)
            cells = update_metric(cells, "4+", "count", count // 2, database=database)
        self.assert_decision(cells, PROMISING)

    def test_five_run_means_not_mean_of_ratios_or_per_run_guard(self) -> None:
        cells = make_cells()
        for bucket in ("all", "4+"):
            cells = update_metric(cells, bucket, "q50", 1.0)
            for run, value in enumerate((1.0, 1.0, 1.0, 1.0, 496.0)):
                cells = update_metric(
                    cells,
                    bucket,
                    "q95",
                    value,
                    arm="R_head3",
                    operator="head3",
                    run=run,
                )
        # R means are 100, T means are 90; four individual ratios are 90.
        self.assert_decision(cells, PROMISING)

    def test_low4_guard_uses_all_five_run_means_not_worst_run(self) -> None:
        cells = make_cells()
        for run, value in enumerate((30.0, 10.0, 10.0, 10.0, 40.0)):
            cells = update_metric(
                cells,
                "low4",
                "q95",
                value,
                arm="T_all3",
                operator="head3",
                run=run,
            )
        self.assert_decision(cells, PROMISING)

    def test_run_quantiles_are_not_pooled_or_count_weighted(self) -> None:
        cells = make_cells()
        for run, value in enumerate((100.0, 100.0, 100.0, 100.0, 50.0)):
            cells = update_metric(
                cells,
                "4+",
                "q95",
                value,
                arm="T_all3",
                operator="head3",
                run=run,
            )
            cells = update_metric(
                cells, "all", "count", 100000 if run < 4 else 1000, run=run
            )
            cells = update_metric(
                cells, "4+", "count", 50000 if run < 4 else 500, run=run
            )
        self.assert_decision(cells, PROMISING)

    def test_order_does_not_change_decision_or_report(self) -> None:
        cells = make_cells()
        shuffled = random.Random(2718).sample(cells, len(cells))
        expected = self.assert_decision(cells, PROMISING)
        self.assertEqual(self.assert_decision(shuffled, PROMISING), expected)
        self.assertEqual(
            self.assert_decision(list(reversed(cells)), PROMISING), expected
        )

    def test_every_main_bucket_count_must_reach_thirty_not_mean_thirty(self) -> None:
        for arm in ARMS:
            for operator in OPERATORS:
                for database in DATABASES:
                    for run in range(5):
                        with self.subTest(
                            arm=arm, operator=operator, db=database, run=run
                        ):
                            cells = update_metric(
                                make_cells(),
                                "low4",
                                "count",
                                29,
                                arm=arm,
                                operator=operator,
                                database=database,
                                run=run,
                            )
                            self.assert_decision(cells, "blocked")

    def test_one_paired_small_run_blocks_even_when_mean_count_exceeds_thirty(
        self,
    ) -> None:
        for database in DATABASES:
            for run in range(5):
                with self.subTest(database=database, run=run):
                    cells = update_metric(
                        make_cells(),
                        "low4",
                        "count",
                        29,
                        database=database,
                        run=run,
                    )
                    # Pairing is intact; (29 + 4*40)/5 > 30 must not rescue it.
                    self.assert_decision(cells, "blocked")

    def test_all_candidate_thresholds_can_pass_at_their_exact_boundaries(self) -> None:
        cells = make_cells(80.0)
        cells = update_metric(
            cells,
            "4+",
            "q95",
            110.0,
            arm="T_all3",
            operator="head3",
            database="chembl",
        )
        # Main macro is (110+80+80)/3 = 90 against reference 100.
        for field, value in (("q50", 11.0), ("q95", 110.0)):
            cells = update_metric(
                cells, "all", field, value, arm="T_all3", operator="head3"
            )
        cells = update_metric(cells, "4+", "q50", 11.0, arm="T_all3", operator="head3")
        cells = update_metric(
            cells, "low4", "q95", 22.0, arm="T_all3", operator="head3"
        )
        cells = update_metric(cells, "low4", "count", 30)
        self.assert_decision(cells, PROMISING)

    def test_paired_count_thirty_boundary_is_inclusive(self) -> None:
        for count, expected in ((29, "blocked"), (30, PROMISING), (31, PROMISING)):
            with self.subTest(count=count):
                self.assert_decision(
                    update_metric(make_cells(), "low4", "count", count), expected
                )

    def test_each_verification_is_required_and_has_priority_over_quality(self) -> None:
        for reproduction, engineering in ((False, True), (True, False), (False, False)):
            for q95 in (90.0, 100.0):
                with self.subTest(
                    reproduction=reproduction, engineering=engineering, q95=q95
                ):
                    self.assert_decision(
                        make_cells(q95),
                        "blocked",
                        reproduction_verified=reproduction,
                        engineering_verified=engineering,
                    )

    def test_verification_flags_reject_truthy_non_bools(self) -> None:
        for invalid in (None, 0, 1, "true", [], {}, float("nan")):
            for flag in ("reproduction_verified", "engineering_verified"):
                with self.subTest(flag=flag, invalid=invalid):
                    self.assert_decision(make_cells(), "blocked", **{flag: invalid})

    def test_keyword_only_verifications_have_no_implicit_success_defaults(self) -> None:
        with self.assertRaises(TypeError):
            decide(make_cells())
        with self.assertRaises(TypeError):
            decide(make_cells(), True, True)

    def test_missing_duplicate_or_extra_cells_block(self) -> None:
        cells = make_cells()
        for invalid in (
            cells[:-1],
            cells[1:],
            cells + [copy.deepcopy(cells[0])],
            [cells[0], *cells[:-1]],
        ):
            with self.subTest(length=len(invalid)):
                self.assert_decision(invalid, "blocked")

    def test_null_empty_and_invalid_container_types_block(self) -> None:
        for invalid in (None, [], {}, "", "cells", 60, True, tuple(make_cells())):
            with self.subTest(input_type=type(invalid).__name__):
                self.assert_decision(invalid, "blocked")

    def test_non_dict_cell_blocks(self) -> None:
        for invalid in (None, [], "cell", 1, True):
            with self.subTest(invalid=invalid):
                self.assert_decision([invalid, *make_cells()[1:]], "blocked")

    def test_each_required_cell_field_is_mandatory(self) -> None:
        for field in ("arm", "operator", "database", "run", "k", "metrics"):
            with self.subTest(field=field):
                cells = replace_first(
                    make_cells(),
                    lambda cell: {
                        key: value for key, value in cell.items() if key != field
                    },
                )
                self.assert_decision(cells, "blocked")

    def test_source_allowlist_rejects_targets_other_sources_and_tricks(self) -> None:
        for database in (
            "imdb",
            "stats",
            "genome",
            "ergastf1",
            "walmart",
            "",
            None,
            [],
            "Chembl",
            "chembl ",
            "chembl​",
            "chеmbl",
            "数据库😀",
            "chembl';--",
        ):
            with self.subTest(database=database):
                cells = replace_first(
                    make_cells(), lambda cell: {**cell, "database": database}
                )
                self.assert_decision(cells, "blocked")

    def test_arm_and_operator_allowlists_reject_deployment_scope_search(self) -> None:
        for field, invalid_values in (
            ("arm", ("A_all5_head3", "R", "T", "T_all5", None, [], "")),
            ("operator", ("all3", "last3", "head5", None, [], "head3​", "")),
        ):
            for invalid in invalid_values:
                with self.subTest(field=field, invalid=invalid):
                    cells = replace_first(
                        make_cells(), lambda cell: {**cell, field: invalid}
                    )
                    self.assert_decision(cells, "blocked")

    def test_k20_and_exact_integer_run_zero_through_four_only(self) -> None:
        for field, invalid_values in (
            ("k", (5, 10, 50, 100, 0, -20, 20.0, "20", True, None, [])),
            ("run", (-1, 5, 10000, 0.0, "0", True, False, None, [])),
        ):
            for invalid in invalid_values:
                with self.subTest(field=field, invalid=invalid):
                    cells = replace_first(
                        make_cells(), lambda cell: {**cell, field: invalid}
                    )
                    self.assert_decision(cells, "blocked")

    def test_metrics_and_each_required_bucket_must_be_dicts(self) -> None:
        for invalid in (None, [], "", 1, True):
            with self.subTest(metrics=invalid):
                cells = replace_first(
                    make_cells(), lambda cell: {**cell, "metrics": invalid}
                )
                self.assert_decision(cells, "blocked")
            for bucket in BUCKETS:
                with self.subTest(bucket=bucket, invalid=invalid):
                    cells = replace_first(
                        make_cells(),
                        lambda cell: {
                            **cell,
                            "metrics": {**cell["metrics"], bucket: invalid},
                        },
                    )
                    self.assert_decision(cells, "blocked")

    def test_missing_bucket_or_metric_field_blocks(self) -> None:
        for bucket in BUCKETS:
            with self.subTest(missing_bucket=bucket):
                cells = replace_first(
                    make_cells(),
                    lambda cell: {
                        **cell,
                        "metrics": {
                            key: value
                            for key, value in cell["metrics"].items()
                            if key != bucket
                        },
                    },
                )
                self.assert_decision(cells, "blocked")
            for field in ("count", "q50", "q95"):
                with self.subTest(bucket=bucket, missing_field=field):
                    cells = replace_first(
                        make_cells(),
                        lambda cell: {
                            **cell,
                            "metrics": {
                                **cell["metrics"],
                                bucket: {
                                    key: value
                                    for key, value in cell["metrics"][bucket].items()
                                    if key != field
                                },
                            },
                        },
                    )
                    self.assert_decision(cells, "blocked")

    def test_counts_reject_empty_negative_noninteger_bool_and_nonfinite(self) -> None:
        for bucket in BUCKETS:
            for invalid in (
                0,
                -1,
                40.0,
                "40",
                None,
                True,
                False,
                math.nan,
                math.inf,
                [],
            ):
                with self.subTest(bucket=bucket, invalid=invalid):
                    self.assert_decision(
                        update_metric(make_cells(), bucket, "count", invalid), "blocked"
                    )

    def test_count_nesting_and_all_arm_operator_pairing_are_required(self) -> None:
        for bucket, invalid in (("all", 499), ("4+", 39), ("low4", 501)):
            with self.subTest(bucket=bucket, invalid=invalid):
                self.assert_decision(
                    update_metric(make_cells(), bucket, "count", invalid), "blocked"
                )
        for bucket in BUCKETS:
            for arm in ARMS:
                for operator in OPERATORS:
                    with self.subTest(bucket=bucket, arm=arm, operator=operator):
                        cells = update_metric(
                            make_cells(),
                            bucket,
                            "count",
                            41 if bucket == "low4" else 501,
                            arm=arm,
                            operator=operator,
                            database="chembl",
                            run=0,
                        )
                        self.assert_decision(cells, "blocked")

    def test_quantiles_reject_invalid_values_even_in_diagnostic_none(self) -> None:
        for operator in OPERATORS:
            for bucket in BUCKETS:
                for field in ("q50", "q95"):
                    for invalid in (
                        None,
                        True,
                        False,
                        "10",
                        [],
                        {},
                        math.nan,
                        math.inf,
                        -math.inf,
                        -1.0,
                        0.0,
                        0.999999,
                    ):
                        with self.subTest(
                            operator=operator,
                            bucket=bucket,
                            field=field,
                            invalid=invalid,
                        ):
                            cells = update_metric(
                                make_cells(),
                                bucket,
                                field,
                                invalid,
                                operator=operator,
                            )
                            self.assert_decision(cells, "blocked")

    def test_quantiles_must_be_ordered_in_every_bucket(self) -> None:
        for bucket in BUCKETS:
            for operator in OPERATORS:
                with self.subTest(bucket=bucket, operator=operator):
                    cells = update_metric(
                        make_cells(), bucket, "q50", 1001.0, operator=operator
                    )
                    self.assert_decision(cells, "blocked")

    def test_integer_unit_quantiles_and_large_finite_metrics_are_valid(self) -> None:
        cells = make_cells()
        for bucket in BUCKETS:
            cells = update_metric(cells, bucket, "q50", 1)
            cells = update_metric(cells, bucket, "q95", 1)
        self.assert_decision(cells, "no_candidate")
        for bucket in ("all", "4+"):
            cells = update_metric(cells, bucket, "q95", 1e300)
            cells = update_metric(
                cells, bucket, "q95", 8e299, arm="T_all3", operator="head3"
            )
        self.assert_decision(cells, PROMISING)

    def test_unknown_fields_and_raw_query_truth_cannot_enable_routing(self) -> None:
        for field, value in (
            ("true", [1, 99, 100]),
            ("query_true", [1, 100]),
            ("n_tables", [4, 5]),
            ("selected_operator", "none"),
            ("route_by_true", True),
            ("database_policy", {"telstra": "none"}),
        ):
            with self.subTest(field=field):
                cells = replace_first(make_cells(), lambda cell: {**cell, field: value})
                self.assert_decision(cells, "blocked")
        cells = replace_first(
            make_cells(),
            lambda cell: {
                **cell,
                "metrics": {**cell["metrics"], "joint": {}},
            },
        )
        self.assert_decision(cells, "blocked")
        cells = replace_first(
            make_cells(),
            lambda cell: {
                **cell,
                "metrics": {
                    **cell["metrics"],
                    "low4": {**cell["metrics"]["low4"], "status": "ok"},
                },
            },
        )
        self.assert_decision(cells, "blocked")

    def test_good_none_cannot_rescue_failed_head_via_truth_or_database_routing(
        self,
    ) -> None:
        cells = update_metric(
            make_cells(70.0),
            "low4",
            "q95",
            23.0,
            arm="T_all3",
            operator="head3",
            database="telstra",
        )
        cells = update_metric(
            cells,
            "low4",
            "q95",
            5.0,
            arm="T_all3",
            operator="none",
            database="telstra",
        )
        self.assert_decision(cells, "no_candidate")

    def test_large_duplicated_matrix_is_blocked_not_partially_accepted(self) -> None:
        self.assert_decision(make_cells() * 200, "blocked")

    def test_success_no_candidate_and_blocked_do_not_modify_input(self) -> None:
        for cells, status, kwargs in (
            (make_cells(), PROMISING, {}),
            (make_cells(100.0), "no_candidate", {}),
            (update_metric(make_cells(), "low4", "count", 29), "blocked", {}),
            (make_cells(), "blocked", {"engineering_verified": False}),
        ):
            with self.subTest(status=status, kwargs=kwargs):
                snapshot = copy.deepcopy(cells)
                self.assert_decision(cells, status, **kwargs)
                self.assertEqual(cells, snapshot)

    def test_concurrent_calls_share_no_mutable_state_or_input_changes(self) -> None:
        cells = make_cells()
        snapshot = copy.deepcopy(cells)
        expected = self.assert_decision(cells, PROMISING)
        with ThreadPoolExecutor(max_workers=4) as executor:
            reports = list(
                executor.map(
                    lambda _: decide(
                        cells, reproduction_verified=True, engineering_verified=True
                    ),
                    range(16),
                )
            )
        for report in reports:
            self.assertEqual(report, expected)
        self.assertEqual(cells, snapshot)


if __name__ == "__main__":
    unittest.main()
