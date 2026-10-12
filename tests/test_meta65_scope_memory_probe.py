"""Tests for the bounded Meta65 scope RSS-trajectory probe instrument.

The first probe run (2026-10-12) produced a trajectory whose every sample
carried ``step: 0`` and whose samples numbered one per validation line rather
than one per poll. Both are instrument defects, not training behaviour, and they
are pinned here before the probe is rerun over a longer horizon.
"""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
from pathlib import Path

from scripts import meta65_scope_memory_probe as probe

# Verbatim monitor line from memory-probe-20261012R.probe.log on the server.
REAL_MONITOR_LINE = (
    '{"step": 100, "train_loss": 0.560346320271492, '
    '"validation_macro_log_q95": 3.0207796187229903, '
    '"counts": {"support_presentations": 42975, "query_presentations": 25600, '
    '"backward_calls": 1600, "optimizer_steps": 100}}'
)
# The final line the child emits once training ends.
REAL_FINAL_LINE = (
    '{"probe": "{\\"done\\": true, \\"last_step\\": 1500, \\"counts\\": '
    '{\\"support_presentations\\": 667950, \\"query_presentations\\": 384000, '
    '\\"backward_calls\\": 24000, \\"optimizer_steps\\": 1500}}"}'
)


class StepParsingTests(unittest.TestCase):
    def test_parses_the_real_monitor_line(self) -> None:
        """The shipped parser looked for a token starting with '"step":'."""
        self.assertEqual(probe.parse_step(REAL_MONITOR_LINE, 0), 100)

    def test_shipped_parser_could_never_match(self) -> None:
        # Once commas are split away the line's first token is '{"step":', so a
        # startswith('"step":') check cannot match it. This is the exact defect.
        tokens = REAL_MONITOR_LINE.replace(",", " ").split()
        self.assertIn('{"step":', tokens)
        self.assertEqual([t for t in tokens if t.startswith('"step":')], [])

    def test_never_goes_backwards_and_ignores_unrelated_lines(self) -> None:
        for line, previous, expected in (
            ('{"step": 200}', 100, 200),
            ('{"step": 1500}', 1500, 1500),
            ('{"done": true, "last_step": 1500}', 1400, 1400),
            ("not json at all", 42, 42),
            ("", 42, 42),
        ):
            with self.subTest(line=line, previous=previous):
                self.assertEqual(probe.parse_step(line, previous), expected)

    def test_final_done_line_reports_its_last_step(self) -> None:
        self.assertIn('last_step', REAL_FINAL_LINE)
        self.assertGreaterEqual(probe.parse_step(REAL_FINAL_LINE, 0), 0)

    def test_nested_counts_key_is_not_mistaken_for_the_step(self) -> None:
        """The pattern must key on the exact name, not a substring suffix."""
        nested = '{"step": 7, "counts": {"optimizer_steps": 9999}}'
        self.assertEqual(probe.parse_step(nested, 0), 7)
        self.assertEqual(probe.parse_step(REAL_MONITOR_LINE, 0), 100)


class DrainTests(unittest.TestCase):
    """The sample loop must not block on the child's monitor output."""

    class _Log:
        def __init__(self) -> None:
            self.text = ""

        def write(self, text: str) -> None:
            self.text += text

        def flush(self) -> None:
            pass

    def _drain(self, chunks: list[bytes], state: dict) -> tuple[int, str]:
        """Feed chunks through a real pipe, then close both ends."""
        read_fd, write_fd = os.pipe()
        try:
            for chunk in chunks:
                os.write(write_fd, chunk)
            os.close(write_fd)
            write_fd = -1
            stream = io.open(read_fd, "rb", closefd=True)
            try:
                log = self._Log()
                step = probe.drain(stream, log, state)
                return step, log.text
            finally:
                stream.close()
        finally:
            if write_fd >= 0:
                os.close(write_fd)

    def test_complete_lines_are_logged_and_mined(self) -> None:
        state: dict = {"step": 0, "pending": ""}
        step, text = self._drain([(REAL_MONITOR_LINE + "\n").encode()], state)
        self.assertEqual(step, 100)
        self.assertEqual(text, REAL_MONITOR_LINE + "\n")
        self.assertEqual(state["pending"], "")

    def test_partial_line_is_buffered_not_counted(self) -> None:
        state: dict = {"step": 0, "pending": ""}
        head, tail = REAL_MONITOR_LINE[:40], REAL_MONITOR_LINE[40:]
        step, text = self._drain([head.encode()], state)
        self.assertEqual(step, 0)
        self.assertEqual(text, "")
        self.assertEqual(state["pending"], head)
        step, text = self._drain([(tail + "\n").encode()], state)
        self.assertEqual(step, 100)
        self.assertEqual(text, REAL_MONITOR_LINE + "\n")

    def test_no_output_keeps_existing_step_and_pending(self) -> None:
        state: dict = {"step": 300, "pending": ""}
        step, text = self._drain([], state)
        self.assertEqual(step, 300)
        self.assertEqual(text, "")

    def test_many_lines_in_one_read_are_all_captured(self) -> None:
        state: dict = {"step": 0, "pending": ""}
        lines = [
            REAL_MONITOR_LINE.replace('"step": 100', f'"step": {n}') + "\n"
            for n in (100, 200, 300)
        ]
        step, text = self._drain([b"".join(line.encode() for line in lines)], state)
        self.assertEqual(step, 300)
        self.assertEqual(text.count("\n"), 3)


class RecordShapeTests(unittest.TestCase):
    def test_poll_interval_and_horizon_cover_a_full_run(self) -> None:
        self.assertEqual(probe.POLL_SECONDS, 1.0)
        self.assertEqual(probe.DEFAULT_STEPS, 4000)
        # The hard cap must outlast a 4000-step R training (observed ~0.37 s/step
        # for the probe's first 1500) plus preparation, without being unbounded.
        self.assertGreater(probe.HARD_SECONDS, 2400.0)
        self.assertLessEqual(probe.HARD_SECONDS, 7200.0)

    def test_kill_limits_are_probe_budgets_not_research_gates(self) -> None:
        self.assertEqual(probe.RSS_KILL_BYTES, 16 * 1024**3)
        self.assertEqual(probe.OUTPUT_KILL_BYTES, 2 * 1024**3)

    def test_default_root_is_the_deployed_scope_code(self) -> None:
        self.assertEqual(
            str(probe.DEFAULT_ROOT), "/gly/weilin/spatial-meta-ce-meta65-scope-20261011-code"
        )


class SuperviseLoopTests(unittest.TestCase):
    """End-to-end loop behaviour with a real child process.

    The ``/proc`` readers are replaced so this runs on any POSIX host; the
    child process, pipe draining, log writing and record shape are real.
    """

    STUB = (
        "import json, time\n"
        "data = bytearray()\n"
        "for step in (100, 200, 300, 400, 500):\n"
        "    data.extend(b'x' * (1024 * 1024))\n"
        "    print(json.dumps({'step': step, 'counts': {'optimizer_steps': step},"
        " 'pad': len(data)}), flush=True)\n"
        "    time.sleep(0.25)\n"
        "print(json.dumps({'done': True, 'last_step': 500}), flush=True)\n"
    )

    def _supervise(self, steps: int = 500) -> tuple[dict, str]:
        import subprocess
        import tempfile
        import textwrap

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "stub.py").write_text(textwrap.dedent(self.STUB), encoding="utf-8")
            out = root / "probe-out"
            real_popen = subprocess.Popen
            real_rss, real_alive = probe.group_rss, probe.group_alive

            def stub_popen(*args: object, **kwargs: object) -> object:
                return real_popen(
                    [sys.executable, "-B", str(root / "stub.py")],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )

            probe.subprocess.Popen = stub_popen
            probe.group_rss = lambda pgid: 1_000_000_000
            probe.group_alive = lambda pgid: []
            try:
                record = probe.supervise(out, steps)
            finally:
                probe.subprocess.Popen = real_popen
                probe.group_rss, probe.group_alive = real_rss, real_alive
            log_text = (root / "probe-out.probe.log").read_text(encoding="utf-8")
        return record, log_text

    def test_step_attribution_advances_and_final_step_is_seen(self) -> None:
        record, log_text = self._supervise()
        steps = [sample["step"] for sample in record["samples"]]
        self.assertEqual(record["returncode"], 0)
        self.assertIsNone(record["failure"])
        self.assertEqual(record["final_step_seen"], 500)
        self.assertEqual(max(steps), 500)
        self.assertEqual(min(steps), 0)
        self.assertTrue(all(a <= b for a, b in zip(steps, steps[1:])), steps)
        self.assertGreater(len(steps), 1)

    def test_all_child_lines_reach_the_log(self) -> None:
        record, log_text = self._supervise()
        for step in (100, 200, 300, 400, 500):
            self.assertIn(f'"step": {step}', log_text)
        self.assertIn('"last_step": 500', log_text)
        self.assertEqual(record["cleanup_verified"], True)
        self.assertGreater(record["output_bytes"], 0)

    def test_record_is_strict_json_with_the_expected_fields(self) -> None:
        record, _ = self._supervise()
        self.assertEqual(
            set(record),
            {
                "schema", "probe", "steps_requested", "device", "cuda_visible_devices",
                "returncode", "failure", "elapsed", "samples", "cleanup_verified",
                "signals", "final_step_seen", "output_bytes",
            },
        )
        self.assertEqual(record["schema"], 1)
        self.assertEqual(record["probe"], "meta65-scope-r-memory")
        self.assertEqual(record["steps_requested"], 500)
        json.dumps(record, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
