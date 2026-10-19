import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import meta65_joint_supervisor as supervisor


class HoldoutSupervisorTests(unittest.TestCase):
    def command(self, output: Path, *, formal: bool) -> list[str]:
        cells = 50 if formal else 0
        micro = 0 if formal else 10
        code = (
            "import json,sys\n"
            "from pathlib import Path\n"
            "out=Path(sys.argv[1]); out.mkdir()\n"
            "(out/'phase.json').write_text(json.dumps({'phase':'certification'}))\n"
            f"(out/'status.json').write_text(json.dumps({{'status':'worker_complete',"
            f"'mode':{('formal' if formal else 'cpu_preflight')!r},'cells':{cells},"
            f"'micro_cells':{micro},'training':True,'target_read':False}}))\n"
        )
        return [sys.executable, "-B", "-c", code, str(output)]

    def test_custom_formal_cell_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "formal"
            with patch.object(supervisor, "_live_owned_descendants", return_value=[]):
                record = supervisor.supervise(
                    self.command(output, formal=True),
                    output,
                    mode="formal",
                    sample_fn=lambda child: {"rss_bytes": 1, "vram_bytes": 0},
                    formal_cells=50,
                    preflight_micro_cells=10,
                )
            self.assertEqual(record["status"], "completed")

    def test_custom_preflight_micro_cell_count(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preflight"
            with patch.object(supervisor, "_live_owned_descendants", return_value=[]):
                record = supervisor.supervise(
                    self.command(output, formal=False),
                    output,
                    mode="cpu_preflight",
                    sample_fn=lambda child: {"rss_bytes": 1, "vram_bytes": 0},
                    formal_cells=50,
                    preflight_micro_cells=10,
                )
            self.assertEqual(record["status"], "preflight_completed")


if __name__ == "__main__":
    unittest.main()
