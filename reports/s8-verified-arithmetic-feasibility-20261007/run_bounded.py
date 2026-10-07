"""Single bounded test run; do not import production packages or use network."""

import json
import resource
import signal
import time
import unittest
from pathlib import Path

import verified_arithmetic as v


def stop(_signal, _frame):
    raise TimeoutError("single run wall/CPU budget exceeded")


signal.signal(signal.SIGALRM, stop)
signal.signal(signal.SIGXCPU, stop)
signal.alarm(30)
resource.setrlimit(resource.RLIMIT_CPU, (20, 21))
started = time.monotonic()
try:
    v.budget()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.discover(
            str(Path(__file__).parent), "test_arithmetic.py"
        )
    )
    rss = v.budget()
    output = {
        "purpose": "bounded synthetic feasibility not production acceptance",
        "tests": result.testsRun,
        "successful": result.wasSuccessful(),
        "failures": len(result.failures),
        "errors": len(result.errors),
        "wall_seconds_budget_check_only": time.monotonic() - started,
        "maxrss_bytes": rss,
        "limits": {
            "wall_seconds": 30,
            "CPU_seconds": 20,
            "RSS_bytes": v.RSS_LIMIT,
            "matrix_shape_limit": [16, 8],
            "sqrt_decimal_scale_digits": 80,
        },
        "production_modified": False,
        "real_P_read": False,
        "network_used": False,
        "GPU_used": False,
        "training": False,
        "G1_status": "failed",
        "rank_acceptance_in_production": "unchanged_unknown",
    }
    print(json.dumps(output, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)
finally:
    signal.alarm(0)
