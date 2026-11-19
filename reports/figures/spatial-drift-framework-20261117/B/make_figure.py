"""Candidate B entry point: spatial-drift framework figure.

The composition is reorganised around the source-to-target spatial-drift
narrative; see b_figure.py for the rebuild and b_figure_lib.py for the shared
drawing helpers that inherit the VA-196 graphic language.
"""
from pathlib import Path
import sys
try:
    from source_asset import run
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'lib'))
    from source_asset import run
from b_figure import customize

if __name__=='__main__':run(__file__,customize)
