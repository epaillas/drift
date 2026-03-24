"""CLI help tests for correlation-matrix plotting scripts."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _help_output(script_name: str) -> str:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def test_pgg_help_describes_covariance_flags():
    output = _help_output("plot_correlation_matrix_pgg.py")
    assert "Use the analytic cubic-box P_gg covariance instead of" in output
    assert "the mock covariance." in output
    assert "Minimum k retained on the mock covariance path" in output
    assert "--kmin" not in output


def test_xigg_help_describes_s_grid_and_reciprocal_k_inference():
    output = _help_output("plot_correlation_matrix_xigg.py")
    assert "Maximum separation shown in the xi_gg matrix" in output
    assert "the script infers a rough" in output
    assert "reciprocal cut from --smax" in output
    assert "from --smax" in output
    assert "Provide at most one of --ds or --ns." in output
    assert "--kmin" not in output


def test_xiqq_help_describes_config_space_and_pair_controls():
    output = _help_output("plot_correlation_matrix_xiqq.py")
    assert "Restrict the plotted xi_qq blocks to DS auto pairs" in output
    assert "only." in output
    assert "Use the analytic cubic-box P_qq covariance before" in output
    assert "propagating it to xi_qq." in output
    assert "Density-split model used to build the analytic" in output
    assert "fiducial P_qq spectra before propagation." in output
    assert "--kmin" not in output
