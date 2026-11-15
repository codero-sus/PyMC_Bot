"""The simulated goal suite must not masquerade as a live or LLM benchmark."""
from __future__ import annotations

import json

import pytest

from pymc_bot.__main__ import main
from pymc_bot.sim_bench import run_sim_bench


def test_sim_tech_benchmark_runs_real_goal_controller():
    report = run_sim_bench()
    assert report["kind"] == "simulated_goal_integration"
    assert report["cases"] == 3 and report["pass_rate"] == 1
    assert all(row["steps"] > 0 and row["completed"] for row in report["results"])
    assert report["results"][2]["paused_safely"] is True
    assert "no server" in report["note"]


def test_sim_bench_bounds_and_cli(tmp_path, capsys):
    with pytest.raises(ValueError):
        run_sim_bench(repeats=0)
    with pytest.raises(ValueError):
        run_sim_bench(max_steps=0)
    path = tmp_path / "report.json"
    assert main(["bench", "--suite", "sim-tech", "--output", str(path)]) == 0
    assert json.loads(path.read_text())["kind"] == "simulated_goal_integration"
    assert json.loads(capsys.readouterr().out)["pass_rate"] == 1
    assert main(["bench"]) == 2
    assert "--model is required" in capsys.readouterr().err
