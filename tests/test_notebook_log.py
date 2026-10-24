"""The notebook's live log: tidied (the full output stays in /kaggle/working/<edition>.log)."""

from __future__ import annotations

import collections
import json
import re
import time
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"


def _tidy(deadline):
    source = "".join(json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"][6]["source"])
    block = source.split("# ---- log tidy\n", 1)[1].split("# ---- end log tidy", 1)[0]
    scope = {"re": re, "time": time, "collections": collections, "DEADLINE": deadline}
    exec(block, scope)  # noqa: S102 - the notebook's own code
    return scope["tidy"]


def test_one_status_line_every_few_minutes_with_loss_and_eta():
    now = 1_000_000.0
    tidy = _tidy(now + 9 * 3600)
    step = "  10%|█         | {}/4230 [33:33<4:40:38,  4.44s/it]"
    assert tidy("ed", "{'loss': 0.0859, 'grad_norm': 0.66, 'learning_rate': 0.00017, 'epoch': 0.21}", now) is None
    first = tidy("ed", step.format(440), now)
    assert first.startswith("step 440/4230 (10%) · loss 0.086 · epoch 0.21 · 4.44s/it · ~4.7 h to go")
    assert "session: 9.0 h left" in first and "won't finish" not in first
    assert tidy("ed", step.format(440), now) is None  # the other T4's copy
    assert tidy("ed", step.format(441), now + 5) is None  # every step: not shown
    assert tidy("ed", step.format(500), now + 301).startswith("step 500/4230")


def test_warns_when_the_steps_do_not_fit_the_session():
    now = 2_000_000.0
    tidy = _tidy(now + 3 * 3600)
    line = tidy("ed", "   1%|▏         | 50/4230 [04:07<5:14:16,  4.51s/it]", now)
    assert "won't finish all steps" in line


def test_other_bars_and_doubled_banners_are_shown_once():
    now = 3_000_000.0
    tidy = _tidy(now + 9 * 3600)
    shards = "Loading checkpoint shards: 100%|██████████| 4/4 [01:04<00:00, 16.20s/it]"
    assert tidy("ed", shards, now) == "Loading checkpoint shards: 4/4 (100%)"
    assert tidy("ed", shards, now + 1) is None
    banner = "Unsloth 2026.10.8 patched 36 layers with 36 QKV layers, 36 O layers and 36 MLP layers."
    assert tidy("ed", banner, now) == banner and tidy("ed", banner, now) is None
    assert tidy("other", banner, now) == banner  # another edition still shows it
    for _ in range(2):  # tracebacks stay whole
        assert tidy("ed", "    ^^^^^^^^^^", now) == "    ^^^^^^^^^^"
        assert tidy("ed", "RuntimeError: Engine core initialization failed.", now)
