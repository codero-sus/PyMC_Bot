"""The vLLM teacher started from inside a DDP (torchrun / accelerate) process, and early stopping once the loss is flat.

A Kaggle run lost 44 min: vLLM inherited TORCHELASTIC_USE_AGENT_STORE=True from the launcher, so its workers waited
for a TCPStore server nobody started (two ~20 min hangs), then the transformers fallback loaded the 16 GB teacher with
no time left and wrote 0 answers. No GPU here: processes and the teacher are faked.
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

from pymc_bot import llm_train

NOTEBOOK = Path(__file__).resolve().parents[1] / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"
LAUNCHER = {"TORCHELASTIC_USE_AGENT_STORE": "True", "TORCHELASTIC_RUN_ID": "none", "RANK": "0", "LOCAL_RANK": "0",
            "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500",
            "GROUP_RANK": "0", "ROLE_RANK": "0", "ACCELERATE_MIXED_PRECISION": "fp16"}


def _jobs(n=3):
    return [{"key": f"k{i}", "messages": [{"role": "user", "content": f"q{i}"}], "thinking": False}
            for i in range(n)]


def test_the_teacher_does_not_inherit_the_ddp_launcher_environment(monkeypatch, tmp_path):
    for key, value in LAUNCHER.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("HF_TOKEN", "x")
    env = llm_train.vllm_env()
    assert not set(LAUNCHER) & set(env)
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1" and env["HF_TOKEN"] == "x" and env["VLLM_LOGGING_LEVEL"] == "INFO"

    seen = {}

    def run(command, stdout=None, stderr=None, env=None, timeout=None):
        seen["env"] = env
        Path(command[-1]).write_text(json.dumps({"k0": "a"}))
        return type("Done", (), {"returncode": 0})()

    assert llm_train.vllm_generate("t", _jobs(), 2048, 600, "/p", run=run, log=lambda *_: None) == {"k0": "a"}
    assert "TORCHELASTIC_USE_AGENT_STORE" not in seen["env"] and "RANK" not in seen["env"]


def test_a_hang_is_not_tried_twice(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(llm_train.time, "time", lambda: clock["now"])
    calls = []

    def hangs(command, stdout=None, stderr=None, env=None, timeout=None):
        calls.append(1)
        clock["now"] += 15 * 60  # stopped by the startup watchdog
        stdout.write("pymc: engine not ready after 15 min - stopped (it hangs)\n")
        return type("Done", (), {"returncode": 124})()

    logs = []
    with pytest.raises(RuntimeError):
        llm_train.vllm_generate("t", _jobs(), 2048, 2400, "/p", run=hangs, log=logs.append)
    assert len(calls) == 1 and "it hangs" in logs[0]

    calls.clear()

    def fails_fast(command, stdout=None, stderr=None, env=None, timeout=None):
        calls.append(1)
        clock["now"] += 60
        return type("Done", (), {"returncode": 1})()

    with pytest.raises(RuntimeError):
        llm_train.vllm_generate("t", _jobs(), 2048, 2400, "/p", run=fails_fast, log=lambda *_: None)
    assert len(calls) == 2  # a quick failure (memory) gets the low-memory try


def _python(code):
    return [sys.executable, "-c", code]


def test_the_watchdog_stops_a_hung_engine_with_all_its_processes(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = ("import subprocess, sys, time; "
            f"p = subprocess.Popen(['sleep', '60']); open({str(pid_file)!r}, 'w').write(str(p.pid)); time.sleep(60)")
    started = time.time()
    with open(tmp_path / "log.txt", "w", encoding="utf-8") as log:
        done = llm_train.run_vllm_process(_python(code), stdout=log, stderr=log, env=dict(os.environ),
                                          timeout=120, ready=tmp_path / "ready", startup=1.5, poll=0.2)
    assert done.returncode == 124 and time.time() - started < 40
    assert "it hangs" in (tmp_path / "log.txt").read_text()
    child = int(pid_file.read_text())
    for _ in range(50):  # the engine's own processes go too (they would keep the T4s' memory)
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        pytest.fail("the engine's child process survived")


def test_the_watchdog_lets_a_started_engine_work(tmp_path):
    ready = tmp_path / "ready"
    code = f"import time; open({str(ready)!r}, 'w').write('ready'); time.sleep(2.5)"
    done = llm_train.run_vllm_process(_python(code), env=dict(os.environ), timeout=60, ready=ready,
                                      startup=1.0, poll=0.2)
    assert done.returncode == 0


def test_the_worker_says_when_the_engine_is_ready(tmp_path, monkeypatch):
    class LLM:
        def __init__(self, **options):
            pass

        def get_tokenizer(self):
            return None

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=LLM, SamplingParams=lambda **k: k))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        cuda=types.SimpleNamespace(device_count=lambda: 2)))
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"teacher": "unsloth/Qwen3-8B", "max_seq": 10, "seconds": 60, "jobs": []}))
    out = tmp_path / "out.json"
    assert llm_train.vllm_worker(str(job), str(out)) == 0
    assert Path(str(out) + ".ready").is_file()


def _args(tmp_path, **extra):
    args = llm_train.build_parser().parse_args(["--preset", "minecraftism", "--out", str(tmp_path / "out")])
    args.stream_cache = str(tmp_path / "cache")
    args.distill_rows = 50
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def _rows():
    return [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"chat {i}?"},
                          {"role": "assistant", "content": f"answer {i}"}], "source": "general", "kind": "general"}
            for i in range(4)]


def test_a_failed_vllm_teacher_is_not_tried_again_in_the_same_session(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PYMC_SCRATCH", str(tmp_path / "scratch"))
    python = tmp_path / "python"
    python.write_text("")
    used = []

    def broken(*_a):
        used.append("vllm")
        raise RuntimeError("no engine")

    def hf(teacher, jobs, max_seq, seconds, batch=0, log=print):
        used.append("hf")
        return {}

    monkeypatch.setattr(llm_train, "vllm_generate", broken)
    monkeypatch.setattr(llm_train, "teacher_generate", hf)
    deadline = time.time() + 10 * 3600
    preset = llm_train.PRESETS["minecraftism"]
    llm_train.distill_offpolicy(_args(tmp_path / "a", vllm_python=str(python), uncensored=True, deadline=deadline), preset, _rows())
    assert used == ["vllm", "hf"]
    used.clear()
    llm_train.distill_offpolicy(_args(tmp_path / "b", vllm_python=str(python), uncensored=True, deadline=deadline), preset, _rows())
    assert used == ["hf"] and "already failed" in capsys.readouterr().out
    used.clear()  # a new session (another deadline) tries vLLM again
    llm_train.distill_offpolicy(_args(tmp_path / "c", vllm_python=str(python), uncensored=True, deadline=deadline + 86400),
                                preset, _rows())
    assert used == ["vllm", "hf"]
    assert not llm_train.vllm_failed_before("teacher", 0)


def test_no_transformers_fallback_without_time_to_use_it(tmp_path, monkeypatch, capsys):
    python = tmp_path / "python"
    python.write_text("")
    clock = {"now": 5000.0}
    monkeypatch.setattr(llm_train.time, "time", lambda: clock["now"])

    def slow_and_broken(*_a):
        clock["now"] += 38 * 60  # the old hang ate the teacher's 40 minutes
        raise RuntimeError("no engine")

    monkeypatch.setattr(llm_train, "vllm_generate", slow_and_broken)
    monkeypatch.setattr(llm_train, "teacher_generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded with no time left")))
    args = _args(tmp_path, vllm_python=str(python))
    rows = llm_train.distill_offpolicy(args, llm_train.PRESETS["minecraftism"], _rows())
    assert len(rows) == 4 and "not loading it with transformers" in capsys.readouterr().out


# --------------------------------------------------------------------------- stop when learned
def test_a_flat_loss_stops_after_most_rows_were_seen_once():
    plateau = llm_train.LossPlateau()
    total, epochs = 4000, 2.0
    stopped = None
    for step in range(10, total + 1, 10):
        loss = max(0.08, 2.0 * 0.97 ** (step / 10))
        if plateau.add(step, loss + 0.01 * ((step // 10) % 3), total, epochs):
            stopped = step
            break
    assert stopped is not None and 1500 <= stopped <= 2200  # >= 3/4 of epoch 1, long before step 4000


def test_a_falling_loss_keeps_training():
    plateau = llm_train.LossPlateau()
    total = 2000
    assert not any(plateau.add(step, 3.0 * 0.995 ** (step / 10) * 0.9 ** (step / 200), total)
                   for step in range(10, total + 1, 10))


def _fake_transformers(monkeypatch):
    class TrainerCallback:
        pass

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=TrainerCallback))


def test_the_learned_callback_stops_every_gpu_at_the_same_step(monkeypatch, capsys):
    _fake_transformers(monkeypatch)
    args = types.SimpleNamespace(stop_when_learned=False, deadline=0)
    assert llm_train.learned_callback(args) is None and llm_train.training_callbacks(args) is None
    args.stop_when_learned = True
    asked = []
    callback = llm_train.learned_callback(args, agree=lambda flag: asked.append(flag) or flag)
    assert llm_train.training_callbacks(args) and len(llm_train.training_callbacks(args)) == 1
    state = types.SimpleNamespace(global_step=0, max_steps=1000, num_train_epochs=1)
    control = types.SimpleNamespace(should_training_stop=False)
    for step in range(10, 1001, 10):
        state.global_step = step
        callback.on_step_end(None, state, control)
        if control.should_training_stop:
            break
        callback.on_log(None, state, control, logs={"loss": 0.1})
    assert control.should_training_stop and 750 <= state.global_step < 1000
    assert args.stopped_early["reason"] == "learned" and "loss stopped falling" in capsys.readouterr().out
    assert len(asked) == state.global_step // 10  # one agreement per 10 steps on every rank, not every step


def test_the_notebook_turns_early_stopping_on():
    cells = ["".join(cell["source"]) for cell in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]
    assert "STOP_WHEN_LEARNED = True" in cells[1]
    assert 'FLAGS += " --stop-when-learned" if STOP_WHEN_LEARNED else ""' in cells[3]
    assert llm_train.build_parser().parse_args(["--stop-when-learned"]).stop_when_learned
