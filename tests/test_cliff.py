"""THE 12 HOURs: the Kaggle notebook's last-minutes report, the vLLM pre-flight, the model prefetch, the slow teacher."""
import os
import re
import shutil
import subprocess
import threading
import time
import types

from test_discord_notify import HOOK, _block, _cells, _helper, _wait

from pymc_bot import llm_train

MONITOR = ("# ---- Discord server monitor", "# ---- end Discord server monitor")


def _monitor(monkeypatch, tmp_path, **extra):
    env, _asked, sent = _helper(monkeypatch, mention="<@7>")
    start = 1_000_000.0
    env.update({"SESSION_START": start, "DEADLINE": start + 11.75 * 3600, "NOTEBOOK_URL": "https://k/nb",
                "DISCORD_EVERY_MIN": 0, "WORK": str(tmp_path), "TMP": str(tmp_path), "shutil": shutil,
                "subprocess": subprocess, "os": os, "DISCORD_URL": "", "CPUS": 4, "EDITIONS": ["minecraftism"]})
    env.update(extra)
    exec(_block(_cells()[3], *MONITOR), env)  # noqa: S102
    return env, start, sent


def test_the_report_says_what_is_safe_and_what_would_be_lost(monkeypatch, tmp_path):
    (tmp_path / "minecraftism-thinking").mkdir()
    (tmp_path / "minecraftism-thinking" / "model.gguf").write_bytes(b"0" * 60_000_000)
    reports = {"minecraftism": {"hub_repo": "me/M-GGUF", "hub_repo_safetensors": "me/M", "safetensors_uploaded": True},
               "minecraftism-thinking": {}}
    env, _start, _sent = _monitor(monkeypatch, tmp_path, READY=list(reports), CODES={"minecraftism": 0},
                                  report_of=reports.get,
                                  DISCORD_STATUS={"minecraftism-thinking": "step 2930/4230 (69%) · loss 0.020"})
    report = env["cliff_report"](quick=True)
    assert "**minecraftism:** ✅ done · [safetensors](https://huggingface.co/me/M) · [GGUF](https://huggingface.co/me/M-GGUF)" \
        in report
    assert "**minecraftism-thinking:** 🏋️ step 2930/4230" in report and "⚠️ 0.1 GB only in /kaggle/working" in report
    assert "1/2 edition(s) safe on Hugging Face" in report and "GPU" not in report  # quick: no nvidia-smi
    env["CODES"]["minecraftism-thinking"] = 0
    reports["minecraftism-thinking"] = dict(reports["minecraftism"])
    assert "The branch holds" in env["cliff_report"](quick=True)


def test_no_countdown_after_the_notebook_finished(monkeypatch, tmp_path):
    env, start, _sent = _monitor(monkeypatch, tmp_path)
    env["DISCORD_STATE"]["finished"] = True
    assert env["server_warnings"](start + 12 * 3600 - 60, {}) == []


def test_closing_at_the_cliff_sends_everything_once_more(monkeypatch, tmp_path):
    env, _start, sent = _monitor(monkeypatch, tmp_path)
    env.update({"KAGGLE_LIMIT": time.time() + 20, "SERVER_URL": "u", "DISCORD_URL": HOOK})
    exec(_block(_cells()[3], "def server_closing", "\n\nif DISCORD_URL:\n    import atexit"), env)  # noqa: S102
    env["server_closing"]()
    _wait(env)
    text = sent[0]["payload"]["embeds"][0]["description"]
    assert "THE 12 HOURs has come" in text and "**minecraftism:**" in text
    assert "content" not in sent[0]["payload"]  # no second ping: the 5-minute warning already pinged


def test_the_end_message_carries_the_full_report():
    cell = _cells()[12]
    assert cell.count("cliff_report()") == 4  # every way the notebook can end tells everything once more


def _vllm_block(tmp_path, popen_script, extra=None):
    """Run cell 3's pre-flight with a fake vLLM process."""
    messages = []
    env = {"os": os, "re": re, "time": time, "subprocess": subprocess, "threading": threading, "shutil": shutil,
           "sys": __import__("sys"), "WORK": str(tmp_path), "TMP": str(tmp_path), "gpus": ["T4", "T4"],
           "TRAINER_MODULE": llm_train, "TEACHER": "auto", "UNCENSORED": True, "VLLM_PREFLIGHT": True,
           "discord": lambda text, title="", level="info", ping=False: messages.append((title, text))}
    env.update(extra or {})
    cell = _cells()[3]
    exec(cell[cell.index("def trainer_args"):cell.index("# ---- prefetch")], env)  # noqa: S102
    exec(cell[cell.index("VLLM_VERSION = "):cell.index("def install_vllm")], env)  # noqa: S102
    exec(_block(cell, "# ---- vLLM pre-flight", "# ---- end vLLM pre-flight"), env)  # noqa: S102
    started = []

    def popen(command, stdout=None, **kwargs):
        started.append((command, kwargs))
        stdout.write(popen_script)
        stdout.flush()
        return types.SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0, pid=2 ** 30)

    return env, popen, started, messages


def test_vllm_preflight_ok_lets_vllm_write(tmp_path):
    env, popen, started, messages = _vllm_block(tmp_path, "INFO loading\nPREFLIGHT OK 95 'OK'\n")
    assert env["vllm_preflight"](popen=popen, timeout=5) is True
    command, kwargs = started[0]
    assert command[3:] == [llm_train.UNCENSORED_TEACHER, "2", ""] and kwargs["start_new_session"]  # TP=2, the 8B
    assert messages[0][0] == "Teacher ready"


def test_vllm_preflight_failure_shows_the_cause_before_any_training_time_is_lost(tmp_path):
    log = ("INFO loading\n(EngineCore pid=7) ERROR torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2 GiB\n"
           "RuntimeError: Engine core initialization failed. See root cause above.\n")
    env, popen, _started, messages = _vllm_block(tmp_path, log)
    assert env["vllm_preflight"](popen=popen, timeout=5) is False
    assert "CUDA out of memory" in env["VLLM_STATE"]["cause"]
    assert "Engine core initialization failed" not in env["VLLM_STATE"]["cause"]
    assert messages[0][0] == "vLLM pre-flight failed" and "no training time lost" in messages[0][1]
    assert (tmp_path / "vllm-preflight.log").is_file()


def test_vllm_preflight_on_one_gpu_loads_in_4_bit_unless_awq(tmp_path):
    env, popen, started, _messages = _vllm_block(tmp_path, "PREFLIGHT OK 1 'OK'\n", {"gpus": ["T4"]})
    env["vllm_preflight"](popen=popen, timeout=5)
    assert started[0][0][-2:] == ["1", "bitsandbytes"]
    env, popen, started, _messages = _vllm_block(tmp_path, "PREFLIGHT OK 1 'OK'\n",
                                                 {"gpus": ["T4"], "UNCENSORED": False, "TEACHER": "Qwen/Qwen3-14B-AWQ"})
    env["vllm_preflight"](popen=popen, timeout=5)
    assert started[0][0][-3:] == ["Qwen/Qwen3-14B-AWQ", "1", ""]


def test_prefetch_picks_the_first_editions_and_the_grader(tmp_path):
    cell = _cells()[3]
    env = {"TRAINER_MODULE": llm_train, "EDITIONS": ["minecraftism-thinking", "minecraftism", "qwen3-4b"],
           "UNCENSORED": True, "DISTILL": True, "ONPOLICY_STEPS": 60, "TEACHER": "auto", "shutil": shutil,
           "time": time, "TMP": str(tmp_path), "threading": threading, "importlib": __import__("importlib"),
           "FAST": False, "online": False}
    exec(cell[cell.index("def trainer_args"):cell.index("# ---- end prefetch")], env)  # noqa: S102
    repos = env["models_to_prefetch"]()
    assert repos == [llm_train.UNCENSORED_MODELS["minecraftism-thinking"], llm_train.UNCENSORED_TEACHER,
                     llm_train.UNCENSORED_MODELS["minecraftism"]]
    got = []
    disk = types.SimpleNamespace(free=100e9)
    env["shutil"] = types.SimpleNamespace(disk_usage=lambda _path: disk)
    assert env["prefetch_models"](repos[:2], download=lambda repo, **kw: got.append((repo, kw))) == repos[:2]
    assert "*.gguf" in got[0][1]["ignore_patterns"]
    disk.free = 10e9  # not enough room: the trainer downloads it itself later
    assert env["prefetch_models"](repos, download=lambda repo, **kw: got.append(repo)) == []


def test_without_vllm_the_slow_teacher_is_skipped_by_default():
    settings = _cells()[1]
    assert "SLOW_TEACHER = False" in settings and "VLLM_PREFLIGHT = True" in settings
    assert "MAX_HOURS = 11.75" in settings and "THE 12 HOURs" in settings
    cell = _cells()[6]
    gate = cell[cell.index('if DISTILL and "VLLM_THREAD" in globals():'):cell.index('if MODE == "parallel" and not HF_REPO')]
    for slow, expected in ((False, "--data x --distill-minutes 0"), (True, "--data x")):
        env = {"DISTILL": True, "VLLM_THREAD": threading.Thread(target=lambda: None), "VLLM_READY": threading.Event(),
               "VLLM_STATE": {"cause": "CUDA out of memory"}, "SLOW_TEACHER": slow, "RUN_FLAGS": "--data x",
               "VLLM_PYTHON": "/p"}
        exec(gate, env)  # noqa: S102
        assert env["RUN_FLAGS"] == expected
    # a pre-flight that is still running when the training wants the T4s is stopped first
    stop, done = threading.Event(), threading.Event()
    slow_thread = threading.Thread(target=lambda: stop.wait(5) and done.set())
    slow_thread.start()
    env = {"DISTILL": True, "VLLM_THREAD": slow_thread, "VLLM_READY": threading.Event(), "VLLM_ABORT": stop,
           "VLLM_WAIT": 0.1, "SLOW_TEACHER": False, "RUN_FLAGS": "", "VLLM_PYTHON": "/p", "VLLM_STATE": {}}
    exec(gate, env)  # noqa: S102
    assert stop.is_set() and done.is_set() and "--vllm-python" not in env["RUN_FLAGS"]
