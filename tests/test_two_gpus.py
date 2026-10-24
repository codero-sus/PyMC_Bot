"""Speed on Kaggle's two T4s: one edition per T4 or both T4s on one edition (DDP), a session deadline that stops
training before Kaggle's 12 h cut-off, one export at a time, cleanup after upload and the notebook's auto exit."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from pymc_bot import llm_dataset as ds
from pymc_bot import llm_train

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "pynb/PyMC_Minecraft_LLM_Kaggle.ipynb"


def _args(argv=()):
    return llm_train.build_parser().parse_args(list(argv))


def test_parser_has_the_speed_flags():
    args = _args(["--deadline", "1700000000", "--reserve-minutes", "90", "--export-lock", "/tmp/x.lock",
                  "--clean-after-upload", "--group-by-length", "--loader-workers", "2"])
    assert args.deadline == 1700000000 and args.reserve_minutes == 90 and args.export_lock == "/tmp/x.lock"
    assert args.clean_after_upload and args.group_by_length and args.loader_workers == 2
    plain = _args()
    assert plain.deadline == 0 and plain.export_lock is None and not plain.group_by_length and plain.loader_workers == 0


def test_ddp_halves_gradient_accumulation_for_the_same_effective_batch():
    preset = {"accum": 4}
    one = llm_train.speed_options(_args(), preset, world=1, text=True)
    two = llm_train.speed_options(_args(["--group-by-length", "--loader-workers", "2"]), preset, world=2, text=True)
    assert one["gradient_accumulation_steps"] == 4 and "ddp_find_unused_parameters" not in one
    assert two["gradient_accumulation_steps"] == 2 and two["ddp_find_unused_parameters"] is False
    assert two["group_by_length"] is True and two["dataloader_num_workers"] == 2 and two["dataloader_pin_memory"]
    assert "group_by_length" not in one and one["dataloader_num_workers"] == 0
    vision = llm_train.speed_options(_args(["--group-by-length"]), {"accum": 8}, world=2, text=False)
    assert vision["gradient_accumulation_steps"] == 4 and "group_by_length" not in vision  # needs input_ids
    assert "dataset_num_proc" not in vision


def test_dist_info_reads_torchrun(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert llm_train.dist_info() == (0, 1)
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert llm_train.dist_info() == (1, 2)


class _Control:
    should_training_stop = False


class _State:
    global_step, max_steps = 120, 400


@pytest.fixture()
def fake_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(TrainerCallback=object))


def test_deadline_stops_training_in_time_for_the_export(fake_transformers, capsys):
    args = _args(["--deadline", "10000", "--reserve-minutes", "60"])
    now = {"t": 10000 - 3700}
    callback = llm_train.deadline_callback(args, clock=lambda: now["t"])
    control = _Control()
    callback.on_step_end(None, _State(), control)
    assert not control.should_training_stop  # 61 min left: keep training
    now["t"] = 10000 - 3500
    callback.on_step_end(None, _State(), control)
    assert control.should_training_stop and args.stopped_early == {"step": 120, "of": 400}
    assert "time limit: stopping at step 120/400" in capsys.readouterr().out
    assert llm_train.deadline_callback(_args()) is None


def test_length_grouping_failure_retries_without_it(capsys):
    class Broken:
        args = types.SimpleNamespace(group_by_length=True)
        state = types.SimpleNamespace(global_step=0)

        def train(self):
            raise ValueError("cannot group")

    class Fine:
        def train(self):
            return "trained"

    seen = []

    def rebuild(exc):
        seen.append(str(exc))
        return Fine()

    assert llm_train.run_training(Broken(), rebuild, world=1) == "trained" and seen == ["cannot group"]
    with pytest.raises(ValueError):  # under DDP both ranks must fail the same way, no silent retry
        llm_train.run_training(Broken(), rebuild, world=2)
    late = Broken()
    late.state = types.SimpleNamespace(global_step=50)
    with pytest.raises(ValueError):  # hours in: a real error, do not train again
        llm_train.run_training(late, rebuild, world=1)
    with pytest.raises(ValueError):  # nothing safer to try
        llm_train.run_training(Broken(), lambda _exc: None, world=1)


def test_ddp_rows_are_prepared_once_and_shared(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "abc")
    rows, eval_rows = [{"messages": [1]}], [{"messages": [2]}]
    llm_train.share_rows(tmp_path, rows, eval_rows, "/data")
    assert llm_train.wait_for_rows(tmp_path) == {"rows": rows, "eval_rows": eval_rows, "data_dir": "/data"}
    llm_train.share_rows(tmp_path, None)
    assert llm_train.wait_for_rows(tmp_path) is None
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "other-run")  # a stale file from another run is never read
    with pytest.raises(TimeoutError):
        llm_train.wait_for_rows(tmp_path, timeout=0.05, poll=0.01)


@pytest.fixture()
def data(tmp_path):
    out = tmp_path / "data"
    ds.build_dataset(out, rule_turns=120, mob_rounds=1, log=lambda _m: None)
    return out


def test_ddp_rank0_prepares_and_hands_the_rows_over(data, tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "r1")
    seen = {}
    monkeypatch.setattr(llm_train, "train", lambda args, preset, rows, eval_rows: seen.update(rows=len(rows)) or 0)
    out = tmp_path / "out"
    assert llm_train.main(["--data", str(data), "--out", str(out), "--general", "0", "--minecraft-qa", "0"]) == 0
    shared = llm_train.wait_for_rows(out)
    assert shared and len(shared["rows"]) == seen["rows"] and shared["data_dir"] == str(data)

    monkeypatch.setenv("RANK", "1")  # the second T4 trains on exactly those rows, without reading the data again
    seen.clear()
    monkeypatch.setattr(llm_train, "prepare_and_train", lambda *_a, **_k: pytest.fail("rank 1 prepared data"))
    stdout = sys.stdout
    try:
        assert llm_train.main(["--data", str(data), "--out", str(out)]) == 0
    finally:
        sys.stdout = stdout
    assert seen["rows"] == len(shared["rows"])


def test_ddp_rank0_that_stops_early_releases_the_other_gpu(data, tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "r2")
    out = tmp_path / "out"
    assert llm_train.main(["--dry-run", "--data", str(data), "--out", str(out), "--general", "0"]) == 0
    assert llm_train.wait_for_rows(out) is None

    def boom(*_a, **_k):
        raise RuntimeError("streaming failed")

    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "r3")
    monkeypatch.setattr(llm_train, "prepare_and_train", boom)
    with pytest.raises(RuntimeError):
        llm_train.main(["--data", str(data), "--out", str(out)])
    assert llm_train.wait_for_rows(out) is None


def test_an_edition_that_cannot_finish_before_the_deadline_waits(data, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(llm_train, "train", lambda *_a: pytest.fail("trained past the deadline"))
    code = llm_train.main(["--data", str(data), "--out", str(tmp_path / "out"), "--general", "0",
                           "--deadline", str(time.time() + 30 * 60), "--reserve-minutes", "60"])
    assert code == 3 and "train it in the next session" in capsys.readouterr().out


def test_export_lock_lets_one_edition_export_at_a_time(tmp_path):
    lock = str(tmp_path / "export.lock")
    order: list[str] = []
    first = llm_train.export_lock(lock).__enter__()

    def second():
        with llm_train.export_lock(lock):
            order.append("second")

    thread = threading.Thread(target=second)
    thread.start()
    time.sleep(0.3)
    order.append("first done")
    first.__exit__(None, None, None)
    thread.join(5)
    assert order == ["first done", "second"]
    with llm_train.export_lock(None):  # no lock file: nothing to wait for
        pass


def test_clean_after_upload_keeps_what_is_not_on_the_hub(tmp_path):
    out = tmp_path
    (out / "m.Q4_K_M.gguf").write_bytes(b"x")
    (out / "lora").mkdir()
    (out / "safetensors").mkdir()
    (out / "README.md").write_text("card")
    assert llm_train.clean_uploaded(out, {"hub_repo": "me/x-GGUF", "safetensors": "safetensors"}) == \
        ["m.Q4_K_M.gguf", "lora"]
    assert (out / "safetensors").is_dir() and (out / "README.md").is_file()  # its upload failed: kept
    assert llm_train.clean_uploaded(out, {"hub_repo": "me/x-GGUF", "safetensors": "safetensors",
                                          "safetensors_uploaded": True}) == ["safetensors"]
    (out / "n.gguf").write_bytes(b"x")
    assert llm_train.clean_uploaded(out, {"safetensors_uploaded": True}) == [] and (out / "n.gguf").exists()


def _notebook():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return nb, "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_notebook_uses_both_t4s_and_the_deadline():
    nb, code = _notebook()
    assert 'GPU_MODE = "parallel"' in code and "FAST = True" in code and "MAX_HOURS = 11.5" in code
    assert '{"merge": "ddp", "merged": "ddp"}.get(GPU_MODE, GPU_MODE)' in code  # merge = both T4s on every edition
    assert 'if MODE in ("auto", "parallel"):' in code  # parallel by default, one edition -> both T4s merge on it
    assert "used = list(range(n_gpus))" in code  # the last edition left takes both T4s when the other is free
    assert '"accelerate.commands.launch", "--multi_gpu"' in code and '"--mixed_precision=fp16"' in code
    assert '"torch.distributed.run", "--standalone"' in code and '"CUDA_VISIBLE_DEVICES"' in code  # no accelerate
    assert '"PYMC_RUN_ID": f"{edition}-{time.time_ns()}"' in code
    assert 'SLOTS = {"parallel": [[0], [1]], "ddp": [[0, 1]]}' in code
    assert "training it again on one T4" in code  # DDP problem -> same edition on one T4
    assert '--deadline {DEADLINE:.0f} --export-lock {TMP}/locks/export.lock --tools-dir {TOOLS}' in code
    assert 'FLAGS += " --token-budget 12288 --loader-workers 2 --eval-limit 60" if FAST' in code
    assert '"--vllm-python", "--tools-dir"))' in code  # an older trainer is replaced by the newest
    assert "!python" not in "".join(nb["cells"][6]["source"])


def _exit_cell():
    nb, _code = _notebook()
    cell = "".join(nb["cells"][-1]["source"])
    assert nb["cells"][-1]["cell_type"] == "code" and "auto exit" in cell
    return cell


def _run_exit(monkeypatch, reports, run_type, auto_exit=True):
    calls: list = []
    fake_os = types.SimpleNamespace(environ={"KAGGLE_KERNEL_RUN_TYPE": run_type}, _exit=lambda code: calls.append(("exit", code)))
    fake_subprocess = types.SimpleNamespace(Popen=lambda command, **kw: calls.append(("popen", command)))
    env = {"os": fake_os, "subprocess": fake_subprocess, "time": types.SimpleNamespace(sleep=lambda _s: None),
           "READY": list(reports), "report_of": reports.get, "AUTO_EXIT": auto_exit, "DISCORD_STATE": {},
           "discord": lambda text, title="", level="info", ping=False: calls.append(("discord", title)),
           "discord_flush": lambda seconds=20: None}
    exec(_exit_cell(), env)
    assert env["DISCORD_STATE"]["finished"] is True  # the server closing after the last cell is expected
    told = [call for call in calls if call[0] == "discord"]
    assert len(told) == 1  # Discord hears how the session ends, every time
    return [call for call in calls if call[0] != "discord"]


def test_auto_exit_ends_an_interactive_session_only_when_everything_is_uploaded(monkeypatch, capsys):
    done = {"hub_repo": "me/a-GGUF", "safetensors": "safetensors", "safetensors_uploaded": True}
    calls = _run_exit(monkeypatch, {"a": done, "b": {"hub_repo": "me/b-GGUF"}}, "Interactive")
    assert calls == [("popen", ["bash", "-c", "sleep 1; kill -9 -1"]), ("exit", 0)]
    assert _run_exit(monkeypatch, {"a": done, "b": {}}, "Interactive") == []  # b is only local
    assert "not ending the session" in capsys.readouterr().out
    assert _run_exit(monkeypatch, {"a": {**done, "safetensors_uploaded": False}}, "Interactive") == []
    assert _run_exit(monkeypatch, {"a": done}, "Batch") == []  # Save & Run All ends by itself, Output kept
    assert _run_exit(monkeypatch, {"a": done}, "Interactive", auto_exit=False) == []
    assert _run_exit(monkeypatch, {}, "Interactive") == []  # nothing trained: nothing to end for


def _scheduler(mode, editions, slow=0.05):
    """Run the notebook's worker code with fake training to see which T4s each edition gets."""
    import threading
    import time as _time
    _nb, code = _notebook()
    body = code[code.index("BUSY = set()"):code.index("\n\n\nthreads = [")]
    calls = []

    def run_edition(edition, used, extra=""):
        calls.append((edition, tuple(used)))
        _time.sleep(slow if len(calls) > 1 else slow * 3)
        return 0, ""

    env = {"MODE": mode, "n_gpus": 2, "queue": list(editions), "queue_lock": threading.Lock(), "FAST": True,
           "run_edition": run_edition, "failed": lambda code, edition: False, "note": lambda *a: None,
           "export_job": lambda edition: "/nonexistent", "CODES": {}, "clean_up": lambda edition: None,
           "EXPORTS": [], "threading": threading, "os": __import__("os"), "run_export": lambda edition: None}
    exec(body, env)  # noqa: S102 - the notebook's own scheduler
    slots = {"parallel": [[0], [1]], "ddp": [[0, 1]]}[mode]
    threads = [threading.Thread(target=env["worker"], args=(slot,)) for slot in slots]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    return calls


def test_parallel_is_the_default_and_never_leaves_a_t4_idle():
    # two editions: one per T4, at the same time
    calls = _scheduler("parallel", ["vl", "txt"])
    assert sorted(used for _, used in calls) == [(0,), (1,)]
    # three editions: the last one starts when the other T4 is still busy -> one T4 each
    calls = _scheduler("parallel", ["vl_think", "vl", "txt"])
    assert len(calls) == 3 and all(len(used) == 1 for _, used in calls)
    # merge mode: both T4s on every edition
    assert all(used == (0, 1) for _, used in _scheduler("ddp", ["txt", "vl"]))
