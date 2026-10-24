"""Kaggle folders: /kaggle/working keeps what is needed again, /kaggle/tmp gets the throwaway files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from pymc_bot import llm_train

NOTEBOOK = Path(__file__).resolve().parents[1] / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"


def test_no_scratch_disk_off_kaggle_keeps_the_old_places(tmp_path, monkeypatch):
    monkeypatch.delenv("PYMC_SCRATCH", raising=False)
    if llm_train.scratch_root() is not None:  # pragma: no cover - only on Kaggle itself
        return
    assert llm_train.scratch_for(tmp_path, "checkpoints") == tmp_path
    assert llm_train.lock_file(tmp_path / "a.json") == str(tmp_path / "a.json") + ".lock"
    assert llm_train._rows_file(tmp_path).parent == tmp_path


def test_throwaway_files_go_to_the_scratch_disk(tmp_path, monkeypatch):
    scratch = tmp_path / "kaggle-tmp"
    monkeypatch.setenv("PYMC_SCRATCH", str(scratch))
    out = tmp_path / "working" / "minecraftism"
    assert llm_train.scratch_root() == scratch
    checkpoints = llm_train.scratch_for(out, "checkpoints")
    assert checkpoints.parent == scratch / "checkpoints" and checkpoints.name.startswith("minecraftism-")
    assert llm_train.scratch_for(tmp_path / "elsewhere" / "minecraftism", "checkpoints") != checkpoints
    lock = Path(llm_train.lock_file(tmp_path / "working" / "cache" / "abc.json"))
    assert lock.parent == scratch / "locks" and lock.suffix == ".lock"
    args = argparse.Namespace(gguf_scratch=None, tools_dir=None)
    assert llm_train.gguf_scratch(args, out) == scratch / "gguf" / "minecraftism"
    assert llm_train.tools_dir(args, out) == scratch / "gguf"
    kept = argparse.Namespace(gguf_scratch="/x/gguf", tools_dir=str(tmp_path / "working" / "tools"))
    assert llm_train.gguf_scratch(kept, out) == Path("/x/gguf")
    assert llm_train.tools_dir(kept, out) == tmp_path / "working" / "tools"


def test_both_gpus_hand_over_through_the_scratch_disk(tmp_path, monkeypatch):
    scratch = tmp_path / "kaggle-tmp"
    monkeypatch.setenv("PYMC_SCRATCH", str(scratch))
    monkeypatch.setenv("PYMC_RUN_ID", "folders")
    out = tmp_path / "working" / "minecraftism"
    out.mkdir(parents=True)
    llm_train.share_rows(out, [{"messages": []}], [], "/data")
    assert llm_train.wait_for_rows(out, timeout=1, poll=0.01)["rows"] == [{"messages": []}]
    assert list(out.iterdir()) == []  # nothing left in the kept output folder
    assert llm_train._distill_file(out, 1).parent == llm_train._rows_file(out).parent
    assert str(llm_train._rows_file(out)).startswith(str(scratch / "handover"))


def test_shared_cache_keeps_data_but_its_locks_go_to_scratch(tmp_path, monkeypatch):
    scratch = tmp_path / "kaggle-tmp"
    monkeypatch.setenv("PYMC_SCRATCH", str(scratch))
    cache = tmp_path / "working" / "cache"
    args = argparse.Namespace(stream_cache=str(cache))
    assert llm_train.cached(args, "rows", lambda: [1, 2]) == [1, 2]
    llm_train.save_teacher_answers(args, "t", {"k": "v"})
    kept = sorted(p.suffix for p in cache.iterdir())
    assert kept == [".json", ".json"]  # the data and the teacher answers - no .lock / .tmp files
    assert list((scratch / "locks").glob("*.lock"))


def test_trainer_uses_the_folders_everywhere():
    source = Path(llm_train.__file__).read_text(encoding="utf-8")
    assert source.count("llama_cpp_tools(tools_dir(args, out))") == 2
    assert source.count('output_dir=str(scratch_for(out, "checkpoints"))') == 2
    assert 'mkdtemp(prefix="pymc-vllm-", dir=' in source
    assert llm_train.build_parser().parse_args(["--tools-dir", "/k/w/tools"]).tools_dir == "/k/w/tools"


def test_notebook_keeps_needed_files_in_working_and_trash_in_tmp():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert 'WORK, TMP = "/kaggle/working", "/kaggle/tmp"' in code
    assert 'CACHE = f"{WORK}/cache"' in code and 'TOOLS = f"{WORK}/tools"' in code
    for env in ('"PYMC_SCRATCH": TMP', '"HF_HUB_CACHE": f"{TMP}/hf-cache"', '"TMPDIR": f"{TMP}/tmp"',
                '"UV_CACHE_DIR": f"{TMP}/uv-cache"'):
        assert env in code
    assert 'VLLM_PYTHON = f"{TMP}/vllm/bin/python"' in code and 'SYSTEM_FILE = f"{TMP}/system_prompt.txt"' in code
    assert 'download("pymc-minecraft-llm-kit.zip", f"{TMP}/pymc-minecraft-llm-kit.zip")' in code
    assert 'archive.extractall("/kaggle/working/kit")' in code  # the data itself is needed every run
    assert 'with open(f"{WORK}/vllm-install.log"' in code  # logs are kept
    assert code.count('cwd=f"{TMP}/run"') == 2  # trainer + CPU export: Unsloth's compiled cache stays out of WORK
    code_only = "\n".join(line.split("#")[0] for line in code.splitlines())
    assert not re.search(r"""["' ]/tmp/""", code_only), "a throwaway path outside /kaggle/tmp"
