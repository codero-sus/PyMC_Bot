"""Safetensors first, then every GGUF quant made from them: compact 16-bit weights (a 4B model stays 4B-sized),
llama.cpp conversion + quantisation with each file uploaded as soon as it is ready, and the tokenizer/config
essentials in both Hugging Face repos."""

from __future__ import annotations

import io
import json
import struct
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pytest

from pymc_bot import llm_train

DTYPES = {"float32": "F32", "float16": "F16"}


def write_st(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        raw = array.tobytes()
        header[name] = {"dtype": DTYPES[array.dtype.name], "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"".join(blobs))


def read_st(path: Path) -> dict[str, tuple[str, list[int], bytes]]:
    header, start = llm_train.read_safetensors_header(path)
    data = path.read_bytes()
    offsets = sorted(info["data_offsets"] for name, info in header.items() if name != "__metadata__")
    assert all(a[1] == b[0] for a, b in zip(offsets, offsets[1:], strict=False))  # packed, no gaps
    assert len(data) == start + (offsets[-1][1] if offsets else 0)
    return {name: (info["dtype"], info["shape"], data[start + info["data_offsets"][0]:start + info["data_offsets"][1]])
            for name, info in header.items() if name != "__metadata__"}


EMBED = np.arange(12, dtype=np.float16).reshape(4, 3)


def write_model(folder: Path, tied: bool = True, same_head: bool = True, dtype: str = "float16") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps({"tie_word_embeddings": tied, "torch_dtype": dtype}))
    write_st(folder / "model-00001-of-00002.safetensors",
             {"model.embed_tokens.weight": EMBED, "model.norm.weight": np.array([1.0, -2.5, 0.5], dtype=np.float32)})
    head = EMBED if same_head else EMBED + 1
    write_st(folder / "model-00002-of-00002.safetensors",
             {"model.layers.0.mlp.weight": np.ones((2, 3), dtype=np.float16), "lm_head.weight": head})
    index = {"metadata": {"total_size": 999}, "weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "model.norm.weight": "model-00001-of-00002.safetensors",
        "model.layers.0.mlp.weight": "model-00002-of-00002.safetensors",
        "lm_head.weight": "model-00002-of-00002.safetensors"}}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index))


def test_compact_drops_the_tied_lm_head_copy_and_float32(tmp_path):
    write_model(tmp_path)
    stats = llm_train.compact_safetensors(tmp_path, log=lambda *_: None)
    assert stats["tied_dropped"] == ["lm_head.weight"] and stats["float32_cast"] == 1 and stats["dtype"] == "F16"
    assert stats["params"] == 12 + 3 + 6  # the 12-value lm_head copy is gone
    first = read_st(tmp_path / "model-00001-of-00002.safetensors")
    second = read_st(tmp_path / "model-00002-of-00002.safetensors")
    assert "lm_head.weight" not in second and second["model.layers.0.mlp.weight"][0] == "F16"
    assert first["model.embed_tokens.weight"][2] == EMBED.tobytes()  # untouched bytes
    dtype, shape, raw = first["model.norm.weight"]
    assert dtype == "F16" and shape == [3] and np.frombuffer(raw, dtype="<f2").tolist() == [1.0, -2.5, 0.5]
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert "lm_head.weight" not in index["weight_map"] and index["metadata"]["total_size"] == (12 + 3 + 6) * 2
    assert not list(tmp_path.glob("*.tmp"))


def test_compact_keeps_a_head_that_is_not_a_copy(tmp_path):
    write_model(tmp_path / "different", same_head=False)
    assert llm_train.compact_safetensors(tmp_path / "different", log=lambda *_: None)["tied_dropped"] == []
    write_model(tmp_path / "untied", tied=False)
    stats = llm_train.compact_safetensors(tmp_path / "untied", log=lambda *_: None)
    assert stats["tied_dropped"] == [] and stats["params"] == 12 + 3 + 6 + 12


def test_compact_turns_float32_into_bfloat16_when_the_model_is_bf16(tmp_path):
    write_model(tmp_path, dtype="bfloat16")
    llm_train.compact_safetensors(tmp_path, log=lambda *_: None)
    dtype, _shape, raw = read_st(tmp_path / "model-00001-of-00002.safetensors")["model.norm.weight"]
    assert dtype == "BF16" and np.frombuffer(raw, dtype="<u2").tolist() == [0x3F80, 0xC020, 0x3F00]
    assert np.frombuffer(llm_train._to_16bit(np.array([np.nan], dtype="<f4").tobytes(), "BF16"),
                         dtype="<u2").tolist() == [0x7FC0]


def test_pick_gguf_never_takes_the_f16_intermediate(tmp_path):
    files = {"unsloth.F16.gguf": 8000, "unsloth.Q4_K_M.gguf": 2500, "unsloth.Q8_0.gguf": 4300, "mmproj-F16.gguf": 600}
    for name, size in files.items():
        (tmp_path / name).write_bytes(b"x" * size)
    found = sorted(str(tmp_path / name) for name in files)
    old_pick = sorted(f for f in found if "mmproj" not in f)[0]
    assert old_pick.endswith("unsloth.F16.gguf")  # what the old sorted(...)[0] picked
    assert llm_train.pick_gguf(found, "q4_k_m").endswith("unsloth.Q4_K_M.gguf")
    assert llm_train.pick_gguf(found, "Q8_0").endswith("unsloth.Q8_0.gguf")
    assert llm_train.pick_gguf(found, "q6_k") is None  # never a wrong quant under a new name
    assert llm_train.pick_gguf(found, "f16").endswith("unsloth.F16.gguf")
    odd = [str(tmp_path / "unsloth.F16.gguf"), str(tmp_path / "model.gguf")]
    (tmp_path / "model.gguf").write_bytes(b"x")
    assert llm_train.pick_gguf(odd, "q4_k_m").endswith("model.gguf")


def _parse(argv):
    return llm_train.build_parser().parse_args(argv)


def test_quant_list_puts_the_main_quant_first():
    assert llm_train.gguf_quants(_parse([])) == ["q4_k_m"]
    assert llm_train.gguf_quants(_parse(["--hub-repo", "auto"])) == ["q4_k_m", "q8_0", "q6_k", "q5_k_m", "q3_k_m"]
    assert llm_train.gguf_quants(_parse(["--gguf-quants", "Q8_0, q4_k_m,q8_0"])) == ["q4_k_m", "q8_0"]
    assert llm_train.gguf_quants(_parse(["--hub-repo", "auto", "--gguf-quants", "none"])) == ["q4_k_m"]
    assert llm_train.gguf_quants(_parse(["--gguf", "none", "--hub-repo", "auto"])) == []
    assert _parse([]).compact is True and _parse(["--no-compact"]).compact is False


class FakeRun:
    """Stands in for subprocess.run: the converter and llama-quantize write files of a believable size."""

    SIZES = {"F16": 8000, "Q8_0": 4300, "Q6_K": 3300, "Q5_K_M": 2900, "Q4_K_M": 2500, "Q3_K_M": 2000}

    def __init__(self, fail: str = "", mmproj_name: str | None = None):
        self.calls: list[list[str]] = []
        self.fail = fail
        self.mmproj_name = mmproj_name

    def __call__(self, command, **kwargs):
        command = [str(c) for c in command]
        self.calls.append(command)
        if self.fail and any(self.fail in part for part in command):
            return types.SimpleNamespace(returncode=1, stdout="", stderr=f"boom: {self.fail}")
        if command[1:2] == ["--help"]:
            return types.SimpleNamespace(returncode=1, stdout="usage: llama-quantize ...", stderr="")
        if "--outfile" in command:
            target = Path(command[command.index("--outfile") + 1])
            if "--mmproj" in command:
                target = target.with_name(self.mmproj_name or target.name)
                target.write_bytes(b"m" * 600)
            else:
                target.write_bytes(b"f" * self.SIZES["F16"])
        elif Path(command[0]).name == "llama-quantize":
            Path(command[2]).write_bytes(b"q" * self.SIZES[command[3]])
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _tools(tmp_path: Path) -> tuple[Path, Path]:
    convert = tmp_path / "llama.cpp" / "convert_hf_to_gguf.py"
    quantize = tmp_path / "llama.cpp" / "build" / "bin" / "llama-quantize"
    quantize.parent.mkdir(parents=True)
    convert.write_text("# converter")
    quantize.write_text("# binary")
    return convert, quantize


def test_make_ggufs_converts_once_then_quantises_each_and_hands_it_over(tmp_path):
    run, seen = FakeRun(mmproj_name="weird-name.gguf"), []
    result = llm_train.make_ggufs(tmp_path / "st", tmp_path / "work", "minecraftism-vl-4b", ["q4_k_m", "q8_0"],
                                  vision=True, tools=_tools(tmp_path),
                                  on_file=lambda path, quant: seen.append((path.name, quant, path.stat().st_size)),
                                  run=run)
    assert seen == [("mmproj-minecraftism-vl-4b.F16.gguf", "MMPROJ", 600),
                    ("minecraftism-vl-4b.Q4_K_M.gguf", "Q4_K_M", 2500), ("minecraftism-vl-4b.Q8_0.gguf", "Q8_0", 4300)]
    assert [f["quant"] for f in result["files"]] == ["Q4_K_M", "Q8_0"] and result["mmproj"].startswith("mmproj-")
    assert run.calls[0][-4:] == ["--outfile", str(tmp_path / "work/minecraftism-vl-4b.F16.gguf"), "--outtype", "f16"]
    assert "--mmproj" in run.calls[1]
    assert run.calls[2][1:] == [str(tmp_path / "work/minecraftism-vl-4b.F16.gguf"),
                                str(tmp_path / "work/minecraftism-vl-4b.Q4_K_M.gguf"), "Q4_K_M"]
    assert not (tmp_path / "work/minecraftism-vl-4b.F16.gguf").exists()  # intermediate cleaned up


def test_make_ggufs_reports_a_failed_tool(tmp_path):
    with pytest.raises(RuntimeError, match="F16 -> Q8_0 failed: boom"):
        llm_train.make_ggufs(tmp_path / "st", tmp_path / "work", "m", ["q8_0"], vision=False, tools=_tools(tmp_path),
                             on_file=lambda *_: None, run=FakeRun(fail="Q8_0"))


def test_llama_cpp_tools_uses_an_existing_copy(tmp_path, monkeypatch):
    convert, quantize = _tools(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path / "llama.cpp"))
    assert llm_train.llama_cpp_tools(tmp_path / "scratch", run=FakeRun(), log=lambda *_: None) == \
        (convert.resolve(), quantize.resolve())


def _tarball(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _no_local_llama(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(llm_train.Path, "home", classmethod(lambda cls: tmp_path / "home"))


def _release_fakes(downloads):
    def fetch_json(url):
        assert "repos/ggml-org/llama.cpp/releases" in url
        return [{"tag_name": "b100", "assets": [{"name": "llama-b100-bin-win-cpu-x64.zip"}]},
                {"tag_name": "b99", "assets": [{"name": "llama-b99-bin-ubuntu-x64.tar.gz",
                                                "browser_download_url": "https://example/bin.tar.gz"}]}]

    def download(url, target):
        downloads.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        if "codeload" in url:
            target.write_bytes(_tarball({"llama.cpp-b99/convert_hf_to_gguf.py": b"# conv",
                                         "llama.cpp-b99/gguf-py/gguf/__init__.py": b""}))
        else:
            target.write_bytes(_tarball({"llama-b99/llama-quantize": b"bin", "llama-b99/libllama.so": b"so"}))
        return target

    return fetch_json, download


def test_llama_cpp_tools_downloads_the_newest_linux_build_and_its_source(tmp_path, monkeypatch):
    _no_local_llama(tmp_path, monkeypatch)
    downloads: list[str] = []
    fetch_json, download = _release_fakes(downloads)
    convert, quantize = llm_train.llama_cpp_tools(tmp_path / "scratch", run=FakeRun(), fetch_json=fetch_json,
                                                  download=download, log=lambda *_: None)
    assert downloads == ["https://example/bin.tar.gz", "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/tags/b99"]
    assert convert.name == "convert_hf_to_gguf.py" and "b99" in str(convert)
    assert quantize.name == "llama-quantize" and quantize.parent.name == "llama-b99"


def test_llama_cpp_tools_compiles_quantize_when_the_prebuilt_one_cannot_start(tmp_path, monkeypatch):
    _no_local_llama(tmp_path, monkeypatch)
    fetch_json, download = _release_fakes([])
    commands: list[list[str]] = []

    def run(command, **kwargs):
        command = [str(c) for c in command]
        commands.append(command)
        if command[1:2] == ["--help"]:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="version `GLIBC_2.38' not found")
        if command[:2] == ["cmake", "--build"]:
            binary = Path(command[2]) / "bin" / "llama-quantize"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("built")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    _convert, quantize = llm_train.llama_cpp_tools(tmp_path / "scratch", run=run, fetch_json=fetch_json,
                                                   download=download, log=lambda *_: None)
    assert quantize.read_text() == "built" and quantize.parent.parent.name == "build"
    assert ["--target", "llama-quantize"] == commands[-1][3:5] and "-DLLAMA_CURL=OFF" in commands[-2]


# --------------------------------------------------------------------------- the whole finish() order
class FakeModel:
    def __init__(self):
        self.calls: list[tuple] = []

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_bytes(b"lora")

    def save_pretrained_merged(self, path, tokenizer, save_method=""):
        self.calls.append(("merged", save_method))
        write_model(Path(path))
        for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
            (Path(path) / name).write_text("{}")

    def save_pretrained_gguf(self, path, tokenizer, quantization_method=None):
        self.calls.append(("gguf", quantization_method))
        folder = Path(path)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "unsloth.F16.gguf").write_bytes(b"f" * 8000)  # sorts first: the old size bug
        for quant in quantization_method:
            (folder / f"unsloth.{quant.upper()}.gguf").write_bytes(b"q" * FakeRun.SIZES[quant.upper()])


class FakeTokenizer:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


@pytest.fixture()
def hub(monkeypatch):
    class Events(list):
        api: type
    events = Events()

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "me"}

        def create_repo(self, repo, **kwargs):
            events.append(("create", repo))

        def upload_file(self, **kwargs):
            path = Path(kwargs["path_or_fileobj"])
            if getattr(HfApi, "fail_files", False):
                raise OSError("network down")
            events.append(("file", kwargs["repo_id"], kwargs["path_in_repo"], path.stat().st_size))

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
            events.append(("folder", kwargs["repo_id"], files, kwargs.get("ignore_patterns")))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    events.api = HfApi
    return events


def _finish(tmp_path, monkeypatch, argv, model=None):
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(llm_train, "llama_cpp_tools", lambda scratch, **_: _tools(tmp_path / "tools"))
    args = llm_train.build_parser().parse_args(["--preset", "minecraftism", "--eval-limit", "0", "--out",
                                                str(tmp_path / "out"), "--gguf-scratch", str(tmp_path / "work")] + argv)
    args.system_prompt = "You are Minecraftism."
    args.data_dir = str(tmp_path)
    model = model or FakeModel()
    code = llm_train.finish(args, llm_train.PRESETS["minecraftism"], model, FakeTokenizer(), [], [], 0.5, 0.0,
                            generate=None, for_inference=lambda: None)
    return code, model, json.loads((tmp_path / "out/pymc_model.json").read_text())


def test_finish_uploads_safetensors_then_each_gguf_then_the_gguf_repo(tmp_path, monkeypatch, hub):
    code, model, report = _finish(tmp_path, monkeypatch, ["--hub-repo", "auto"])
    out = tmp_path / "out"
    assert code == 0 and model.calls == [("merged", "merged_16bit")]  # Unsloth's GGUF exporter is not needed
    kinds = [(e[0], e[1]) for e in hub]
    assert kinds == [("create", "me/Minecraftism-4B"), ("folder", "me/Minecraftism-4B"),
                     ("create", "me/Minecraftism-4B-GGUF")] + [("file", "me/Minecraftism-4B-GGUF")] * 4 + \
                    [("create", "me/Minecraftism-4B-GGUF"), ("folder", "me/Minecraftism-4B-GGUF")]
    files = [e[2] for e in hub if e[0] == "file"]
    stem = llm_train.PRESETS["minecraftism"]["file"]
    assert files == [f"{stem}.{q}.gguf" for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q3_K_M")]
    st_files = hub[1][2]
    assert "tokenizer.json" in st_files and "README.md" in st_files and "model.safetensors.index.json" in st_files
    gguf_files, ignore = hub[-1][2], hub[-1][3]
    main = report["gguf"]
    assert main.endswith(".Q4_K_M.gguf") and main in gguf_files and report["gguf_gb"] == 0.0
    assert {"tokenizer.json", "config.json", "chat_template.jinja", "Modelfile", "README.md", "pymc_model.json",
            "generation_config.json", "tokenizer_config.json"} <= set(gguf_files)
    assert not [f for f in gguf_files if f.endswith(".gguf") and f != main]  # the others went up one by one
    assert "stream_images/*" in ignore and "safetensors/*" in ignore
    assert [g["quant"] for g in report["ggufs"]] == ["Q4_K_M", "Q8_0", "Q6_K", "Q5_K_M", "Q3_K_M"]
    assert [g["uploaded"] for g in report["ggufs"]] == [False, True, True, True, True]
    assert report["gguf_route"] == "llama.cpp" and report["hub_repo"] == "me/Minecraftism-4B-GGUF"
    assert report["safetensors_uploaded"] is True and report["compact"]["tied_dropped"] == ["lm_head.weight"]
    assert report["params_b"] == 0.0 and "tokenizer.json" in report["essentials"]
    card = (out / "README.md").read_text()
    assert "| `Q8_0` |" in card and "ollama run hf.co/me/Minecraftism-4B-GGUF:Q8_0" in card
    assert "ollama run hf.co/me/Minecraftism-4B-GGUF:Q4_K_M" in card and "`tokenizer.json`" in card
    assert not (tmp_path / "work").exists()


def test_a_failed_quant_upload_keeps_the_file(tmp_path, monkeypatch, hub):
    hub.api.fail_files = True
    try:
        code, _model, report = _finish(tmp_path, monkeypatch, ["--hub-repo", "auto", "--gguf-quants", "q8_0"])
    finally:
        hub.api.fail_files = False
    assert code == 0 and (tmp_path / "out" / report["ggufs"][1]["file"]).is_file()
    assert report["ggufs"][1]["uploaded"] is False
    assert report["ggufs"][1]["file"] in hub[-1][2]  # the folder upload at the end takes it


def test_gguf_repo_failure_is_not_reported_as_uploaded(tmp_path, monkeypatch, hub):
    def broken(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr(llm_train, "publish", broken)
    code, _model, report = _finish(tmp_path, monkeypatch, ["--hub-repo", "auto", "--gguf-quants", "none"])
    assert code == 0 and "hub_repo" not in report and report["safetensors_uploaded"] is True


def test_without_safetensors_unsloth_makes_the_quants_by_name(tmp_path, monkeypatch, hub):
    code, model, report = _finish(tmp_path, monkeypatch, ["--safetensors", "none", "--gguf-quants", "q8_0"])
    out = tmp_path / "out"
    assert code == 0 and model.calls == [("gguf", ["q4_k_m", "q8_0"])]
    assert report["gguf_route"] == "unsloth" and [g["quant"] for g in report["ggufs"]] == ["Q4_K_M", "Q8_0"]
    assert (out / report["gguf"]).stat().st_size == 2500  # the Q4_K_M file, not the 8000-byte F16
    assert (out / report["ggufs"][1]["file"]).stat().st_size == 4300
    assert not list(out.glob("*F16*")) and not (out / "gguf").exists() and hub == []


def test_llama_cpp_failure_falls_back_to_unsloth(tmp_path, monkeypatch, hub):
    def no_tools(scratch, **_):
        raise RuntimeError("offline")

    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    args = llm_train.build_parser().parse_args(["--preset", "minecraftism", "--eval-limit", "0", "--out",
                                                str(tmp_path / "out"), "--gguf-scratch", str(tmp_path / "work")])
    args.system_prompt, args.data_dir = "", str(tmp_path)
    monkeypatch.setattr(llm_train, "llama_cpp_tools", no_tools)
    model = FakeModel()
    assert llm_train.finish(args, llm_train.PRESETS["minecraftism"], model, FakeTokenizer(), [], [], 0.5, 0.0,
                            generate=None, for_inference=lambda: None) == 0
    report = json.loads((tmp_path / "out/pymc_model.json").read_text())
    assert model.calls[-1] == ("gguf", ["q4_k_m"]) and report["gguf_route"] == "unsloth"


def test_local_import_takes_the_main_quant_not_the_biggest(tmp_path):
    from pymc_bot import llm_tools

    (tmp_path / "m.Q8_0.gguf").write_bytes(b"x" * 4300)
    (tmp_path / "m.Q4_K_M.gguf").write_bytes(b"x" * 2500)
    assert llm_tools.find_gguf(tmp_path).name == "m.Q8_0.gguf"  # no report: the biggest, as before
    (tmp_path / "pymc_model.json").write_text(json.dumps({"gguf": "m.Q4_K_M.gguf"}))
    assert llm_tools.find_gguf(tmp_path).name == "m.Q4_K_M.gguf"
