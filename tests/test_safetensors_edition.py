"""Every trained Minecraftism edition ships twice on Hugging Face: GGUF (Ollama / PyMC Bot) and safetensors
(the LoRA merged into the base, 16-bit, for transformers / vLLM / further fine-tuning), each in its own repo."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from pymc_bot import llm_tools, llm_train

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "pynb/PyMC_Minecraft_LLM_Kaggle.ipynb"


def test_safetensors_repo_sits_next_to_the_gguf_repo():
    assert llm_train.safetensors_repo("me/Minecraftism-4B-GGUF") == "me/Minecraftism-4B"
    assert llm_train.safetensors_repo("me/Minecraftism-VL-4B-Thinking-Uncensored-GGUF") == \
        "me/Minecraftism-VL-4B-Thinking-Uncensored"
    assert llm_train.safetensors_repo("me/my-model") == "me/my-model-safetensors"


def test_llm_import_swaps_a_safetensors_repo_for_its_gguf_twin():
    assert llm_tools.gguf_repo_for("me/Minecraftism-4B-Uncensored") == "me/Minecraftism-4B-Uncensored-GGUF"
    assert llm_tools.gguf_repo_for("me/Minecraftism-VL-4B") == "me/Minecraftism-VL-4B-GGUF"
    assert llm_tools.gguf_repo_for("me/Minecraftism-4B-GGUF") == "me/Minecraftism-4B-GGUF"
    assert llm_tools.gguf_repo_for("me/other-model") == "me/other-model"


def test_parser_exports_safetensors_by_default():
    args = llm_train.build_parser().parse_args([])
    assert args.safetensors == "16bit"
    assert llm_train.build_parser().parse_args(["--safetensors", "none"]).safetensors == "none"


class FakeModel:
    """Stands in for an Unsloth model: records what was saved and writes small files."""

    def __init__(self, write_weights: bool = True):
        self.write_weights = write_weights
        self.calls: list[tuple[str, str]] = []

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.safetensors").write_bytes(b"lora")

    def save_pretrained_merged(self, path: str, tokenizer, save_method: str = "") -> None:
        self.calls.append(("merged", save_method))
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "config.json").write_text("{}")
        if self.write_weights:
            (Path(path) / "model-00001-of-00002.safetensors").write_bytes(b"x" * 3000)
            (Path(path) / "model-00002-of-00002.safetensors").write_bytes(b"x" * 1000)


class FakeTokenizer:
    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "tokenizer.json").write_text("{}")


def test_export_safetensors_merges_in_16bit(tmp_path):
    model = FakeModel()
    size = llm_train.export_safetensors(model, FakeTokenizer(), tmp_path / "safetensors")
    assert model.calls == [("merged", "merged_16bit")]
    assert size == 0.0 and len(list((tmp_path / "safetensors").glob("*.safetensors"))) == 2
    with pytest.raises(RuntimeError):
        llm_train.export_safetensors(FakeModel(write_weights=False), FakeTokenizer(), tmp_path / "empty")


def test_safetensors_card_is_for_transformers_and_links_the_gguf():
    preset = llm_train.PRESETS["minecraftism"]
    report = {"quant": "Q4_K_M", "safetensors": "safetensors", "safetensors_gb": 8.05, "hub_repo": "me/Minecraftism-4B-GGUF",
              "hub_repo_safetensors": "me/Minecraftism-4B", "system_prompt": "You are Minecraftism."}
    card = llm_train.model_card("minecraftism", preset, report, "me/Minecraftism-4B", kind="safetensors")
    assert "library_name: transformers" in card and "- safetensors" in card and "- gguf" not in card
    assert "AutoModelForCausalLM" in card and "vllm serve me/Minecraftism-4B" in card
    assert "https://huggingface.co/me/Minecraftism-4B-GGUF" in card and "llm import --hf me/Minecraftism-4B-GGUF" in card
    assert "~8.05 GB of 16-bit weights" in card and "(safetensors)" in card.split("\n# ", 1)[1].split("\n", 1)[0]
    assert "You are Minecraftism." in card and "ollama run" not in card

    gguf = llm_train.model_card("minecraftism", preset, report, "me/Minecraftism-4B-GGUF")
    assert "library_name: gguf" in gguf and "[me/Minecraftism-4B](https://huggingface.co/me/Minecraftism-4B)" in gguf
    assert "ollama run hf.co/me/Minecraftism-4B-GGUF:Q4_K_M" in gguf

    vl = llm_train.model_card("minecraftism-vl", llm_train.PRESETS["minecraftism-vl"], report, "me/Minecraftism-VL-4B",
                              kind="safetensors")
    assert "AutoModelForImageTextToText" in vl and "AutoProcessor" in vl and "pipeline_tag: image-text-to-text" in vl


def _args(tmp_path: Path, **extra) -> object:
    args = llm_train.build_parser().parse_args(["--preset", "minecraftism", "--gguf", "none", "--eval-limit", "0",
                                                "--out", str(tmp_path / "out")] + list(extra.pop("argv", [])))
    args.system_prompt = "You are Minecraftism."
    args.data_dir = str(tmp_path)
    for key, value in extra.items():
        setattr(args, key, value)
    return args


@pytest.fixture()
def fake_hub(monkeypatch):
    uploads: list[dict] = []

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "me"}

        def create_repo(self, repo, **kwargs):
            uploads.append({"create": repo, **kwargs})

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            files = sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())
            uploads.append({**kwargs, "files": files})

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    return uploads


def test_finish_uploads_the_gguf_and_the_safetensors_edition(tmp_path, fake_hub):
    args = _args(tmp_path, argv=["--hub-repo", "auto"])
    model = FakeModel()
    code = llm_train.finish(args, llm_train.PRESETS["minecraftism"], model, FakeTokenizer(), [], [], 0.5, 0.0,
                            generate=None, for_inference=lambda: None)
    assert code == 0 and model.calls == [("merged", "merged_16bit")]
    out = tmp_path / "out"
    report = json.loads((out / "pymc_model.json").read_text())
    assert report["hub_repo"] == "me/Minecraftism-4B-GGUF"
    assert report["hub_repo_safetensors"] == "me/Minecraftism-4B" and report["safetensors_uploaded"] is True
    created = [u["create"] for u in fake_hub if "create" in u]
    assert created == ["me/Minecraftism-4B-GGUF", "me/Minecraftism-4B"]
    gguf_upload, st_upload = (u for u in fake_hub if "folder_path" in u)
    assert "safetensors/*" in gguf_upload["ignore_patterns"] and "lora/*" in gguf_upload["ignore_patterns"]
    assert st_upload["repo_id"] == "me/Minecraftism-4B" and st_upload["folder_path"] == str(out / "safetensors")
    assert "README.md" in st_upload["files"] and "model-00001-of-00002.safetensors" in st_upload["files"]
    assert "library_name: transformers" in (out / "safetensors/README.md").read_text()
    assert "https://huggingface.co/me/Minecraftism-4B" in (out / "README.md").read_text()


def test_safetensors_upload_failure_keeps_the_folder_and_the_gguf(tmp_path, fake_hub, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr(llm_train, "publish_safetensors", broken)
    args = _args(tmp_path, argv=["--hub-repo", "auto"])
    assert llm_train.finish(args, llm_train.PRESETS["minecraftism"], FakeModel(), FakeTokenizer(), [], [], 0.5, 0.0,
                            generate=None, for_inference=lambda: None) == 0
    report = json.loads((tmp_path / "out/pymc_model.json").read_text())
    assert report["hub_repo"] == "me/Minecraftism-4B-GGUF" and not report.get("safetensors_uploaded")
    assert (tmp_path / "out/safetensors/model-00001-of-00002.safetensors").is_file()


def test_gguf_upload_failure_is_not_reported_as_uploaded(tmp_path, fake_hub, monkeypatch):
    def broken(*_args, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr(llm_train, "publish", broken)
    args = _args(tmp_path, argv=["--hub-repo", "auto"])
    llm_train.finish(args, llm_train.PRESETS["minecraftism"], FakeModel(), FakeTokenizer(), [], [], 0.5, 0.0,
                     generate=None, for_inference=lambda: None)
    report = json.loads((tmp_path / "out/pymc_model.json").read_text())
    assert "hub_repo" not in report and not report.get("safetensors_uploaded")  # the notebook keeps every file


def test_safetensors_none_skips_the_export(tmp_path):
    args = _args(tmp_path, argv=["--safetensors", "none"])
    model = FakeModel()
    llm_train.finish(args, llm_train.PRESETS["minecraftism"], model, FakeTokenizer(), [], [], 0.5, 0.0,
                     generate=None, for_inference=lambda: None)
    assert model.calls == [] and not (tmp_path / "out/safetensors").exists()
    assert "safetensors" not in json.loads((tmp_path / "out/pymc_model.json").read_text())


def test_failed_export_does_not_stop_the_run(tmp_path):
    args = _args(tmp_path)
    assert llm_train.finish(args, llm_train.PRESETS["minecraftism"], FakeModel(write_weights=False), FakeTokenizer(),
                            [], [], 0.5, 0.0, generate=None, for_inference=lambda: None) == 0
    assert not (tmp_path / "out/safetensors").exists()


def test_notebook_exports_and_uploads_both_editions():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    text = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "SAFETENSORS = True" in code
    assert '" --safetensors 16bit" if SAFETENSORS else " --safetensors none"' in code
    assert "--safetensors none --general" in code  # QUICK_TEST exports nothing
    assert '"--safetensors"))' in code  # an older trainer without it is replaced by the newest one
    assert 'info.get("safetensors_uploaded")' in code  # never deletes a folder that is not on Hugging Face
    assert "Minecraftism-4B-Uncensored-GGUF" in text and "`you/Minecraftism-4B-Uncensored`" in text
