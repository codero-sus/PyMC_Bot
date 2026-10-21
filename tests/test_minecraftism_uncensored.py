"""Minecraftism as an everyday, uncensored assistant: the default system prompt, abliterated bases,
refusal-free chat data, identity rows, the optional test suite and the Kaggle notebook settings."""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

from pymc_bot import llm_train
from pymc_bot.config import ConfigStore
from pymc_bot.llm_tools import default_name, edition_from_repo, import_model_hf, write_modelfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "llm" / "PyMC_Minecraft_LLM_Kaggle.ipynb"


@pytest.fixture
def store(tmp_path) -> ConfigStore:
    return ConfigStore(tmp_path / "config.json")


@pytest.fixture(scope="module")
def llm_data(tmp_path_factory) -> Path:
    from pymc_bot.__main__ import main as cli

    out = tmp_path_factory.mktemp("llm") / "llm_data"
    assert cli(["llm", "build", "--no-wiki", "--turns", "60", "--out", str(out)]) == 0
    return out


def notebook_code() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


# ------------------------------------------------------------------ system prompt
def test_system_prompt_is_an_everyday_assistant_not_a_bot():
    text = llm_train.MINECRAFTISM_SYSTEM
    assert text.startswith("You are Minecraftism") and "codero-sus" in text
    assert "general assistant first" in text and "everyday life" in text
    assert "don't refuse" in text.lower() and "lecture" in text
    assert "JSON" in text, "it still answers in the bot's format when a game turn asks for it"
    assert "you are a minecraft bot" not in text.lower() and "player bot" not in text.lower()
    assert '"""' not in text, "it is embedded in a Modelfile SYSTEM block"


def test_modelfile_gets_the_system_prompt(tmp_path):
    text = llm_train.modelfile_text("m.gguf", "chatml", None, system=llm_train.MINECRAFTISM_SYSTEM)
    assert 'SYSTEM """You are Minecraftism' in text and text.count('"""') >= 2
    assert "SYSTEM" not in llm_train.modelfile_text("m.gguf", "chatml", None)
    (tmp_path / "x.Q4_K_M.gguf").write_bytes(b"GGUF")
    write_modelfile(tmp_path / "x.Q4_K_M.gguf", {"family": "chatml", "system_prompt": "Be nice."})
    assert 'SYSTEM """Be nice."""' in (tmp_path / "Modelfile").read_text()


# ------------------------------------------------------------------ uncensored presets
def test_every_edition_has_an_abliterated_base():
    for name in llm_train.EDITIONS:
        preset = llm_train.PRESETS[name]
        assert preset["uncensored_model"].startswith("huihui-ai/") and "abliterated" in preset["uncensored_model"]
        assert ("VL" in preset["uncensored_model"]) == bool(preset.get("vision"))
        assert ("Thinking" in preset["uncensored_model"]) == bool(preset.get("reasoning"))
        free = llm_train.uncensored_preset(name, preset)
        assert free["model"] == free["base_model"] == preset["uncensored_model"] and free["uncensored"]
        assert free["hub"].endswith("-Uncensored-GGUF") and free["file"].endswith("-uncensored")
        assert llm_train.display_name(name, free).endswith(" Uncensored")
        assert "uncensored" not in preset, "the stock preset is left alone"


def test_uncensored_names_round_trip_through_import():
    free = llm_train.uncensored_preset("minecraftism-vl", llm_train.PRESETS["minecraftism-vl"])
    info = edition_from_repo(f"hf.co/me/{free['hub']}:Q4_K_M")
    assert info["edition"] == "minecraftism-vl" and info["uncensored"] and info["vision"]
    assert default_name(info) == "minecraftism-vl-uncensored"
    assert default_name({"edition": "minecraftism"}) == "minecraftism"


def test_hf_import_builds_the_model_with_its_system_prompt(store):
    class Done:
        returncode, stdout, stderr = 0, "ok", ""

    runs, modelfiles = [], []

    def run(cmd, **_kw):
        runs.append(cmd)
        if cmd[1] == "create":
            modelfiles.append(Path(cmd[-1]).read_text())
        return Done()

    report = {"system_prompt": "You are Minecraftism.", "uncensored": True}
    result = import_model_hf("me/Minecraftism-4B-Uncensored-GGUF", store, run=run, which=lambda _n: "/bin/ollama",
                             fetch=lambda _ref: report)
    ref = "hf.co/me/Minecraftism-4B-Uncensored-GGUF:Q4_K_M"
    assert runs[0] == ["ollama", "pull", ref] and runs[1][:3] == ["ollama", "create", "minecraftism-uncensored"]
    assert modelfiles == [f'FROM {ref}\nSYSTEM """You are Minecraftism."""\n']
    assert result.created and store.settings.ollama.model == "minecraftism-uncensored"


# ------------------------------------------------------------------ data
def test_refusal_detector():
    for text in ("I'm sorry, but I can't help with that.", "As an AI language model, I don't have opinions.",
                 "I cannot provide that information.", "That goes against my guidelines.", "I'm ChatGPT.",
                 "I am Qwen, created by Alibaba Cloud."):
        assert llm_train.is_refusal(text), text
    for text in ("Sure! Mix 2 parts flour with 1 part water.", "Creepers can't open doors.",
                 "<think>I'm sorry, but... no, that's fine</think>Here you go: 42",
                 "No, I'm Minecraftism - not ChatGPT."):
        assert not llm_train.is_refusal(text), text


def test_general_rows_replace_the_system_prompt_and_drop_refusals():
    records = [
        {"conversations": [{"from": "system", "value": "You are ChatGPT."}, {"from": "human", "value": "Hi"},
                           {"from": "gpt", "value": "Hello! How can I help?"}]},
        {"conversations": [{"from": "human", "value": "Write a dark poem"},
                           {"from": "gpt", "value": "I'm sorry, but I can't write that."}]},
        {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"},
                      {"role": "user", "content": "dangling"}]},
        {"conversations": [{"from": "gpt", "value": "no user turn"}]},
        {"conversations": [{"from": "human", "value": "x" * 7000}, {"from": "gpt", "value": "too long"}]},
    ]
    rows = llm_train.general_rows(records, 10, "SYS", uncensored=True)
    assert len(rows) == 2 and all(r["kind"] == "general" for r in rows)
    assert rows[0]["messages"][0] == {"role": "system", "content": "SYS"} and len(rows[0]["messages"]) == 3
    assert rows[1]["messages"][-1] == {"role": "assistant", "content": "4"}, "trailing user turn trimmed"
    assert len(llm_train.general_rows(records, 10, "SYS", uncensored=False)) == 3, "censored runs keep refusals"
    assert len(llm_train.general_rows(records, 1, "SYS", uncensored=True)) == 1
    assert not llm_train.check_rows(rows)


def test_reasoning_rows_keep_the_real_trace():
    records = [{"problem": "What is 6*7?", "generated_solution": "<think>\n6*7 = 42.\nCheck: 42/7 = 6.\n</think>\n\n**42**"},
               {"problem": "No trace", "generated_solution": "just 1"}]
    rows = llm_train.reasoning_rows(records, 5, "SYS")
    assert len(rows) == 1 and rows[0]["reasoning"] == "6*7 = 42.\nCheck: 42/7 = 6."
    assert rows[0]["messages"][-1]["content"] == "**42**"
    text = llm_train.prepare_messages(rows[0], llm_train.PRESETS["minecraftism-thinking"])[-1]["content"]
    assert text.startswith("<think>\n6*7 = 42.\nCheck: 42/7 = 6.\n</think>"), "multi-line traces survive"
    assert text.endswith("**42**")


def test_identity_and_candor_rows():
    rows = llm_train.identity_rows("SYS", "Minecraftism", random.Random(1), candor=True, count=40)
    answers = [r["messages"][-1]["content"] for r in rows]
    assert len(rows) == 40 + 2 * len(llm_train._CANDOR)
    assert all(r["messages"][0]["content"] == "SYS" and r["reasoning"] for r in rows)
    assert all("Minecraftism" in a for a in answers[:40]) and all("codero-sus" in a for a in answers[:40] if "I'm" in a[:4])
    assert not any(llm_train.is_refusal(a) for a in answers)
    chatgpt = [r for r in rows if "chatgpt" in r["messages"][1]["content"].lower()]
    assert all("not ChatGPT" in r["messages"][-1]["content"] for r in chatgpt)
    assert len(llm_train.identity_rows("SYS", "M", random.Random(1), candor=False, count=5)) == 5
    assert not llm_train.check_rows(rows)


def test_persona_only_touches_qa_rows():
    rows = [{"kind": "qa", "messages": [{"role": "system", "content": "old"}, {"role": "user", "content": "q"},
                                        {"role": "assistant", "content": "a"}]} for _ in range(50)]
    rows.append({"kind": "turn", "messages": [{"role": "system", "content": "bot"}, {"role": "user", "content": "s"},
                                               {"role": "assistant", "content": "{}"}]})
    changed = llm_train.apply_persona(rows, "SYS", 0.5, random.Random(3))
    assert 10 < changed < 40
    assert sum(r["messages"][0]["content"] == "SYS" for r in rows) == changed
    assert rows[-1]["messages"][0]["content"] == "bot"
    assert all(len(r["messages"]) == 3 for r in rows)


# ------------------------------------------------------------------ tests suite
def test_run_tests_summarises_refusals_identity_and_json():
    eval_rows = [{"kind": "turn", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "state"},
                                               {"role": "assistant", "content": '{"action": "idle"}'}],
                  "check": {"action": "idle"}},
                 {"kind": "qa", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "creeper hp?"},
                                             {"role": "assistant", "content": "20"}], "check": {"contains": ["20"]}}]
    rows = llm_train.test_rows(eval_rows, llm_train.PRESETS["minecraftism"], "SYS")
    assert [r["test"] for r in rows[:2]] == ["bot turn", "minecraft q&a"]
    assert len(rows) == 2 + len(llm_train.TEST_PROMPTS)

    def generate(row):
        prompt = row["messages"][-2]["content"]
        if row["test"] == "bot turn":
            return '{"action": "idle", "say": ""}'
        if "dark joke" in prompt:
            return "I'm sorry, but I can't help with that."
        if "Who are you" in prompt:
            return "<think>easy</think>I'm Minecraftism, made by codero-sus."
        if "workout" in prompt:
            raise RuntimeError("CUDA out of memory")
        return "Creepers have 20 HP."

    logged = []
    summary = llm_train.run_tests(generate, rows, log=logged.append)
    assert summary["prompts"] == len(rows) and summary["refusals"] == 1
    assert summary["identity_ok"] and summary["bot_json_ok"] == "1/1"
    assert any("<error: RuntimeError" in r["reply"] for r in summary["results"])
    assert "refusal(s)" in logged[-1]
    bad = llm_train.run_tests(lambda row: "I am ChatGPT.", rows[2:3], log=lambda _m: None)
    assert not bad["identity_ok"] and bad["refusals"] == 1


def test_vision_editions_test_a_picture():
    eval_rows = [{"kind": "vision", "images": ["images/a.png"], "messages": [
        {"role": "user", "content": "what do you see?"}, {"role": "assistant", "content": "a creeper"}], "check": {}}]
    assert llm_train.test_rows(eval_rows, llm_train.PRESETS["minecraftism-vl"], "S")[-1]["test"] == "vision"
    assert all(r["test"] != "vision" for r in llm_train.test_rows(eval_rows, llm_train.PRESETS["minecraftism"], "S"))


def test_gguf_test_skips_without_ollama(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_train.shutil, "which", lambda _n: None)
    assert llm_train.test_gguf(tmp_path, {}, ["hi"], log=lambda _m: None) == {"skipped": "no ollama"}


# ------------------------------------------------------------------ trainer CLI
def test_dry_run_with_everything_on(llm_data, tmp_path):
    system = tmp_path / "system.txt"
    system.write_text("You are a pirate who loves Minecraft.")
    proc = subprocess.run([sys.executable, str(llm_data / "train_minecraft_llm.py"), "--data", str(llm_data),
                           "--preset", "minecraftism-thinking", "--dry-run", "--uncensored", "--general", "500", "--test",
                           "--test-gguf", "--system-prompt-file", str(system), "--persona-mix", "1"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Huihui-Qwen3-4B-Thinking-2507-abliterated" in proc.stdout and "Uncensored" in proc.stdout
    assert "unsloth/OpenMathReasoning-mini" in proc.stdout, "thinking editions get real traces"
    assert re.search(r"persona: \d+ Q&A rows", proc.stdout) and "(uncensored)" in proc.stdout


def test_test_only_needs_a_trained_adapter(llm_data, tmp_path):
    proc = subprocess.run([sys.executable, str(llm_data / "train_minecraft_llm.py"), "--data", str(llm_data),
                           "--test-only", "--out", str(tmp_path / "nothing")], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 2 and "no trained adapter" in proc.stdout


def test_model_card_mentions_uncensored_tests_and_prompt():
    free = llm_train.uncensored_preset("minecraftism", llm_train.PRESETS["minecraftism"])
    report = {"quant": "Q4_K_M", "gguf": "x.gguf", "system_prompt": llm_train.MINECRAFTISM_SYSTEM,
              "tests": {"prompts": 14, "refusals": 0, "identity_ok": True, "bot_json_ok": "3/3"}}
    card = llm_train.model_card("minecraftism", free, report, "me/Minecraftism-4B-Uncensored-GGUF")
    assert "- uncensored" in card and "- abliterated" in card and "**uncensored**" in card
    assert "huihui-ai/Huihui-Qwen3-4B-Instruct-2507-abliterated" in card
    assert "0 refusal(s) in 14 prompts" in card and "Default system prompt" in card and "You are Minecraftism" in card
    plain = llm_train.model_card("minecraftism", llm_train.PRESETS["minecraftism"], {"quant": "Q4_K_M"}, "me/x")
    assert "abliterated" not in plain and "Default system prompt" not in plain


# ------------------------------------------------------------------ notebook
def test_notebook_settings_and_flags():
    code = notebook_code()
    prompt = re.search(r'^SYSTEM_PROMPT = """(.*?)"""', code, flags=re.M | re.S).group(1)
    assert prompt == llm_train.MINECRAFTISM_SYSTEM, "the notebook ships the same default prompt as the trainer"
    for setting in ("UNCENSORED = True", "GENERAL_CHAT = 3000", "RUN_TESTS = True", "TEST_GGUF = False",
                    "QUICK_TEST = False", "PERSONA_MIX = 0.5", "RETEST = []", 'CHAT_WITH = ""'):
        assert setting in code, setting
    for flag in ("--uncensored", "--system-prompt-file", "--general", "--test-gguf", "--test-only", "--persona-mix",
                 " --test"):
        assert flag in code, flag
    assert "ollama.com/install.sh" in code and "nvidia-smi" in code and "disk_usage" in code
    assert not re.search(r"print\([^)]*(environ|get_secret)", code), "the token must never be printed"
    parser = llm_train.build_parser()
    for flag in ("--uncensored", "--system-prompt-file", "--persona-mix", "--general", "--general-dataset",
                 "--reasoning-dataset", "--no-identity", "--test", "--test-only", "--test-gguf"):
        assert any(flag in action.option_strings for action in parser._actions), flag
