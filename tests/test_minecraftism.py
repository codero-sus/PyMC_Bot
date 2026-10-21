"""Minecraftism: the four editions (text/VL x fast/thinking), rule-based thinking traces, picture rows
from the mod's screenshots, vision at play time and Hugging Face publishing/import."""

from __future__ import annotations

import base64
import json
import random
import re
import struct
import zlib
from pathlib import Path

import httpx
import pytest

from pymc_bot import llm_dataset as ds
from pymc_bot import llm_reasoning as lr
from pymc_bot import llm_train
from pymc_bot.__main__ import main as cli
from pymc_bot.agent import VISION_NOTE, AgentLoop, parse_decision
from pymc_bot.config import ConfigStore
from pymc_bot.cortex import CortexClient, image_data_uri
from pymc_bot.llm_tools import (
    edition_from_repo,
    hf_reference,
    import_model,
    import_model_hf,
    write_modelfile,
)
from pymc_bot.ollama import OllamaClient

ROOT = Path(__file__).resolve().parents[1]
EDITIONS = ("minecraftism", "minecraftism-thinking", "minecraftism-vl", "minecraftism-vl-thinking")


def png(width: int = 8, height: int = 6) -> bytes:
    """A tiny valid PNG (no Pillow needed)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([40, 160, 60]) * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture
def store(tmp_path) -> ConfigStore:
    return ConfigStore(tmp_path / "config.json")


@pytest.fixture
def shots_playtime(tmp_path) -> Path:
    """Recorded playtime where every 4th sample has a screenshot, like mod 0.10 writes it."""
    from pymc_bot.playtime import synthesize_playtime

    root = tmp_path / "pymcbot"
    root.mkdir()
    dataset = root / "dataset.jsonl"
    synthesize_playtime(dataset, minutes=2, seed=5)
    (root / "shots").mkdir()
    lines = []
    for number, line in enumerate(dataset.read_text(encoding="utf-8").splitlines()):
        row = json.loads(line)
        if number % 4 == 0:
            name = f"shots/shot-{number:05d}.png"
            (root / name).write_bytes(png(8 + number % 3, 6))
            row["input"]["shot"] = name
        lines.append(json.dumps(row))
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


# ------------------------------------------------------------------ editions
def test_four_editions_on_the_right_qwen_models():
    assert llm_train.DEFAULT_PRESET == "minecraftism" and tuple(llm_train.EDITIONS) == EDITIONS
    expected = {"minecraftism": ("unsloth/Qwen3-4B-Instruct-2507", False, False),
                "minecraftism-thinking": ("unsloth/Qwen3-4B-Thinking-2507", True, False),
                "minecraftism-vl": ("unsloth/Qwen3-VL-4B-Instruct", False, True),
                "minecraftism-vl-thinking": ("unsloth/Qwen3-VL-4B-Thinking", True, True)}
    hubs = set()
    for name, (model, reasoning, vision) in expected.items():
        preset = llm_train.PRESETS[name]
        assert (preset["model"], preset["reasoning"], preset["vision"]) == (model, reasoning, vision)
        assert preset["train_gb"] <= 16 and preset["edition"] == name
        hubs.add(preset["hub"])
        assert preset["hub"].startswith("Minecraftism") and preset["hub"].endswith("-GGUF")
    assert len(hubs) == 4
    assert llm_train.display_name("minecraftism-vl-thinking", llm_train.PRESETS["minecraftism-vl-thinking"]) == \
        "Minecraftism VL 4B Thinking"
    assert llm_train.output_name("minecraftism", llm_train.PRESETS["minecraftism"], "q4_k_m") == "minecraftism-4b.Q4_K_M.gguf"


def test_thinking_editions_learn_the_trace_and_fast_ones_do_not():
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                        {"role": "assistant", "content": '{"action": "flee"}'}],
           "reasoning": "A creeper is 3 blocks away. Run."}
    fast = llm_train.prepare_messages(row, llm_train.PRESETS["minecraftism"])
    assert fast[-1]["content"] == '{"action": "flee"}'
    thinking = llm_train.prepare_messages(row, llm_train.PRESETS["minecraftism-thinking"])
    assert thinking[-1]["content"] == '<think>\nA creeper is 3 blocks away. Run.\n</think>\n\n{"action": "flee"}'
    assert row["messages"][-1]["content"] == '{"action": "flee"}', "the row itself is not changed"
    assert llm_train.strip_thinking(thinking[-1]["content"]) == '{"action": "flee"}'


def test_text_editions_skip_picture_only_rows():
    rows = [{"kind": "turn", "images": ["images/a.jpg"]}, {"kind": "vision", "images": ["images/b.jpg"]}, {"kind": "qa"}]
    assert len(llm_train.select_rows(rows, llm_train.PRESETS["minecraftism"])) == 2
    assert len(llm_train.select_rows(rows, llm_train.PRESETS["minecraftism-vl"])) == 3


def test_vision_messages_put_the_picture_before_the_question():
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "what do you see?"},
                        {"role": "assistant", "content": "a zombie"}], "images": ["images/x.jpg"], "reasoning": "dark"}
    messages = llm_train.vision_messages(row, llm_train.PRESETS["minecraftism-vl-thinking"], lambda name: f"<{name}>")
    assert messages[1]["content"][0] == {"type": "image", "image": "<images/x.jpg>"}
    assert messages[1]["content"][1] == {"type": "text", "text": "what do you see?"}
    assert messages[2]["content"][0]["text"].startswith("<think>\ndark\n</think>")


# ------------------------------------------------------------------ thinking traces
@pytest.mark.parametrize(("dx", "dz", "yaw", "side"), [
    (0, 5, 0, "ahead"), (0, -5, 0, "behind"), (5, 0, 0, "left"), (-5, 0, 0, "right"),
    (-5, 0, 90, "ahead"), (0, 5, 90, "left"),
])
def test_relative_direction_follows_minecraft_yaw(dx, dz, yaw, side):
    """Yaw 0 looks south (+Z), so east (+X) is on the player's left."""
    assert side in lr.relative_direction(dx, dz, yaw)


def test_rule_traces_are_grounded_and_safe(tmp_path):
    card = ds.build_dataset(tmp_path / "data", rule_turns=300, mob_rounds=1, log=lambda _m: None)
    rows = ds.read_jsonl(tmp_path / "data" / "train.jsonl")
    assert card["dataset_version"] >= 2 and rows
    assert all(row.get("reasoning") for row in rows), "every row teaches the thinking editions something"
    for row in rows:
        trace = row["reasoning"]
        assert "{" not in trace and "}" not in trace and "<think>" not in trace.lower() and "</think>" not in trace.lower()
        assert len(trace) <= lr.MAX_TRACE
    assert ds.validate_rows(rows) == []
    turns = [row for row in rows if row["kind"] == "turn"]
    assert any("Health" in row["reasoning"] for row in turns)


def test_validate_rows_rejects_braces_in_traces_and_missing_pictures(tmp_path):
    good = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "reasoning": "fine"}
    assert ds.validate_rows([good]) == []
    bad = dict(good, reasoning="use {json} here")
    assert ds.validate_rows([bad])
    missing = dict(good, images=["images/nope.jpg"])
    assert ds.validate_rows([missing], base=tmp_path)


def test_chat_questions_about_mobs_pull_in_the_facts():
    line = lr.chat_line([{"username": "Priya", "message": "Bot_Ravi how hard do wolfs hit"}], "Bot_Ravi")
    assert "a question for me" in line and "What I know about the wolf" in line and "4" in line
    assert "not aimed at me" in lr.chat_line([{"username": "a", "message": "x"}, {"username": "Leo", "message": "lol"}],
                                            "Bot_Ravi")


def test_thinking_model_replies_parse_without_the_opening_tag():
    reply = 'Health is fine. Zombie close, {not json} maybe.\n</think>\n\n{"action": "flee", "thought": "run"}'
    decision = parse_decision(reply)
    assert decision.action == "flee" and decision.thought == "run"
    assert parse_decision('<think>hmm</think>{"action": "idle"}').action == "idle"


# ------------------------------------------------------------------ pictures
def test_screenshots_become_picture_rows(shots_playtime, tmp_path):
    out = tmp_path / "data"
    examples, stats = ds.vision_examples(shots_playtime / "dataset.jsonl", out, random.Random(2))
    assert stats["shots"] > 0 and stats.get("missing", 0) == 0
    assert stats["captions"] == stats["shots"] and examples
    for example in examples:
        assert len(example.images) == 1 and (out / example.images[0]).is_file()
        assert "{" not in example.reasoning
    captions = [e for e in examples if e.kind == "vision"]
    turns = [e for e in examples if e.kind == "turn"]
    assert captions and all(e.messages[0]["content"] == ds.VISION_SYSTEM for e in captions)
    assert all(e.messages[1]["content"].endswith(VISION_NOTE) for e in turns), "same words as at play time"


def test_build_with_screenshots_and_vl_dry_run(shots_playtime, tmp_path, capsys):
    out = tmp_path / "data"
    card = ds.build_dataset(out, rule_turns=200, mob_rounds=1, screenshots=shots_playtime / "dataset.jsonl",
                            log=lambda _m: None)
    assert card["images"] > 0 and (out / "images").is_dir()
    rows = ds.read_jsonl(out / "train.jsonl") + ds.read_jsonl(out / "eval.jsonl")
    assert ds.validate_rows(rows, base=out) == []
    for edition in EDITIONS:
        assert llm_train.main(["--dry-run", "--data", str(out), "--preset", edition]) == 0, edition
        printed = capsys.readouterr().out
        assert llm_train.PRESETS[edition]["model"] in printed


def test_vl_dry_run_refuses_a_dataset_without_pictures(tmp_path, capsys):
    out = tmp_path / "data"
    ds.build_dataset(out, rule_turns=120, mob_rounds=1, log=lambda _m: None)
    assert llm_train.main(["--dry-run", "--data", str(out), "--preset", "minecraftism-thinking"]) == 0
    capsys.readouterr()
    assert llm_train.main(["--dry-run", "--data", str(out), "--preset", "minecraftism-vl"]) == 2
    assert "--screenshots" in capsys.readouterr().out


def test_cli_build_screenshots_needs_a_folder(tmp_path, capsys):
    code = cli(["llm", "build", "--no-wiki", "--turns", "20", "--out", str(tmp_path / "d"), "--screenshots",
                "--config", str(tmp_path / "c.json")])
    assert code == 2 and "--screenshots needs" in capsys.readouterr().err


# ------------------------------------------------------------------ publishing
def test_model_card_and_hub_names():
    preset = llm_train.PRESETS["minecraftism-vl"]
    report = {"gguf": "minecraftism-vl-4b.Q4_K_M.gguf", "mmproj": "mmproj-minecraftism-vl-4b.F16.gguf", "quant": "Q4_K_M",
              "loss": 0.41, "train_rows": 900, "images": 120}
    card = llm_train.model_card("minecraftism-vl", preset, report, "me/Minecraftism-VL-4B-GGUF")
    assert card.startswith("---\n") and "base_model: Qwen/Qwen3-VL-4B-Instruct" in card
    assert "Minecraftism VL 4B" in card and "mmproj-minecraftism-vl-4b.F16.gguf" in card
    assert "ollama pull hf.co/me/Minecraftism-VL-4B-GGUF" in card or "hf.co/me/Minecraftism-VL-4B-GGUF" in card
    assert "codero-sus" in card and "Noncommercial" in card
    partial = llm_train.model_card("minecraftism", llm_train.PRESETS["minecraftism"], {"eval": {"json_valid_pct": 97.0}}, "a/b")
    assert "| 97.0% | - | - |" in partial and "None" not in partial
    assert llm_train.resolve_repo("auto", preset, lambda: {"name": "me"}) == "me/Minecraftism-VL-4B-GGUF"
    assert llm_train.resolve_repo("mine", preset, lambda: {"name": "me"}) == "me/mine"
    assert llm_train.resolve_repo("you/x", preset) == "you/x"
    params = llm_train.ollama_params(llm_train.PRESETS["minecraftism-thinking"])
    assert params["temperature"] == llm_train.RUN_THINKING["temperature"] and "<|im_end|>" in params["stop"]


def test_trainer_takes_the_hub_options():
    args = llm_train.build_parser().parse_args(["--hub-repo", "auto", "--hub-private", "--image-side", "448"])
    assert args.hub_repo == "auto" and args.hub_private and args.image_side == 448
    source = Path(llm_train.__file__).read_text(encoding="utf-8")
    assert 'os.environ.get("HF_TOKEN")' in source and "print(token" not in source


def test_modelfile_with_projector(tmp_path):
    folder = tmp_path / "minecraftism-vl"
    folder.mkdir()
    gguf = folder / "minecraftism-vl-4b.Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF" + b"0" * 64)
    (folder / "mmproj-minecraftism-vl-4b.F16.gguf").write_bytes(b"GGUF" + b"1" * 16)
    info = {"preset": "minecraftism-vl", "mmproj": "mmproj-minecraftism-vl-4b.F16.gguf"}
    text = write_modelfile(gguf, info).read_text()
    lines = text.splitlines()
    assert lines[0] == f"FROM {gguf.as_posix()}" and lines[1].startswith("FROM ") and lines[1].endswith(".F16.gguf")
    assert f"PARAMETER top_p {llm_train.RUN_VISION['top_p']}" in text


def test_import_folder_names_the_model_after_the_edition(tmp_path, store):
    folder = tmp_path / "minecraftism-vl"
    folder.mkdir()
    (folder / "minecraftism-vl-4b.Q4_K_M.gguf").write_bytes(b"GGUF" + b"0" * 64)
    (folder / "mmproj-minecraftism-vl-4b.F16.gguf").write_bytes(b"GGUF")
    (folder / "pymc_model.json").write_text(json.dumps({"preset": "minecraftism-vl", "edition": "minecraftism-vl",
                                                        "vision": True, "compact_prompt": True}))

    class Done:
        returncode, stdout, stderr = 0, "success", ""

    runs = []
    result = import_model(folder, store, run=lambda cmd, **_k: runs.append(cmd) or Done(), which=lambda _n: "/bin/ollama")
    assert result.name == "minecraftism-vl" and runs[0][:3] == ["ollama", "create", "minecraftism-vl"]
    settings = store.settings
    assert settings.ollama.model == "minecraftism-vl" and settings.agent.vision and settings.agent.compact_prompt


def test_import_from_hugging_face(store):
    assert hf_reference("me/Minecraftism-4B-GGUF") == "hf.co/me/Minecraftism-4B-GGUF:Q4_K_M"
    assert hf_reference("https://huggingface.co/me/Minecraftism-4B-GGUF/", "q8_0") == "hf.co/me/Minecraftism-4B-GGUF:Q8_0"
    with pytest.raises(ValueError):
        hf_reference("not-a-repo")
    assert edition_from_repo("hf.co/me/Minecraftism-VL-4B-Thinking-GGUF:Q4_K_M")["edition"] == "minecraftism-vl-thinking"
    assert edition_from_repo("hf.co/me/Minecraftism-4B-GGUF:Q4_K_M")["edition"] == "minecraftism"

    class Done:
        returncode, stdout, stderr = 0, "ok", ""

    runs = []
    result = import_model_hf("me/Minecraftism-VL-4B-GGUF", store, run=lambda cmd, **_k: runs.append(cmd) or Done(),
                             which=lambda _n: "/bin/ollama", fetch=lambda _r: {})
    assert runs == [["ollama", "pull", "hf.co/me/Minecraftism-VL-4B-GGUF:Q4_K_M"],
                    ["ollama", "cp", "hf.co/me/Minecraftism-VL-4B-GGUF:Q4_K_M", "minecraftism-vl"]]
    assert result.created and result.activated
    assert store.settings.ollama.model == "minecraftism-vl" and store.settings.agent.vision


def test_failed_hf_pull_keeps_the_settings(store):
    class Failed:
        returncode, stdout, stderr = 1, "", "file does not exist"

    before = store.settings.ollama.model
    result = import_model_hf("me/Minecraftism-4B-GGUF", store, run=lambda *_a, **_k: Failed(), which=lambda _n: "x",
                             fetch=lambda _r: {})
    assert not result.created and not result.activated and "does not exist" in result.message
    assert store.settings.ollama.model == before


def test_cli_import_needs_exactly_one_source(tmp_path, capsys):
    assert cli(["llm", "import", "--config", str(tmp_path / "c.json")]) == 2
    assert "either" in capsys.readouterr().err


# ------------------------------------------------------------------ vision at play time
def test_ollama_sends_pictures_with_the_last_user_message():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": '{"action": "idle"}'}})

    client = OllamaClient("http://ollama", "minecraftism-vl", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.chat([{"role": "user", "content": "hi"}], system="s", images=["QUJD"])
    assert seen["messages"][-1] == {"role": "user", "content": "hi", "images": ["QUJD"]}
    assert "images" not in seen["messages"][0]


def test_cortex_sends_pictures_as_data_uris():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = CortexClient("http://cortex", client=httpx.Client(transport=httpx.MockTransport(handler)))
    client.decide("look", images=["/9j/AAAA"])
    parts = seen["messages"][-1]["content"]
    assert parts[0] == {"type": "text", "text": "look"}
    assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,/9j/AAAA"
    assert image_data_uri("iVBORw0") == "data:image/png;base64,iVBORw0"
    assert image_data_uri("data:image/png;base64,x") == "data:image/png;base64,x"


def test_agent_attaches_a_picture_every_n_turns(store):
    from pymc_bot.bot import MinecraftBot
    from pymc_bot.events import EventLog

    store.update({"agent": {"vision": True, "vision_every": 2}})
    log = EventLog()
    loop = AgentLoop(MinecraftBot(store, log), store, log)

    class Eyes:
        calls = 0

        def view_image(self, width):
            Eyes.calls += 1
            return {"image": base64.b64encode(b"jpeg").decode(), "mime": "image/jpeg"}

    loop.bot._backend = Eyes()
    frames = [loop.vision_frame(store.settings) for _ in range(4)]
    assert [bool(f) for f in frames] == [True, False, True, False] and Eyes.calls == 2

    class Blind:
        def view_image(self, width):
            raise RuntimeError("vision is not installed")

    loop.bot._backend = Blind()
    assert loop.vision_frame(store.settings) is None
    assert any("vision is not installed" in entry["message"] for entry in log.tail())
    store.update({"agent": {"vision": False}})
    assert loop.vision_frame(store.settings) is None


def test_bridge_has_an_optional_snapshot_command():
    source = (ROOT / "pymc_bot" / "node" / "minecraft_bridge.js").read_text(encoding="utf-8")
    assert "snapshot: cmdSnapshot," in source and "setFirstPersonCamera" in source
    assert "prismarine-viewer node-canvas-webgl three@0.128.0" in source
    package = json.loads((ROOT / "pymc_bot" / "node" / "package.json").read_text(encoding="utf-8"))
    assert "prismarine-viewer" not in package.get("dependencies", {}), "vision stays an optional install"


def test_panel_has_the_vision_switch():
    html = (ROOT / "pymc_bot" / "web" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "pymc_bot" / "web" / "app.js").read_text(encoding="utf-8")
    assert 'id="cfg-vision"' in html and 'id="cfg-vision-every"' in html
    assert re.search(r"vision: \$\('cfg-vision'\)\.checked", app) and "vision_every" in app
