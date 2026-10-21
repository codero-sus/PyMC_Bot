"""The Minecraft LLM pipeline: dataset builder, Kaggle trainer (dry run), Ollama import and eval."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import httpx
import pytest

from pymc_bot import llm_dataset as ds
from pymc_bot import llm_train
from pymc_bot.__main__ import main as cli
from pymc_bot.agent import AgentLoop, parse_decision
from pymc_bot.config import ConfigStore
from pymc_bot.llm_tools import evaluate, find_gguf, import_model
from pymc_bot.mind import PERSONAS
from pymc_bot.mobs import load_mobs

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "training" / "llm" / "PyMC_Minecraft_LLM_Kaggle.ipynb"


def _hidden_texts() -> list[str]:
    return [text for p in PERSONAS.values() for text in (p.who, p.playstyle, p.voice) if len(text) > 25]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("llm") / "data"
    card = ds.build_dataset(out, rule_turns=400, mob_rounds=1, log=lambda _m: None)
    return out, card


# -------------------------------------------------------------------- prompts
def test_compact_prompt_is_short_and_public_only():
    allowed = {"movement": True, "chat": False, "mining": True, "attacking": False}
    prompt = ds.compact_system_prompt("Kira", "explorer", allowed, "loves cherry groves", 90)
    assert prompt.startswith(ds.COMPACT_TAG)
    assert "You are Kira" in prompt and "Explorer" in prompt and "loves cherry groves" in prompt
    assert "Switched off: chat, attacking." in prompt and "at most 90 characters" in prompt
    assert all(action in prompt for action in ("flee", "search", "wear_armor"))
    assert not any(hidden in prompt for hidden in _hidden_texts())
    assert len(prompt) < len(ds.build_system_prompt("Kira", "explorer", "", allowed, 90)) / 3


def test_decision_text_round_trips():
    text = ds.decision_text("give", "share", "help Leo", player="Leo", item="bread", count=4)
    decision = parse_decision(text)
    assert (decision.action, decision.thought, decision.goal) == ("give", "share", "help Leo")
    assert decision.params == {"player": "Leo", "item": "bread", "count": 4}
    assert list(json.loads(text))[:3] == ["thought", "goal", "action"]


# -------------------------------------------------------------------- sources
def test_mob_answers_use_the_database_numbers():
    db = load_mobs()
    zombie = db.get("zombie")
    rows = ds.mob_answers(db, zombie, random.Random(1))
    damage = next(r for r in rows if "damage" in r.messages[1]["content"].lower() and "mobs" not in r.messages[1]["content"])
    normal = ds._num(zombie["attacks"][0]["normal"])
    assert normal in damage.messages[2]["content"] and damage.check["contains"] == [normal]
    version = next(r for r in rows if zombie["added_in"] in r.messages[2]["content"] and "added in Java" in r.messages[2]["content"])
    assert version.kind == "qa" and version.messages[0]["content"] == ds.QA_SYSTEM


def test_newer_mobs_mention_mineflayer():
    db = load_mobs()
    rows = ds.mob_answers(db, db.get("sulfur_cube"), random.Random(2))
    assert any("mineflayer" in r.messages[2]["content"] for r in rows)


def test_rule_scenarios_are_valid_allowed_decisions():
    rows = ds.rule_examples(300, random.Random(3))
    actions = {r.check["action"] for r in rows}
    assert {"flee", "fight", "eat", "say", "give", "sleep", "mine", "wander"} <= actions
    for row in rows:
        reply = parse_decision(row.messages[2]["content"])
        assert reply.action == row.check["action"] and reply.action in ds.ACTION_NAMES
        system, user = row.messages[0]["content"], row.messages[1]["content"]
        allowed = json.loads(user.split("EXACT DATA (JSON): ")[1].split("\n")[0])["allowed"]
        assert ds.decision_allowed(reply, allowed), (reply.action, allowed)
        assert system.startswith(ds.COMPACT_TAG) and "SITUATION" in user
        assert not any(hidden in system for hidden in _hidden_texts())


def test_chat_off_waves_instead_of_talking():
    maker = ds.SituationMaker(random.Random(4))
    sit, reply, check = maker._chat_off()
    assert check == {"action": "emote"} and "Switched off right now: " in sit.user_prompt() and "chat" in sit.system_prompt()


def test_teacher_keeps_only_valid_allowed_decisions():
    replies = iter(['{"thought": "ok", "action": "wander"}', "I think I should wander", '{"action": "attack", "target": "Leo"}',
                    '<think>hmm</think>{"action": "eat"}'] * 50)
    seen = []

    def ask(system, prompt):
        seen.append(system)
        return next(replies)

    rows, stats = ds.teacher_examples(ask, 6, random.Random(5), log=lambda _m: None)
    assert len(rows) == 6 and stats["invalid"] >= 2 and stats["not_allowed"] >= 2
    assert all(json.loads(r.messages[2]["content"])["action"] in ("wander", "eat") for r in rows)
    assert "<think>" not in rows[0].messages[2]["content"]
    # the teacher sees the full hidden prompt, the saved rows only the short public one
    assert len(seen[0]) > 3 * len(rows[0].messages[0]["content"])


def test_recorded_turns_skip_failures(tmp_path):
    path = tmp_path / "turns.jsonl"
    base = {"name": "Bot", "persona": "chill", "allowed": {"chat": True}, "prompt": "SITUATION\nx"}
    path.write_text("\n".join(json.dumps(row) for row in [
        {**base, "reply": '{"action": "wander", "thought": "go"}', "ok": True},
        {**base, "reply": '{"action": "mine", "block": "stone"}', "ok": False},
        {**base, "reply": "not json", "ok": True},
        {**base, "reply": '{"action": "say", "message": "hi"}', "ok": True, "allowed": {"chat": False}},
    ]) + "\nbroken line\n", encoding="utf-8")
    rows = ds.recorded_examples(path)
    assert len(rows) == 1 and rows[0].source == "recorded"
    assert json.loads(rows[0].messages[2]["content"]) == {"thought": "go", "goal": "", "action": "wander"}
    assert ds.recorded_examples(tmp_path / "missing.jsonl") == []


def test_playtime_becomes_turns(tmp_path):
    from pymc_bot.playtime import synthesize_playtime

    synthesize_playtime(tmp_path / "play", minutes=3, seed=3)
    rows = ds.playtime_examples(tmp_path / "play", random.Random(1), limit=200)
    assert rows and {r.source for r in rows} == {"playtime"}
    assert {parse_decision(r.messages[2]["content"]).action for r in rows} <= {"fight", "mine", "goto", "jump", "look_around"}


# -------------------------------------------------------------------- wiki
PAGE_HTML = """<div><p>The <b>creeper</b> is a common hostile mob that sneaks up on players and explodes, destroying blocks.
[1]</p><h2><span>Spawning</span></h2><p>Creepers spawn in the Overworld in groups of up to four at a light level of 0,
on the surface at night and in caves at any time.</p><h2>Trivia</h2><p>Short.</p><h2>Drops</h2><p>tiny</p></div>"""


def _wiki_transport(calls):
    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        if params.get("list") == "categorymembers":
            return httpx.Response(200, json={"query": {"categorymembers": [
                {"title": "Creeper"}, {"title": "Damage/Mobs"}, {"title": "Ankle Monitor"}, {"title": "Bleach"}]}})
        title = params["page"]
        cats = [{"category": "Joke_items"}] if title == "Ankle Monitor" else [{"category": "Education_Edition"}] if title == "Bleach" else [{"category": "Hostile_mobs"}]
        return httpx.Response(200, json={"parse": {"title": title, "text": PAGE_HTML, "categories": cats}})

    return httpx.MockTransport(handler)


def test_wiki_fetch_filters_noise_and_caches(tmp_path):
    calls: list[dict] = []
    pages = ds.fetch_wiki(["Mobs"], limit=10, cache_dir=tmp_path, transport=_wiki_transport(calls), log=lambda _m: None, delay=0)
    assert [p.title for p in pages] == ["Creeper"]
    assert all(c.get("page") != "Damage/Mobs" for c in calls)
    assert calls[0]["cmnamespace"] == "0"
    assert "explodes" in pages[0].lead and ("Spawning", pages[0].sections[0][1]) == pages[0].sections[0]
    # second run: everything from the cache (no parse requests)
    calls.clear()
    again = ds.fetch_wiki(["Mobs"], limit=10, cache_dir=tmp_path, transport=_wiki_transport(calls), log=lambda _m: None, delay=0)
    assert [p.title for p in again] == ["Creeper"] and not any("page" in c for c in calls)
    rows = ds.wiki_examples(pages, random.Random(1))
    questions = [r.messages[1]["content"] for r in rows]
    assert len(rows) == 2 and any("creeper" in q.lower() and "spawn" in q.lower() for q in questions)
    assert all("[1]" not in r.messages[2]["content"] for r in rows)


# -------------------------------------------------------------------- build
def test_build_writes_train_eval_and_card(built):
    out, card = built
    train, held = ds.read_jsonl(out / "train.jsonl"), ds.read_jsonl(out / "eval.jsonl")
    assert card["train"] == len(train) and card["eval"] == len(held) and held
    assert ds.validate_rows(train) == [] and ds.validate_rows(held) == []
    assert {"mobs", "facts", "rules", "chat"} <= set(card["sources"])
    assert {r["source"] for r in held} >= {"mobs", "rules", "chat"}  # stratified
    assert all(r.get("check") for r in held if r["kind"] == "turn")
    blob = (out / "train.jsonl").read_text(encoding="utf-8")
    assert not any(hidden in blob for hidden in _hidden_texts())
    assert json.loads((out / "dataset_card.json").read_text())["licenses"] == []  # no wiki text in this build


def test_build_is_deterministic(tmp_path):
    a = ds.build_dataset(tmp_path / "a", rule_turns=60, mob_rounds=1, log=lambda _m: None)
    b = ds.build_dataset(tmp_path / "b", rule_turns=60, mob_rounds=1, log=lambda _m: None)
    assert a["sources"] == b["sources"]
    strip = [json.loads(line)["messages"][2] for line in (tmp_path / "a" / "train.jsonl").read_text().splitlines()]
    assert strip == [json.loads(line)["messages"][2] for line in (tmp_path / "b" / "train.jsonl").read_text().splitlines()]


def test_validate_rows_reports_problems():
    assert ds.validate_rows([{"messages": [{"role": "user", "content": "x"}]}])
    assert ds.validate_rows([{"kind": "turn", "messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "nope"}]}])


# -------------------------------------------------------------------- trainer
def test_trainer_dry_run_and_presets(built, capsys):
    out, _ = built
    assert llm_train.main(["--dry-run", "--data", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "dry run: data is fine" in printed and "unsloth/Qwen3-4B-Instruct-2507" in printed
    assert llm_train.main(["--list-presets"]) == 0
    listing = capsys.readouterr().out
    assert all(name in listing for name in llm_train.PRESETS)
    assert llm_train.DEFAULT_PRESET == "minecraftism"
    for preset in llm_train.PRESETS.values():
        assert preset["family"] in llm_train.TURN_MARKERS and preset["model"].startswith("unsloth/")
        assert preset["train_gb"] <= 16  # one Kaggle T4


def test_trainer_is_standalone():
    source = Path(llm_train.__file__).read_text(encoding="utf-8")
    top = source.split("def find_data")[0]
    assert "pymc_bot" not in top.replace("pymc_bot llm", "").replace("pymc-minecraft", "")
    assert "import torch" not in top and "unsloth" not in top.split('"""', 2)[2].split("PRESETS")[0]


def test_trainer_rejects_bad_data(tmp_path, capsys):
    (tmp_path / "train.jsonl").write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
    assert llm_train.main(["--dry-run", "--data", str(tmp_path)]) == 2


def test_score_and_summary():
    turn = {"kind": "turn", "source": "rules", "check": {"action": "flee"}}
    assert llm_train.score(turn, '<think>run</think>{"action": "flee"}') == {"kind": "turn", "source": "rules", "json_ok": True, "action_ok": True}
    assert llm_train.score(turn, "flee!")["json_ok"] is False
    chat = {"kind": "turn", "source": "chat", "check": {"action": "say", "contains": ["20"]}}
    assert llm_train.score(chat, '{"action": "say", "message": "it has 20 hp"}')["contains_ok"] is True
    answer = {"kind": "qa", "source": "mobs", "check": {"contains": ["1.14"]}}
    assert llm_train.score(answer, "Added in 1.14.")["contains_ok"] is True
    summary = llm_train.summarize([llm_train.score(turn, '{"action":"flee"}'), llm_train.score(turn, '{"action":"eat"}')])
    assert summary == {"rows": 2, "json_valid_pct": 100.0, "action_match_pct": 50.0, "answer_keywords_pct": None}


def test_modelfile_helpers():
    text = llm_train.modelfile_text("m.gguf", "chatml", "{{ .Prompt }}")
    assert text.startswith("FROM ./m.gguf\n") and 'PARAMETER stop "<|im_end|>"' in text and "PARAMETER num_ctx 4096" in text
    assert llm_train.template_from_modelfile(text) == "{{ .Prompt }}"
    assert llm_train.template_from_modelfile("FROM x") is None


def test_notebook_is_valid_and_uses_the_script():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4 and nb["metadata"]["kaggle"]["accelerator"] == "nvidiaTeslaT4"
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "train_minecraft_llm.py" in code and "pip install unsloth" in code
    editions = re.search(r"^EDITIONS = (\[.*\])$", code, flags=re.M).group(1)
    assert json.loads(editions) == ["minecraftism"], "one edition fits a 12 h Kaggle session comfortably"
    assert all(name in llm_train.PRESETS for name in llm_train.EDITIONS) and llm_train.EDITIONS[0] == "minecraftism"
    assert 'get_secret("HF_TOKEN")' in code and 'os.environ["HF_TOKEN"]' in code
    assert not re.search(r"print\([^)]*(environ|get_secret)", code), "the token must never be printed"
    assert "--hub-repo" in code and "--dry-run" in code and "for edition in READY" in code
    assert all(c["cell_type"] == "markdown" or c["outputs"] == [] for c in nb["cells"])


# -------------------------------------------------------------------- import / eval
def test_import_creates_the_ollama_model_and_switches_the_bot(tmp_path, store: ConfigStore):
    folder = tmp_path / "pymc-minecraft"
    folder.mkdir()
    (folder / "pymc-minecraft-qwen3-4b.Q4_K_M.gguf").write_bytes(b"GGUF" + b"0" * 100)
    (folder / "pymc_model.json").write_text(json.dumps({"preset": "qwen3-4b", "family": "chatml", "compact_prompt": True}))
    (folder / "Modelfile").write_text('FROM ./old.gguf\nTEMPLATE """{{ .System }}{{ .Prompt }}"""\n')
    runs = []

    class Done:
        returncode, stdout, stderr = 0, "success", ""

    result = import_model(folder, store, run=lambda cmd, **kw: runs.append(cmd) or Done(), which=lambda _n: "/usr/bin/ollama")
    assert result.created and result.activated
    assert runs == [["ollama", "create", "pymc-minecraft", "-f", str(folder / "Modelfile")]]
    modelfile = (folder / "Modelfile").read_text()
    assert modelfile.startswith(f"FROM {(folder / 'pymc-minecraft-qwen3-4b.Q4_K_M.gguf').resolve().as_posix()}\n")
    assert '{{ .System }}{{ .Prompt }}' in modelfile
    settings = store.load()
    assert settings.ollama.enabled and settings.ollama.model == "pymc-minecraft"
    assert settings.agent.mode == "ollama" and settings.agent.compact_prompt is True


def test_import_without_ollama_explains(tmp_path, store: ConfigStore):
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    result = import_model(gguf, store, name="mc", activate=False, which=lambda _n: None)
    assert not result.created and not result.activated and "ollama create mc -f" in result.message
    assert store.load().ollama.model != "mc"
    with pytest.raises(FileNotFoundError):
        find_gguf(tmp_path / "nothing")


def test_evaluate_scores_a_model(built):
    out, _ = built
    rows = ds.read_jsonl(out / "eval.jsonl")
    answers = {json.dumps(r["messages"][1:-1]): r["messages"][-1]["content"] for r in rows}
    perfect = evaluate(rows, lambda messages, system: answers[json.dumps(messages)], log=lambda _m: None)
    assert perfect["json_valid_pct"] == 100.0 and perfect["action_match_pct"] == 100.0
    assert perfect["answer_keywords_pct"] == 100.0 and perfect["errors"] == 0 and "rules" in perfect["by_source"]

    def broken(messages, system):
        raise TimeoutError("slow")

    bad = evaluate(rows, broken, limit=5, log=lambda _m: None)
    assert bad["errors"] == 5 and bad["rows"] == 5


# -------------------------------------------------------------------- agent
class FakeOllama:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []
        self.base_url = self.model = ""
        self.temperature = self.timeout = 0

    def chat(self, messages, system=None):
        self.calls.append({"messages": messages, "system": system})
        return self.reply


def test_parse_decision_strips_think_blocks():
    assert parse_decision('<think>\nzombie close, run\n</think>\n\n{"action": "flee"}').action == "flee"
    assert parse_decision('<think>{"action": "fight"}</think>{"action": "eat"}').action == "eat"


def test_compact_prompt_and_turn_recording(live_bot, store, log):
    store.update({"agent": {"mode": "ollama", "compact_prompt": True, "record_turns": True, "human_chat": False,
                            "persona": "miner"}})
    fake = FakeOllama('{"thought": "explore", "goal": "find iron", "action": "wander"}')
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    ok, detail = agent.run_once()
    assert ok, detail
    assert fake.calls[-1]["system"].startswith(ds.COMPACT_TAG) and "miner - Miner" in fake.calls[-1]["system"]
    path = agent.turns_path()
    assert path == store.path.parent / "llm_turns.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert row["ok"] is True and row["persona"] == "miner" and row["reply"] == fake.reply
    assert "SITUATION" in row["prompt"] and row["decision"]["action"] == "wander"
    # and it becomes training data
    assert ds.recorded_examples(path)[0].messages[1]["content"] == row["prompt"]


def test_turns_are_not_recorded_by_default(live_bot, store, log):
    store.update({"agent": {"mode": "ollama", "human_chat": False}})
    fake = FakeOllama('{"action": "wander"}')
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    agent.run_once()
    assert not agent.turns_path().exists()
    assert not fake.calls[-1]["system"].startswith(ds.COMPACT_TAG)


# -------------------------------------------------------------------- CLI
def test_cli_build_presets_and_eval_errors(tmp_path, capsys):
    config = tmp_path / "cfg.json"
    out = tmp_path / "llm_data"
    assert cli(["llm", "build", "--no-wiki", "--turns", "40", "--out", str(out), "--config", str(config),
                "--kaggle-user", "me"]) == 0
    assert (out / "train_minecraft_llm.py").read_text() == Path(llm_train.__file__).read_text()
    assert json.loads((out / "dataset-metadata.json").read_text())["id"] == "me/pymc-minecraft-data"
    assert "llm import" in capsys.readouterr().out
    assert cli(["llm", "presets"]) == 0
    assert "qwen3-14b" in capsys.readouterr().out
    assert cli(["llm", "eval", "--data", str(tmp_path / "missing.jsonl"), "--config", str(config)]) == 2
    assert cli(["llm", "import", str(tmp_path / "nothing"), "--config", str(config)]) == 2
