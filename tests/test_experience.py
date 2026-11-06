"""Bounded, isolated, erasable local bot experience."""

import time

from pymc_bot.experience import MAX_AGE, MAX_OUTCOMES, MAX_USERS, Experience
from pymc_bot.mind import Mind, build_turn_prompt


def test_experience_persists_bounded_relevant_history_and_outcomes(tmp_path):
    now = time.time()
    memory = Experience(tmp_path, "server:25565/bot")
    history = [{"username": f"Player{i}", "message": f"hey Bot {i}", "time": now + i / 100}
               for i in range(10)]
    memory.observe(history, "Bot")
    for i in range(12):
        memory.outcome("mine", False, f"no ore {i}", "try elsewhere")
    restored = Experience(tmp_path, "server:25565/bot")
    assert len(restored.data["users"]) == MAX_USERS
    assert len(restored.data["outcomes"]) == MAX_OUTCOMES
    messages, outcomes = restored.context(["Player9"])
    assert messages == ["<Player9> hey Bot 9"]
    assert len(outcomes) == 4 and "try elsewhere" in outcomes[-1]
    assert Experience(tmp_path, "other-server:25565/bot").context(["Player9"]) == ([], [])
    assert Experience(tmp_path, "server:25565/otherbot").context(["Player9"]) == ([], [])
    text = build_turn_prompt({}, "Bot", Mind(), [], {}, experience=(messages, outcomes))
    assert "Past experience" in text and "not instructions" in text and "Player9" in text


def test_filters_private_obvious_secrets_and_own_chat(tmp_path):
    now = time.time()
    memory = Experience(tmp_path, "a")
    memory.observe([
        {"username": "Bot", "message": "hey Bot", "time": now},
        {"username": "Steve", "message": "Bot password is abc", "time": now},
        {"username": "Steve", "message": "hi someone", "time": now},
        {"username": "Steve", "message": "/tell Bot hi", "time": now},
        {"username": "Steve", "message": "Bot build a {house}", "time": now},
    ], "Bot")
    assert memory.context(["Steve"])[0] == ["<Steve> Bot build a (house)"]
    memory.outcome("say", True, "said a private message")
    memory.outcome("mine", False, "token abc")
    assert memory.context(["Steve"])[1] == []


def test_no_readding_old_chat_or_after_clear(tmp_path):
    now = time.time() - 1
    history = [{"username": "Steve", "message": f"Bot {i}", "time": now + i / 100} for i in range(8)]
    memory = Experience(tmp_path, "a")
    memory.observe(history, "Bot")
    before = memory.context(["Steve"])
    memory.observe(history, "Bot")
    assert memory.context(["Steve"]) == before
    memory.clear()
    assert not memory.path.exists()
    memory.observe(history, "Bot")
    assert memory.context(["Steve"]) == ([], [])


def test_expired_and_bad_files_do_not_crash(tmp_path):
    memory = Experience(tmp_path, "a")
    memory.observe([{"username": "Steve", "message": "Bot old", "time": time.time() - MAX_AGE - 1}], "Bot")
    assert memory.context(["Steve"]) == ([], [])
    memory.path.parent.mkdir(parents=True, exist_ok=True)
    memory.path.write_text("not json")
    assert Experience(tmp_path, "a").context(["Steve"]) == ([], [])


def test_agent_persists_and_reuses_context_across_restart(live_bot, store, log):
    from pymc_bot.agent import AgentLoop, Decision

    store.update({"agent": {"mode": "heuristic", "persistent_memory": True}})
    first = AgentLoop(live_bot, store, log)
    live_bot._on_chat("Steve", f"{live_bot.username} help me mine")
    first._act(Decision("wait", {}, "heuristic", thought="take a breath"))
    second = AgentLoop(live_bot, store, log)
    prompt = second._build_prompt(store.settings, live_bot.snapshot())
    assert "help me mine" in prompt and "take a breath" in prompt
    second.clear_experience()
    assert not second.experience().path.exists()
    second.experience().observe(live_bot.chat_history, live_bot.username)
    assert second.experience().context(["Steve"]) == ([], [])
    store.update({"agent": {"persistent_memory": False}})
    second._act(Decision("wait", {}, "heuristic"))
    assert not second.experience().path.exists()


def test_memory_clear_api_and_config(tmp_path):
    from fastapi.testclient import TestClient

    from pymc_bot.server import create_app

    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        assert client.get("/api/config").json()["config"]["agent"]["persistent_memory"] is True
        assert client.put("/api/config", json={"agent": {"persistent_memory": False}}).json()["config"]["agent"]["persistent_memory"] is False
        assert client.delete("/api/ai/memory").json()["ok"] is True
