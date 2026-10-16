"""The human-like LLM brain: system prompt, turn prompt, memory and chat clean-up."""

from __future__ import annotations

import json
import random
import time

from pymc_bot.agent import AgentLoop, Decision, parse_decision
from pymc_bot.config import CortexSettings, OllamaSettings
from pymc_bot.mind import (
    LEGACY_SYSTEM_PROMPT_START,
    PERSONAS,
    Mind,
    build_system_prompt,
    build_turn_prompt,
    chat_lines,
    compass,
    humanize_message,
    mentions,
    typing_delay,
)

ALL_ON = {"movement": True, "chat": True, "mining": True, "attacking": True}


# ----------------------------------------------------------------- system prompt
def test_system_prompt_is_detailed_and_personal():
    prompt = build_system_prompt("Alex", "builder", "from Delhi, mixes Hindi and English", ALL_ON, 120)
    assert "You are Alex" in prompt
    assert "real" in prompt and "person" in prompt
    assert PERSONAS["builder"].who in prompt
    assert "from Delhi, mixes Hindi and English" in prompt
    assert "Reply with a single JSON object" in prompt  # contract other tests rely on
    assert '"thought"' in prompt and '"goal"' in prompt
    assert "120" in prompt  # chat length limit
    for action in ("say", "goto", "follow", "wander", "mine", "attack", "eat", "wait"):
        assert action in prompt
    assert len(prompt) > 3000  # detailed, not a one-liner


def test_system_prompt_is_honest_about_being_a_bot():
    prompt = build_system_prompt("Alex", "friendly", "", ALL_ON, 100).lower()
    assert "are you a bot" in prompt
    assert "honest" in prompt
    assert "never claim to be human" in prompt


def test_system_prompt_respects_switched_off_actions_and_unknown_personas():
    prompt = build_system_prompt("Bob", "pirate", "x" * 2000, {"movement": True, "chat": False,
                                                              "mining": True, "attacking": False}, 80)
    assert PERSONAS["friendly"].who in prompt  # unknown persona -> default
    assert "x" * 601 not in prompt  # owner notes are capped
    assert 'never use "say"' in prompt
    assert "attack" in prompt.split("RULES", 1)[-1]


def test_every_persona_builds():
    for key in PERSONAS:
        assert f"You are P_{key}" in build_system_prompt(f"P_{key}", key, "", ALL_ON, 100)


def test_old_default_prompt_in_configs_switches_to_the_built_in_one():
    legacy = LEGACY_SYSTEM_PROMPT_START + " Respond with JSON only."
    assert OllamaSettings(system_prompt=legacy).system_prompt == ""
    assert CortexSettings(system_prompt=legacy).system_prompt == ""
    assert OllamaSettings(system_prompt="talk like a pirate").system_prompt == "talk like a pirate"
    assert OllamaSettings().system_prompt == ""


# ------------------------------------------------------------------- turn prompt
STATE = {
    "position": {"x": 10, "y": 64, "z": -5},
    "health": 6,
    "food": 5,
    "time_of_day": "night",
    "dimension": "minecraft:overworld",
    "held_item": "stone_pickaxe",
    "inventory": [{"name": "bread", "count": 3}, {"name": "cobblestone", "count": 40}],
    "players": [{"name": "Steve", "x": 14, "y": 64, "z": -5, "distance": 4.0}],
    "online_players": ["Alex", "Steve", "Notch"],
    "entities": [
        {"type": "minecraft:zombie", "dx": 0, "dy": 0, "dz": -6, "dist": 6, "hostile": True},
        {"type": "minecraft:cow", "dx": 5, "dy": 0, "dz": 5, "dist": 7, "hostile": False},
    ],
}


def _turn(history=None, mind=None):
    return build_turn_prompt(STATE, "Alex", mind or Mind(), history or [], ALL_ON)


def test_turn_prompt_reads_like_what_a_player_sees():
    now = time.time()
    history = [
        {"time": now - 30, "username": "Steve", "message": "alex can you {help} me?"},
        {"time": now - 3600, "username": "Old", "message": "ancient history"},
    ]
    text = _turn(history)
    narrative, data = text.split("EXACT DATA (JSON):", 1)
    assert "{" not in narrative and "}" not in narrative  # only the tail is JSON (the stub relies on it)
    assert "night" in narrative and "Health 6/20" in narrative and "Food 5/20" in narrative
    assert "Steve (4 m east)" in narrative
    assert "zombie 6 m north" in narrative
    assert "cow x1" in narrative
    assert "bread x3" in narrative
    assert "Notch" in narrative
    assert "Also online (far away): Notch." in narrative  # never lists the bot itself
    assert "Waiting for your answer" in narrative and "(help)" in narrative
    assert "ancient history" not in narrative
    payload = json.loads(data.split("\n", 1)[0])
    assert payload["position"] == STATE["position"]
    assert payload["players"][0]["name"] == "Steve"


def test_turn_prompt_carries_memory():
    mind = Mind(4)
    mind.note_decision(Decision("mine", {"block": "oak_log"}, "ollama", thought="need wood", goal="get wood"))
    mind.note_result(Decision("mine", {"block": "oak_log"}, "ollama"), False, "no oak_log nearby")
    text = _turn(mind=mind)
    assert "Current goal: get wood" in text
    assert "mine block=oak_log -> FAILED: no oak_log nearby" in text
    assert "need wood" in text


def test_unknown_inventory_is_not_reported_as_empty():
    state = {k: v for k, v in STATE.items() if k != "inventory"}
    assert "Inventory: unknown" in build_turn_prompt(state, "Alex", Mind(), [], ALL_ON)


def test_chat_lines_find_unanswered_messages():
    now = time.time()
    history = [
        {"time": now - 20, "username": "Steve", "message": "hey alex"},
        {"time": now - 10, "username": "Alex", "message": "hi"},
        {"time": now - 5, "username": "Steve", "message": "alex where are you?"},
    ]
    lines, unanswered = chat_lines(history, "Alex", nearby=1, now=now)
    assert len(lines) == 3
    assert len(unanswered) == 1 and "where are you" in unanswered[0]


def test_mentions_and_compass():
    assert mentions("hey Alex!", "Alex")
    assert mentions("@alex come here", "Alex")
    assert not mentions("alexander is cool", "Alex")
    assert compass(0, -10) == "north"
    assert compass(10, 0) == "east"
    assert compass(0, 10) == "south"
    assert compass(-10, 0) == "west"


# ------------------------------------------------------------------------ memory
def test_mind_tracks_goals_repeats_and_failures():
    mind = Mind(3)
    for _ in range(3):
        mind.note_decision(Decision("mine", {}, "ollama", goal="get wood"))
    assert mind.goal == "get wood" and mind.goal_turns == 3
    mind.note_decision(Decision("wander", {}, "ollama", goal="explore"))
    assert mind.goal_turns == 1
    for _ in range(5):
        mind.note_result(Decision("goto", {"x": 1, "y": 2, "z": 3}, "ollama"), False, "blocked")
    assert len(mind.actions) == 3  # bounded
    assert mind.failures_in_a_row() == 3
    mind.note_result(Decision("say", {"message": "hey there!"}, "ollama"), True, "said")
    assert mind.already_said("Hey there")
    assert not mind.already_said("something new")
    assert mind.failures_in_a_row() == 0
    mind.resize(10)
    assert mind.snapshot()["goal"] == "explore"


def test_parse_decision_reads_thought_and_goal():
    decision = parse_decision(
        '{"thought": "steve asked, answer", "goal": "help steve", "action": "say", "message": "omw"}', source="ollama:x"
    )
    assert decision.action == "say"
    assert decision.params == {"message": "omw"}
    assert decision.thought == "steve asked, answer" and decision.goal == "help steve"
    assert decision.to_dict()["thought"] == "steve asked, answer"
    plain = parse_decision('{"action": "wander", "reason": "bored"}')
    assert plain.thought == "bored" and plain.params == {}


# ---------------------------------------------------------------------- chat out
def test_humanize_message_cleans_llm_chat():
    assert humanize_message('"hey there"', "Alex") == "hey there"
    assert humanize_message("Alex: omw", "Alex") == "omw"
    assert humanize_message("<Alex> **sure**", "Alex") == "sure"
    assert humanize_message("/op me", "Alex") == "op me"  # never runs commands
    assert humanize_message("As an AI language model, I can help 😊", "Alex") == "I can help"
    assert humanize_message("line one\nline two", "Alex") == "line one line two"
    long = humanize_message("word " * 100, "Alex", 50)
    assert len(long) <= 50 and not long.endswith(" ")


def test_typing_delay_is_human_and_bounded():
    rng = random.Random(1)
    short, longer = typing_delay("hi", rng), typing_delay("x" * 60, rng)
    assert 0.5 <= short < longer <= 4.0
    assert typing_delay("y" * 1000, rng) == 4.0


# -------------------------------------------------------------------- agent loop
class FakeOllama:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[dict] = []
        self.base_url = self.model = ""
        self.temperature = self.timeout = 0

    def chat(self, messages, system=None):
        self.calls.append({"messages": messages, "system": system})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def test_agent_uses_the_human_prompt_and_memory(live_bot, store, log):
    store.update({"agent": {"mode": "ollama", "persona": "explorer", "persona_notes": "loves cherry groves",
                            "human_chat": False}})
    fake = FakeOllama(['{"thought": "lets look around", "goal": "explore", "action": "wander"}'])
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    ok, detail = agent.run_once()
    assert ok, detail
    system = fake.calls[-1]["system"]
    assert f"You are {live_bot.username}" in system and "loves cherry groves" in system
    assert "EXACT DATA (JSON):" in fake.calls[-1]["messages"][-1]["content"]
    status = agent.status
    assert status["last_decision"]["thought"] == "lets look around"
    assert status["memory"]["goal"] == "explore"
    assert status["persona"] == "explorer"
    assert any("💭 lets look around" in event["message"] for event in log.tail(30))
    # the second turn remembers the first
    agent.run_once()
    assert "Current goal: explore" in fake.calls[-1]["messages"][-1]["content"]


def test_agent_uses_a_custom_prompt_verbatim(live_bot, store, log):
    store.update({"agent": {"mode": "ollama"}, "ollama": {"system_prompt": "Only ever wander. JSON."}})
    fake = FakeOllama(['{"action": "wander"}'])
    AgentLoop(live_bot, store, log, ollama=fake).run_once()
    assert fake.calls[-1]["system"] == "Only ever wander. JSON."


def test_llm_does_not_repeat_itself_or_spam(live_bot, store, log):
    store.update({"agent": {"mode": "ollama", "human_chat": True, "chat_cooldown": 60}})
    fake = FakeOllama(['{"action": "say", "message": "nice base"}', '{"action": "say", "message": "nice base!"}',
                       '{"action": "say", "message": "anyone around?"}'])
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    ok, detail = agent.run_once()
    assert ok and detail == "said: nice base"
    ok, detail = agent.run_once()
    assert not ok and "already said" in detail
    ok, detail = agent.run_once()
    assert not ok and "held back" in detail  # unprompted chat waits for the cooldown


def test_llm_brain_mutes_canned_replies(live_bot, store, log):
    store.update({"agent": {"mode": "ollama", "greet_players": True}, "ollama": {"decision_interval": 300}})
    agent = AgentLoop(live_bot, store, log, ollama=FakeOllama(['{"action": "wait", "seconds": 1}']))
    agent.start()
    try:
        assert live_bot.llm_chat is True
    finally:
        agent.stop()
    assert live_bot.llm_chat is False


# ------------------------------------------------------------------ behaviours
def test_behaviour_options_show_names_but_never_the_prompt():
    from pymc_bot.mind import behaviours

    options = behaviours()
    keys = [option["key"] for option in options]
    assert {"friendly", "chill", "social", "companion", "builder", "explorer", "miner", "newbie", "pro", "quiet"} <= set(keys)
    for option in options:
        assert set(option) == {"key", "label", "emoji", "blurb"}
        assert option["label"] and option["blurb"] and option["emoji"]
        persona = PERSONAS[option["key"]]
        prompt_text = (persona.who, persona.playstyle, persona.voice)
        assert not any(part in " ".join(option.values()) for part in prompt_text)


def test_unknown_behaviour_falls_back_to_friendly():
    from pymc_bot.config import AppSettings

    assert AppSettings(agent={"persona": "PRO"}).agent.persona == "pro"
    assert AppSettings(agent={"persona": "pirate"}).agent.persona == "friendly"


def test_behaviours_endpoint_lists_choices_without_prompts(tmp_path):
    from fastapi.testclient import TestClient

    from pymc_bot.server import create_app

    with TestClient(create_app(tmp_path / "cfg.json")) as client:
        data = client.get("/api/ai/behaviours").json()
        assert data["selected"] == "friendly"
        assert len(data["behaviours"]) == len(PERSONAS)
        assert "Reply with a single JSON object" not in json.dumps(data)
        assert client.get("/api/ai/prompt").status_code in (404, 405)
        client.put("/api/config", json={"agent": {"persona": "companion"}})
        assert client.get("/api/ai/behaviours").json()["selected"] == "companion"
