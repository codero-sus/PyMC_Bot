"""Decision parsing, permissions and the agent loop."""

from __future__ import annotations

import time

import pytest

from pymc_bot.agent import ACTIONS, AgentLoop, Decision, HeuristicPolicy, parse_decision
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog


# ------------------------------------------------------------------- parsing
def test_parse_plain_json():
    decision = parse_decision('{"action": "wander"}')
    assert decision.action == "wander"
    assert decision.source == "ollama"


def test_parse_with_thinking_and_code_fence():
    raw = """Sure! Here is my move:
```json
{"action": "goto", "x": 12, "y": 64, "z": -7}
```
That should help."""
    decision = parse_decision(raw)
    assert decision.action == "goto"
    assert decision.params["x"] == 12
    assert decision.params["z"] == -7


def test_parse_nested_params_and_aliases():
    decision = parse_decision('{"name": "say", "params": {"message": "hi there"}}')
    assert decision.action == "say"
    assert decision.params["message"] == "hi there"


def test_parse_namespaced_action_and_case():
    assert parse_decision('{"action": "Minecraft:MINE", "block": "stone"}').action == "mine"


@pytest.mark.parametrize(
    "raw",
    [
        "no json at all",
        "{not json}",
        '{"message": "hello"}',
        "[]",
    ],
)
def test_parse_rejects_bad_output(raw):
    with pytest.raises(ValueError):
        parse_decision(raw)


def test_all_actions_are_documented():
    assert "wander" in ACTIONS and "mine" in ACTIONS and "say" in ACTIONS


# ----------------------------------------------------------------- heuristics
def test_heuristic_eats_when_hurt():
    policy = HeuristicPolicy(seed=1)
    decision = policy.decide({"health": 4, "food": 20, "players": []}, [])
    assert decision.action == "eat"
    assert decision.source == "heuristic"


def test_heuristic_replies_when_mentioned():
    policy = HeuristicPolicy(seed=2)
    decision = policy.decide({"health": 20, "food": 20, "players": []}, ["<Steve> PyMC_Bot where are you?"])
    assert decision.action == "say"


def test_heuristic_follows_nearby_players():
    policy = HeuristicPolicy(seed=3)
    state = {"health": 20, "food": 20, "players": [{"name": "Steve", "distance": 4}]}
    actions = [policy.decide(state, []).action for _ in range(6)]
    assert "follow" in actions


def test_heuristic_keeps_busy():
    policy = HeuristicPolicy(seed=4)
    state = {"health": 20, "food": 20, "players": [], "time_of_day": "day"}
    actions = {policy.decide(state, []).action for _ in range(20)}
    assert actions <= {"wander", "say", "mine"}
    assert "wander" in actions


# ---------------------------------------------------------------- permissions
def test_execute_blocks_disabled_movement(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    store.update({"agent": {"allow_movement": False}})
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.execute(Decision("wander", {}, "heuristic"))
    assert ok is False
    assert "movement is disabled" in detail


def test_manual_actions_bypass_movement_flag_but_not_attacking(
    store: ConfigStore, live_bot: MinecraftBot, log: EventLog
):
    store.update({"agent": {"allow_movement": False, "allow_attacking": False}})
    agent = AgentLoop(live_bot, store, log)

    ok, _detail = agent.execute(Decision("jump", {}, "manual"), enforce_permissions=False)
    assert ok is True

    ok, detail = agent.execute(Decision("attack", {"player": "Steve"}, "manual"), enforce_permissions=False)
    assert ok is False
    assert "attacking is disabled" in detail


def test_execute_unknown_action(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    ok, detail = AgentLoop(live_bot, store, log).execute(Decision("dance", {}, "manual"))
    assert ok is False and "unsupported action" in detail


def test_execute_goto_validates_coordinates(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.execute(Decision("goto", {"x": "abc"}, "manual"), enforce_permissions=False)
    assert ok is False and "numeric" in detail

    ok, _detail = agent.execute(Decision("goto", {"x": 3, "y": 64, "z": -3}, "manual"), enforce_permissions=False)
    assert ok is True


def test_execute_say_and_wait(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.execute(Decision("say", {"message": "hi"}, "manual"), enforce_permissions=False)
    assert ok and "hi" in detail
    ok, detail = agent.execute(Decision("wait", {}, "manual"), enforce_permissions=False)
    assert ok and detail == "waited"


# ------------------------------------------------------------------- the loop
def test_agent_loop_runs_and_stops(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    store.update({"ollama": {"enabled": False, "decision_interval": 0.5}})
    agent = AgentLoop(live_bot, store, log)
    assert agent.start() is True
    assert agent.start() is False  # already running
    time.sleep(1.6)
    assert agent.status["decisions"] >= 1
    assert agent.status["last_decision"] is not None
    assert agent.stop() is True
    assert agent.running is False


def test_agent_loop_waits_for_connection(store: ConfigStore, log: EventLog):
    backend_bot = MinecraftBot(store, log, auto_reconnect=False)
    agent = AgentLoop(backend_bot, store, log)
    assert agent.start() is True
    time.sleep(0.4)
    assert agent.status["decisions"] == 0  # never connected -> nothing decided
    agent.stop()


def test_run_once_records_the_decision(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.run_once()
    assert isinstance(ok, bool) and detail
    assert agent.status["decisions"] == 1
    assert agent.status["last_decision"]["source"] == "heuristic"


def test_disabled_ai_never_calls_ollama(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    """With ollama.enabled = False the loop must not touch the network."""
    store.update({"ollama": {"enabled": False, "base_url": "http://127.0.0.1:1"}})
    agent = AgentLoop(live_bot, store, log)
    ok, _detail = agent.run_once()
    assert isinstance(ok, bool)
    assert agent.status["errors"] == 0
