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
    assert actions <= {"wander", "say", "mine", "look_around"}  # no inventory known -> no slot fidgets
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


# ------------------------------------------------------- trained brain gates
def test_allowed_trained_actions_follow_the_switches(live_bot: MinecraftBot, store: ConfigStore, log: EventLog):
    agent = AgentLoop(live_bot, store, log)
    store.update({"agent": {"allow_movement": False, "allow_mining": False, "allow_attacking": False}})
    assert agent.allowed_trained_actions() == {"look", "none"}
    store.update({"agent": {"allow_movement": True, "allow_attacking": True, "allow_mining": True}})
    allowed = agent.allowed_trained_actions()
    assert {"forward", "jump", "attack", "use"} <= allowed
    assert "say" not in allowed  # the low level vocabulary only


def test_trained_attack_is_refused_when_attacking_is_off(
    live_bot: MinecraftBot, store: ConfigStore, log: EventLog, monkeypatch
):
    """A model trained on playtime must not be able to punch people when the config says no."""

    class StubPolicy:
        def act(self, bot, decision):  # pragma: no cover - must not be reached
            raise AssertionError("the policy must not run a forbidden action")

    store.update({"agent": {"allow_attacking": False}})
    agent = AgentLoop(live_bot, store, log)
    monkeypatch.setattr(agent, "policy", lambda settings=None: StubPolicy())
    decision = Decision(action="attack", params={"seconds": 0.2}, source="trained:policy")
    ok, detail = agent.execute(decision)
    assert ok is False and "attacking is disabled" in detail


def test_trained_use_is_gated_by_mining(
    live_bot: MinecraftBot, store: ConfigStore, log: EventLog, monkeypatch
):
    class StubPolicy:
        def act(self, bot, decision):  # pragma: no cover - must not be reached
            raise AssertionError("the policy must not run a forbidden action")

    store.update({"agent": {"allow_mining": False}})
    agent = AgentLoop(live_bot, store, log)
    monkeypatch.setattr(agent, "policy", lambda settings=None: StubPolicy())
    ok, detail = agent.execute(Decision(action="use", params={}, source="trained:policy"))
    assert ok is False and "mining is disabled" in detail


def test_heuristic_only_replies_to_new_other_player_messages():
    policy = HeuristicPolicy(seed=4)
    state = {"username": "Bot", "health": 20, "food": 20, "players": []}
    chat = ["<Steve> Bot hello"]
    assert policy.decide(state, chat).action == "say"
    assert policy.decide(state, chat).action != "say"
    assert policy.decide(state, chat + ["<Bot> Bot hello"]).action != "say"
    assert policy.decide(state, chat + ["<Bot> Bot hello", "<Steve> Bot hello"]).action == "say"


def test_heuristic_recognizes_zero_food_and_health():
    policy = HeuristicPolicy(seed=4)
    assert policy.decide({"health": 20, "food": 0}, []).action == "eat"
    assert policy.decide({"health": 0, "food": 20}, []).action == "eat"


def test_free_movement_validation_and_permissions(live_bot, store, log):
    agent = AgentLoop(live_bot, store, log)
    calls = []
    live_bot.controlled_move = lambda controls, seconds: calls.append((controls, seconds)) or True
    assert agent.execute(Decision("move_controls", {"controls": ["forward", "left"], "seconds": 0.5}))[0]
    assert calls == [({"forward": True, "left": True}, 0.5)]
    for controls, seconds in [(["left", "right"], 1), (["attack"], 1), (["forward"], float("nan")), (["forward"], 10)]:
        assert not agent.execute(Decision("move_controls", {"controls": controls, "seconds": seconds}))[0]
    store.update({"agent": {"allow_movement": False}})
    assert not agent.execute(Decision("move_controls", {"controls": ["forward"]}))[0]


def test_controls_are_released_if_press_fails(live_bot):
    from pymc_bot.backends import BackendError

    calls = []

    def control(name, pressed):
        calls.append((name, pressed))
        if name == "left" and pressed:
            raise BackendError("simulated failure")

    live_bot.backend.set_control = control
    assert not live_bot.controlled_move({"forward": True, "left": True}, 0.1)
    assert ("forward", False) in calls and ("left", False) in calls


def test_stop_during_slow_decision_does_not_execute_it(live_bot, store, log):
    import threading

    agent = AgentLoop(live_bot, store, log)
    entered, release = threading.Event(), threading.Event()
    actions = []

    def decide(*_args):
        entered.set()
        release.wait(3)
        return Decision("say", {"message": "too late"}, source="ollama")

    agent._decide = decide
    agent._act = lambda decision: actions.append(decision)
    assert agent.start()
    assert entered.wait(2)
    assert agent.stop(wait=False)
    assert live_bot.cancelled
    release.set()
    agent._thread.join(timeout=3)
    assert not agent.running and not actions and not live_bot.cancelled


def test_heuristic_uses_totem_when_low_but_flees_immediate_threat():
    state = {"health": 6, "food": 20, "inventory": [{"name": "totem_of_undying", "count": 1}]}
    policy = HeuristicPolicy(seed=1)
    assert policy.decide(state, []).action == "totem"
    assert policy.decide(state, [], {"inventory": False}).action == "wander"
    threatened = {**state, "entities": [{"type": "zombie", "hostile": True, "dist": 3}]}
    assert policy.decide(threatened, []).action == "flee"
    assert policy.decide({**state, "offhand": "totem_of_undying"}, []).action == "wander"


def test_heuristic_ranged_defense_reassesses_and_respects_permissions():
    policy = HeuristicPolicy(seed=1)
    state = {"health": 20, "food": 20, "inventory": [{"name": "bow", "count": 1}, {"name": "arrow", "count": 5}],
             "entities": [{"type": "minecraft:skeleton", "hostile": True, "dist": 12}]}
    decision = policy.decide(state, [])
    assert decision.action == "shoot" and decision.params == {"mob": "skeleton", "weapon": "bow", "shots": 1}
    assert policy.decide(state, [], {"fighting": False}).action != "shoot"
    assert policy.decide(state, [], {"inventory": False}).action != "shoot"
    assert policy.decide({**state, "inventory": [{"name": "bow", "count": 1}]}, []).action != "shoot"
    neutral = {**state, "entities": [{"type": "minecraft:spider", "hostile": True, "dist": 12}]}
    assert policy.decide(neutral, []).action != "shoot"
    neutral["entities"][0]["dist"] = 3
    assert policy.decide(neutral, []).action != "fight"


def test_heuristic_backs_away_from_close_creeper_and_detects_zero_distance():
    policy = HeuristicPolicy(seed=1)
    state = {"health": 20, "food": 20, "entities": [{"type": "minecraft:creeper", "hostile": True, "dist": 0}]}
    assert policy.decide(state, []).action == "flee"
    assert policy.decide({**state, "food": 2}, []).action == "flee"
    state["entities"][0]["type"] = "minecraft:zombie"
    assert policy.decide(state, []).action == "fight"
