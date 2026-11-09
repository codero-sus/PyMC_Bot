"""Temporary, local recovery from repeated failed AI actions."""

from pymc_bot.agent import AgentLoop, Decision, HeuristicPolicy
from pymc_bot.recovery import RecoveryMemory


def state(**changes):
    return {"server": {"host": "example", "port": 25565}, "dimension": "overworld",
            "position": {"x": 16, "y": 64, "z": 0}, "health": 20, "food": 20,
            "inventory": [], "entities": [], "players": [], **changes}


def test_recovery_is_bounded_temporary_and_context_sensitive():
    clock = [100.0]
    memory = RecoveryMemory(clock=lambda: clock[0])
    decision = Decision("mine", {"block": "diamond_ore"}, "ollama")
    snapshot = state()
    memory.record(decision, snapshot, False, "no ore here")
    assert not memory.should_defer(decision, snapshot)
    memory.record(decision, snapshot, False, "no ore here")
    assert memory.should_defer(decision, snapshot)
    assert not memory.should_defer(Decision("mine", {"block": "iron_ore"}), snapshot)
    assert not memory.should_defer(decision, state(position={"x": 32, "y": 64, "z": 0}))
    assert not memory.should_defer(decision, state(inventory=[{"name": "iron_pickaxe", "count": 1}]))
    memory.record(decision, snapshot, True, "worked")
    assert not memory.should_defer(decision, snapshot)
    memory.record(decision, snapshot, False, "no ore here")
    memory.record(decision, snapshot, False, "no ore here")
    clock[0] += 46
    assert not memory.should_defer(decision, snapshot)
    for i in range(40):
        memory.record(Decision("mine", {"block": str(i)}), snapshot, False)
    assert len(memory._entries) == 32


def test_recovery_does_not_block_emergencies_manual_or_cancellation():
    memory = RecoveryMemory()
    for name in ("flee", "eat", "totem", "block_projectile", "fight", "say", "stop"):
        action = Decision(name)
        for _ in range(3):
            memory.record(action, state(), False, "failed")
        assert not memory.should_defer(action, state())
    action = Decision("mine", {"block": "stone"})
    for detail in ("cancelled", "disconnected", "busy"):
        memory.record(action, state(), False, detail)
    assert not memory.should_defer(action, state())
    trained = Decision("mine", {"block": "stone"}, "trained:policy")
    memory.record(trained, state(), False, "failed")
    memory.record(trained, state(), False, "failed")
    assert not memory.should_defer(trained, state())


def test_agent_switches_after_repeated_failure_but_manual_execute_is_untouched(live_bot, store, log, monkeypatch):
    agent = AgentLoop(live_bot, store, log)
    calls = []
    monkeypatch.setattr(live_bot, "mine", lambda name: calls.append(name) or False)
    monkeypatch.setattr(agent._heuristic, "decide", lambda *_args: Decision("wait", {}, "heuristic"))
    decision = Decision("mine", {"block": "diamond_ore"}, "ollama:test")
    assert agent._act(decision)[0] is False
    assert agent._act(decision)[0] is False
    assert agent._act(decision) == (True, "waited")
    assert len(calls) == 2 and agent.status["recovery"]["skipped"] == 1
    assert any("deferred after repeated failures" in row["result"] for row in agent.status["memory"]["recent_actions"])
    assert agent.execute(decision, enforce_permissions=False)[0] is False
    assert len(calls) == 3


def test_hungry_brain_looks_for_food_instead_of_eating_empty_inventory():
    policy = HeuristicPolicy(seed=1)
    empty = state(health=18, food=3, inventory=[])
    decision = policy.decide(empty, [])
    assert decision.action == "wander" and "no food" in decision.thought
    animal = {"type": "minecraft:cow", "dist": 4, "hostile": False}
    assert policy.decide(state(health=18, food=3, entities=[animal]), []).action == "hunt"
    assert policy.decide(state(health=4, food=3, entities=[animal]), []).action == "wander"
    assert policy.decide(empty, [], {"movement": False}).action == "wait"


def test_low_level_trained_actions_do_not_fill_persistent_experience(live_bot, store, log, monkeypatch):
    store.update({"agent": {"persistent_memory": True}})
    agent = AgentLoop(live_bot, store, log)
    monkeypatch.setattr(agent, "execute", lambda _decision: (True, "moved"))
    agent._act(Decision("forward", {}, "trained:example"))
    assert agent.experience().context([])[1] == []


def test_failed_eat_and_invalid_navigation_are_reported(live_bot, store, log, monkeypatch):
    agent = AgentLoop(live_bot, store, log)
    monkeypatch.setattr(live_bot, "eat", lambda: False)
    ok, _ = agent.execute(Decision("eat"))
    assert not ok
    for value in (float("nan"), float("inf"), -float("inf")):
        ok, message = agent.execute(Decision("goto", {"x": value, "z": 0}))
        assert not ok and "finite" in message
