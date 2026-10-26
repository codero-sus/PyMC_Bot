"""Living like a player: hotbar, armour, items, mobs and the built-in brain that uses them."""

from __future__ import annotations

from fastapi.testclient import TestClient

from pymc_bot.agent import ACTIONS, AgentLoop, Decision, HeuristicPolicy
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog
from pymc_bot.inventory import Inventory, Stack
from pymc_bot.mind import ACTION_REFERENCE, build_system_prompt


def bar(bot: MinecraftBot) -> list[str | None]:
    return [stack.name if stack else None for stack in bot.inventory().slots[:9]]


# ------------------------------------------------------------------ bot methods
def test_select_hold_and_move_between_hotbar_and_backpack(live_bot: MinecraftBot):
    assert live_bot.select_slot(5)
    assert live_bot.inventory().selected == 4
    assert not live_bot.select_slot(10) and "1-9" in live_bot.last_failure

    assert live_bot.hold("bread")  # bread starts in the backpack
    inv = live_bot.inventory()
    assert inv.held.name == "bread" and inv.find("bread", "backpack") is None

    assert live_bot.to_inventory(item="bread")
    assert live_bot.inventory().find("bread", "backpack") is not None
    assert live_bot.to_hotbar("bread", slot=9)
    assert bar(live_bot)[8] == "bread"
    assert not live_bot.to_hotbar("diamond") and "no diamond" in live_bot.last_failure


def test_wear_armor_equip_offhand_and_swap(live_bot: MinecraftBot):
    assert live_bot.wear_best_armor()
    assert live_bot.snapshot()["armor_slots"]["head"]["name"] == "leather_helmet"
    assert not live_bot.wear_best_armor()  # nothing better left

    assert live_bot.equip("torch", to="offhand")
    assert live_bot.snapshot()["offhand"] == "torch"
    held = live_bot.inventory().held.name
    assert live_bot.swap_hands()
    snap = live_bot.snapshot()
    assert snap["held_item"] == "torch" and snap["offhand"] == held
    assert live_bot.unequip("helmet")
    assert live_bot.snapshot()["armor_slots"]["head"] is None


def test_organize_hotbar_like_a_player(live_bot: MinecraftBot):
    assert live_bot.organize_hotbar()
    layout = bar(live_bot)
    assert layout[0] == "iron_sword" and layout[1] == "iron_pickaxe" and layout[2] == "stone_axe"
    assert layout[5] == "torch" and layout[8] == "bread"
    assert not live_bot.organize_hotbar()  # already tidy


def test_mining_switches_to_the_right_tool(live_bot: MinecraftBot):
    assert live_bot.inventory().held.name == "iron_sword"
    assert live_bot.mine("stone")
    assert live_bot.inventory().held.name == "iron_pickaxe"
    assert live_bot.mine("oak_log")
    assert live_bot.inventory().held.name == "stone_axe"


def test_eat_fight_collect_drop_and_place(live_bot: MinecraftBot):
    live_bot.backend._food = 9.0
    assert live_bot.eat()
    assert live_bot.snapshot()["food"] > 9

    assert live_bot.fight("zombie")
    assert live_bot.inventory().held.name == "iron_sword"  # took the weapon out first
    assert live_bot.stats["mobs_killed"] == 1
    assert not live_bot.fight("zombie")  # it is dead (respawns later)

    assert live_bot.collect_items(radius=40)
    assert live_bot.inventory().count("rotten_flesh") == 1

    assert live_bot.drop("rotten_flesh")
    assert live_bot.inventory().count("rotten_flesh") == 0
    before = live_bot.inventory().count("dirt")
    assert live_bot.place_block("dirt")
    assert live_bot.inventory().count("dirt") == before - 1
    assert live_bot.stats["blocks_placed"] == 1


def test_sleep_only_at_night(live_bot: MinecraftBot):
    assert not live_bot.sleep_in_bed() and "night" in live_bot.last_failure
    live_bot.backend._world_time = 18000.0
    assert live_bot.snapshot()["time_of_day"] == "night"
    assert live_bot.sleep_in_bed()
    assert live_bot.snapshot()["time_of_day"] == "day"


def test_give_walks_over_and_tosses(live_bot: MinecraftBot):
    assert live_bot.give("Steve", "bread", 1)
    assert live_bot.inventory().count("bread") == 2
    assert not live_bot.give("Nobody", "bread")


def test_emotes_and_look_around(live_bot: MinecraftBot):
    for kind in ("wave", "crouch", "nod", "shake", "spin"):
        assert live_bot.emote(kind)
    assert live_bot.look_around()


# ------------------------------------------------------------------ the agent
def test_every_new_action_is_documented_for_the_llm():
    for action in ("hold_item", "select_slot", "to_hotbar", "to_inventory", "organize", "wear_armor", "equip",
                   "unequip", "swap_hands", "drop", "give", "fight", "flee", "collect", "sleep", "place",
                   "use_item", "emote", "look_around"):
        assert action in ACTIONS
        assert action in ACTION_REFERENCE


def test_permissions_gate_the_ai_but_not_the_owner(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.execute(Decision("place", {"block": "dirt"}, "ollama"))
    assert not ok and "building is disabled" in detail  # off by default for the AI
    ok, _ = agent.execute(Decision("place", {"block": "dirt"}, "manual"), enforce_permissions=False)
    assert ok
    store.update({"agent": {"allow_inventory": False, "allow_mob_fighting": False}})
    assert "inventory actions are disabled" in agent.execute(Decision("organize", {}, "ollama"))[1]
    assert "fighting mobs is disabled" in agent.execute(Decision("fight", {}, "ollama"))[1]
    prompt = build_system_prompt(allowed=agent.allowed(store.settings))
    assert "Inventory actions are switched off" in prompt and "Fighting mobs is switched off" in prompt


def test_execute_runs_life_actions(store: ConfigStore, live_bot: MinecraftBot, log: EventLog):
    agent = AgentLoop(live_bot, store, log)
    assert agent.execute(Decision("select_slot", {"slot": "3"}, "ollama")) == (True, "switched to hotbar slot 3")
    assert agent.execute(Decision("hold_item", {"item": "pickaxe"}, "ollama"))[0]
    assert live_bot.inventory().held.name == "iron_pickaxe"
    assert agent.execute(Decision("wear_armor", {}, "ollama"))[0]
    ok, detail = agent.execute(Decision("fight", {"mob": "warden"}, "ollama"))
    assert not ok and "warden" in detail
    ok, detail = agent.execute(Decision("equip", {"item": "diamond_helmet"}, "ollama"))
    assert not ok and "no diamond_helmet" in detail.lower()


def test_heuristic_lives_like_a_player():
    policy = HeuristicPolicy(seed=1)
    inv = Inventory()
    inv.slots[0] = Stack("dirt", 10)
    inv.slots[12] = Stack("iron_sword")
    inv.slots[13] = Stack("bread", 4)
    inv.slots[14] = Stack("iron_chestplate")
    base = {"health": 20, "food": 20, "players": [], **inv.to_snapshot()}

    zombie = {"type": "minecraft:zombie", "dist": 4, "dx": 4, "dz": 0, "hostile": True}
    assert policy.decide({**base, "entities": [zombie]}, []).action == "fight"
    assert policy.decide({**base, "health": 6, "entities": [zombie]}, []).action == "flee"
    assert policy.decide({**base, "entities": [zombie]}, [], {"fighting": False}).action != "fight"
    warden = {**zombie, "type": "minecraft:warden"}
    assert policy.decide({**base, "entities": [warden]}, []).action == "flee"
    assert policy.decide({**base, "food": 12}, []).action == "eat"
    assert policy.decide(base, []).action == "wear_armor"
    assert policy.decide(base, [], {"inventory": False}).action != "wear_armor"

    give = policy.decide(base, ["<Steve> PyMC_Bot can i have some bread?"])
    assert give.action == "give" and give.params["player"] == "Steve" and give.params["item"] == "bread"
    assert policy.decide(base, ["<Steve> PyMC_Bot where are you?"]).action == "say"


def test_heuristic_greets_tidies_and_collects():
    policy = HeuristicPolicy(seed=5)
    inv = Inventory()
    inv.slots[10] = Stack("iron_sword")
    inv.slots[11] = Stack("iron_pickaxe")
    inv.slots[12] = Stack("bread", 3)
    state = {"health": 20, "food": 20, **inv.to_snapshot(),
             "players": [{"name": "Alex", "distance": 3, "x": 0, "z": 3}],
             "ground_items": [{"item": "minecraft:oak_log", "count": 1, "dist": 3}]}
    actions = [policy.decide(state, []).action for _ in range(12)]
    assert actions.count("emote") == 1  # greets once, not every turn
    assert "organize" in actions and "collect" in actions


def test_panel_sends_flat_actions(store: ConfigStore, tmp_path):
    """The quick-action buttons post {"action": ..., "block": ...}: those fields must arrive."""
    from pymc_bot.server import create_app

    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        client.put("/api/config", json={"minecraft": {"backend": "simulated"}})
        client.post("/api/bot/start")
        for _ in range(100):
            if client.get("/api/status").json().get("state") == "connected":
                break
            import time

            time.sleep(0.05)
        reply = client.post("/api/bot/action", json={"action": "select_slot", "slot": 4}).json()
        assert reply["ok"], reply
        assert reply["decision"]["params"] == {"slot": 4}
        commands = client.get("/api/contract").json()["bridge_commands"]
        assert "attack_mob" in commands and "chop_tree" in commands
        client.post("/api/bot/stop")
