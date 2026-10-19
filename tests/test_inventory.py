"""The inventory model: slots, moves, and how a tidy player lays out the hotbar."""

from __future__ import annotations

import pytest

from pymc_bot.inventory import (
    Inventory,
    InventoryError,
    Stack,
    armor_score,
    armor_slot_for,
    food_score,
    max_stack,
    tool_for_block,
)


def messy() -> Inventory:
    """Everything stuffed in the backpack, nothing useful in the hotbar."""
    inv = Inventory()
    for index, (name, count) in enumerate(
        [("dirt", 40), ("bread", 5), ("oak_log", 12), ("stone_pickaxe", 1), ("iron_sword", 1), ("torch", 20),
         ("rotten_flesh", 3), ("iron_helmet", 1), ("cobblestone", 64), ("wooden_pickaxe", 1), ("cooked_beef", 4),
         ("iron_axe", 1), ("bow", 1)]
    ):
        inv.slots[9 + index] = Stack(name, count)
    return inv


def test_item_knowledge():
    assert tool_for_block("stone") == tool_for_block("deepslate_iron_ore") == "pickaxe"
    assert tool_for_block("oak_log") == tool_for_block("crafting_table") == "axe"
    assert tool_for_block("grass_block") == tool_for_block("gravel") == "shovel"
    assert tool_for_block("white_wool") == "shears" and tool_for_block("cobweb") == "sword"
    assert tool_for_block("air") is None
    assert armor_slot_for("diamond_chestplate") == "chest" and armor_slot_for("turtle_helmet") == "head"
    assert armor_slot_for("iron_sword") is None
    assert armor_score("netherite_boots") > armor_score("diamond_boots") > armor_score("iron_boots") > armor_score("leather_boots")
    assert armor_score("diamond_chestplate") > armor_score("elytra")
    assert max_stack("iron_pickaxe") == 1 and max_stack("ender_pearl") == 16 and max_stack("cobblestone") == 64
    assert max_stack("bowl") == 64 and max_stack("water_bucket") == 1


def test_food_choice_is_sensible():
    # no steak wasted on a nearly full bar, rotten flesh only when starving, golden apples for emergencies
    assert food_score("cooked_beef", food=10) > food_score("bread", food=10) > 0
    assert food_score("rotten_flesh", food=12) < 0 < food_score("rotten_flesh", food=4)
    assert food_score("golden_apple", food=10, health=20) < 0 < food_score("golden_apple", food=10, health=6)
    assert food_score("melon_slice", food=19) > food_score("cooked_beef", food=19)


def test_pickup_fills_stacks_then_hotbar_then_backpack():
    inv = Inventory()
    assert inv.add("cobblestone", 100) == 0
    assert inv.slots[0] == Stack("cobblestone", 64) and inv.slots[1] == Stack("cobblestone", 36)
    inv.add("cobblestone", 30)
    assert inv.slots[1].count == 64 and inv.slots[2].count == 2
    for _ in range(6):
        inv.add("iron_sword")
    assert inv.slots[8].name == "iron_sword" and inv.slots[9] is None  # hotbar full now
    inv.add("dirt", 3)
    assert inv.slots[9] == Stack("dirt", 3)
    assert inv.remove("cobblestone", 70) == 70 and inv.count("cobblestone") == 60


def test_moves_swap_merge_and_equip():
    inv = Inventory()
    inv.slots[0] = Stack("dirt", 60)
    inv.slots[10] = Stack("dirt", 10)
    inv.slots[11] = Stack("iron_helmet")
    inv.slots[12] = Stack("shield")
    inv.move(10, 0)  # merge into the hotbar stack, leftovers stay
    assert inv.slots[0].count == 64 and inv.slots[10] == Stack("dirt", 6)
    inv.move(11, 1)
    assert inv.slots[1].name == "iron_helmet" and inv.slots[11] is None
    inv.equip(1, "head")
    assert inv.armor["head"].name == "iron_helmet" and inv.slots[1] is None
    inv.equip(12, "offhand")
    assert inv.offhand.name == "shield"
    inv.swap_hands()
    assert inv.held.name == "shield" and inv.offhand.name == "dirt"
    assert inv.unequip("head") >= 9
    with pytest.raises(InventoryError):
        inv.equip(0, "legs")  # a shield is not leggings
    with pytest.raises(InventoryError):
        inv.move(30, 1)  # empty slot
    with pytest.raises(InventoryError):
        inv.select(9)


def test_holding_from_the_backpack_swaps_into_the_selected_slot():
    inv = messy()
    inv.selected = 3
    index = inv.resolve("sword")
    inv.equip(index, "hand")
    assert inv.held.name == "iron_sword" and inv.selected == 3


def test_resolve_understands_player_words():
    inv = messy()
    assert inv.slots[inv.resolve("pickaxe")].name == "stone_pickaxe"  # the better one
    assert inv.slots[inv.resolve("food", food=10)].name == "cooked_beef"
    assert inv.slots[inv.resolve("blocks")].name == "cobblestone"
    assert inv.slots[inv.resolve("oak")].name == "oak_log"
    assert inv.slots[inv.resolve("minecraft:torch")].name == "torch"
    assert inv.resolve("diamond") is None
    assert inv.slots[inv.best_tool("oak_log")].name == "iron_axe"
    assert inv.best_tool("sand") is None  # no shovel: the hand will do


def test_plan_hotbar_lays_out_like_a_player():
    inv = messy()
    moves = inv.plan_hotbar(food=12)
    for source, target in moves:
        inv.move(source, target)
    bar = [stack.name if stack else None for stack in inv.slots[:9]]
    assert bar[0] == "iron_sword"
    assert bar[1] == "stone_pickaxe"
    assert bar[2] == "iron_axe"
    assert bar[4] == "cobblestone"
    assert bar[5] == "torch"
    assert bar[6] == "bow"
    assert bar[8] == "cooked_beef"
    assert inv.plan_hotbar(food=12) == []  # tidy now
    assert not inv.hotbar_is_messy(food=12)
    assert inv.count("dirt") == 40  # nothing lost


def test_plan_hotbar_is_stable_when_already_tidy():
    inv = Inventory()
    inv.slots[0] = Stack("diamond_sword")
    inv.slots[1] = Stack("iron_pickaxe")
    inv.slots[8] = Stack("bread", 10)
    inv.slots[20] = Stack("wooden_pickaxe")
    assert inv.plan_hotbar() == []


def test_armor_upgrades_and_description():
    inv = messy()
    inv.armor["head"] = Stack("leather_helmet")
    inv.armor["feet"] = Stack("diamond_boots")
    inv.slots[30] = Stack("iron_boots")
    upgrades = inv.armor_upgrades()
    assert set(upgrades) == {"head"} and inv.slots[upgrades["head"]].name == "iron_helmet"
    text = "\n".join(inv.describe(food=12))
    assert "Hotbar" in text and "[1*]" in text
    assert "head leather_helmet" in text and "Off-hand: empty" in text
    assert "wear_armor" in text and "organize" in text
    assert "{" not in text and "}" not in text  # the LLM narrative must stay JSON-free


def test_snapshot_round_trip_and_legacy_lists():
    inv = messy()
    inv.armor["chest"] = Stack("iron_chestplate")
    inv.offhand = Stack("shield")
    inv.selected = 4
    snap = inv.to_snapshot()
    again = Inventory.from_snapshot(snap)
    assert again.slots == inv.slots and again.armor == inv.armor and again.offhand == inv.offhand
    assert again.selected == 4 and snap["armor"] == ["iron_chestplate"]
    # backends that only report a flat list (older bridges, tests)
    legacy = Inventory.from_snapshot({
        "inventory": [{"name": "minecraft:bread", "count": 3}, {"name": "stone_pickaxe", "count": 1}],
        "held_item": "minecraft:stone_pickaxe", "armor": ["minecraft:iron_helmet"], "selected_slot": 1,
    })
    assert legacy.count("bread") == 3 and legacy.armor["head"].name == "iron_helmet"
    assert legacy.find("stone_pickaxe", "hotbar") is not None
