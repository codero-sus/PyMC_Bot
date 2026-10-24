"""Inventory knowledge: slots, the hotbar, armour, and how a real player organises them.

The bot sees its inventory the way a player does::

    hotbar   slots 0-8   (keys 1-9; one of them is selected = the item in the hand)
    backpack slots 9-35  (the three rows above the hotbar)
    armour   head / chest / legs / feet
    off-hand one item (shield, torch, totem...)

:class:`Inventory` holds that layout (from a backend snapshot, or as the simulated world's own
inventory) and knows the moves a person makes: pick a hotbar slot, put something from the backpack
into the hotbar and back, swap hands, wear armour, eat the right food, hold the right tool for a
block. :meth:`Inventory.plan_hotbar` lays the hotbar out the way most players do (weapon, pickaxe,
axe, shovel, blocks, torches, ranged, -, food).

Everything is plain Python with no Minecraft data files: item ids are matched by name
(``iron_pickaxe`` = tier ``iron`` + kind ``pickaxe``), which covers vanilla 1.8 - 26.x.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

HOTBAR_SIZE = 9
INVENTORY_SIZE = 36
ARMOR_SLOTS: tuple[str, ...] = ("head", "chest", "legs", "feet")
HAND_DESTINATIONS: tuple[str, ...] = ("hand", "offhand", *ARMOR_SLOTS)

# --------------------------------------------------------------------------- item knowledge
TIERS: dict[str, float] = {
    "wooden": 1.0, "golden": 1.5, "stone": 2.0, "leather": 1.0, "chainmail": 3.0,
    "iron": 3.5, "turtle": 3.5, "copper": 2.5, "diamond": 5.0, "netherite": 6.0,
}
TOOL_KINDS: tuple[str, ...] = ("pickaxe", "axe", "shovel", "hoe", "sword")
ARMOR_PIECES: dict[str, str] = {
    "helmet": "head", "cap": "head", "chestplate": "chest", "tunic": "chest", "elytra": "chest",
    "leggings": "legs", "pants": "legs", "boots": "feet",
}
UNSTACKABLE_WORDS: tuple[str, ...] = (
    "_sword", "_pickaxe", "_axe", "_shovel", "_hoe", "_helmet", "_chestplate", "_leggings", "_boots",
    "bow", "crossbow", "trident", "shield", "elytra", "fishing_rod", "shears", "flint_and_steel",
    "potion", "_bucket", "bucket", "totem_of_undying", "saddle", "mace", "_stew", "_soup",
)
SMALL_STACK_WORDS: tuple[str, ...] = ("ender_pearl", "snowball", "egg", "_sign", "honey_bottle", "banner")
OFFHAND_ITEMS: tuple[str, ...] = ("shield", "totem_of_undying", "torch", "firework_rocket", "map")

#: hunger points, saturation - what a player knows about food.
FOODS: dict[str, tuple[int, float]] = {
    "golden_carrot": (6, 14.4), "cooked_beef": (8, 12.8), "cooked_porkchop": (8, 12.8),
    "cooked_mutton": (6, 9.6), "cooked_salmon": (6, 9.6), "cooked_chicken": (6, 7.2),
    "cooked_rabbit": (5, 6.0), "cooked_cod": (5, 6.0), "baked_potato": (5, 6.0), "bread": (5, 6.0),
    "rabbit_stew": (10, 12.0), "mushroom_stew": (6, 7.2), "beetroot_soup": (6, 7.2),
    "suspicious_stew": (6, 7.2), "pumpkin_pie": (8, 4.8), "apple": (4, 2.4), "carrot": (3, 3.6),
    "melon_slice": (2, 1.2), "sweet_berries": (2, 0.4), "glow_berries": (2, 0.4), "cookie": (2, 0.4),
    "dried_kelp": (1, 0.6), "beetroot": (1, 1.2), "potato": (1, 0.6), "honey_bottle": (6, 1.2),
    "beef": (3, 1.8), "porkchop": (3, 1.8), "mutton": (2, 1.2), "chicken": (2, 1.2),
    "rabbit": (3, 1.8), "cod": (2, 0.4), "salmon": (2, 0.4), "tropical_fish": (1, 0.2),
    "golden_apple": (4, 9.6), "enchanted_golden_apple": (4, 9.6), "chorus_fruit": (4, 2.4),
    "rotten_flesh": (4, 0.8), "spider_eye": (2, 3.2), "poisonous_potato": (2, 1.2), "pufferfish": (1, 0.2),
}
#: Eaten only when there is nothing else (they hurt or waste something precious).
BAD_FOODS = {"rotten_flesh", "spider_eye", "poisonous_potato", "pufferfish", "chicken", "chorus_fruit"}
PRECIOUS_FOODS = {"golden_apple", "enchanted_golden_apple"}

BUILDING_BLOCKS: tuple[str, ...] = (
    "cobblestone", "cobbled_deepslate", "dirt", "stone", "deepslate", "netherrack", "andesite",
    "diorite", "granite", "tuff", "sandstone", "blackstone", "end_stone", "mud_bricks", "bricks",
    "stone_bricks",
)
BUILDING_SUFFIXES: tuple[str, ...] = ("_planks", "_log", "_wool", "_concrete", "_terracotta", "_bricks")
RANGED: tuple[str, ...] = ("bow", "crossbow", "trident")

_AXE_BLOCKS = ("_log", "_wood", "_planks", "_stem", "_hyphae", "crafting_table", "chest", "barrel",
               "bookshelf", "_fence", "_fence_gate", "ladder", "pumpkin", "melon", "_door", "_trapdoor",
               "_sign", "note_block", "jukebox", "lectern", "loom", "composter", "bamboo_block", "campfire")
_SHOVEL_BLOCKS = ("dirt", "grass_block", "sand", "red_sand", "gravel", "clay", "snow", "snow_block",
                  "soul_sand", "soul_soil", "mud", "podzol", "mycelium", "farmland", "coarse_dirt",
                  "rooted_dirt", "dirt_path", "powder_snow", "concrete_powder", "suspicious_sand",
                  "suspicious_gravel")
_HOE_BLOCKS = ("hay_block", "sculk", "moss_block", "nether_wart_block", "warped_wart_block", "shroomlight",
               "dried_kelp_block", "target", "sponge", "_leaves")


def short(name: Any) -> str:
    """``minecraft:iron_pickaxe`` -> ``iron_pickaxe`` (lower case, no namespace)."""
    text = str(name or "").strip().lower().replace(" ", "_")
    return text.split(":", 1)[1] if ":" in text else text


def tier(name: str) -> float:
    name = short(name)
    for material, value in TIERS.items():
        if name.startswith(material + "_"):
            return value
    return 1.0


def tool_kind(name: str) -> str | None:
    name = short(name)
    for kind in TOOL_KINDS:
        if name.endswith("_" + kind):
            return kind
    if name in ("shears", "mace", "trident"):
        return name
    return None


def armor_slot_for(name: str) -> str | None:
    name = short(name)
    if name in ("carved_pumpkin", "turtle_helmet") or name.endswith(("_head", "_skull")):
        return "head"
    for piece, slot in ARMOR_PIECES.items():
        if name == piece or name.endswith("_" + piece):
            return slot
    return None


def armor_score(name: str) -> float:
    name = short(name)
    if armor_slot_for(name) is None or name in ("carved_pumpkin",) or name.endswith(("_head", "_skull")):
        return 0.0
    if name == "elytra":
        return 0.5  # flying, not protection - a player keeps a chestplate on instead
    return tier(name)


def max_stack(name: str) -> int:
    name = short(name)
    if any(word in name for word in UNSTACKABLE_WORDS) and name not in ("bowl", "bone"):
        return 1
    if any(word in name for word in SMALL_STACK_WORDS):
        return 16
    return 64


def is_food(name: str) -> bool:
    return short(name) in FOODS


def is_building_block(name: str) -> bool:
    name = short(name)
    return name in BUILDING_BLOCKS or name.endswith(BUILDING_SUFFIXES)


def tool_for_block(block: str) -> str | None:
    """Which tool kind a player reaches for to break ``block`` (None = hand is fine)."""
    block = short(block)
    if not block or block in ("air", "torch", "flower", "grass", "short_grass", "tall_grass", "fern"):
        return None
    if block == "cobweb":
        return "sword"
    if block.endswith("_wool") or block in ("vine", "glow_lichen"):
        return "shears"
    if any(block == word or block.endswith(word) for word in _SHOVEL_BLOCKS):
        return "shovel"
    if any(block.endswith(word) or block == word.lstrip("_") for word in _AXE_BLOCKS):
        return "axe"
    if any(block.endswith(word) or block == word.lstrip("_") for word in _HOE_BLOCKS):
        return "hoe"
    return "pickaxe"  # stone, ores, deepslate, bricks, metal blocks, furnaces...


def weapon_score(name: str) -> float:
    name = short(name)
    kind = tool_kind(name)
    if kind == "sword":
        return 10.0 + tier(name)
    if kind == "mace":
        return 16.0
    if kind == "trident":
        return 14.0
    if kind == "axe":
        return 8.0 + tier(name)  # hits hard but slowly - most players still prefer the sword
    if kind in ("pickaxe", "shovel", "hoe"):
        return 2.0 + tier(name) / 2
    return 0.0


def food_score(name: str, food: float = 20.0, health: float = 20.0) -> float:
    """How much a player would like to eat ``name`` right now (<= 0: rather not)."""
    name = short(name)
    if name not in FOODS:
        return 0.0
    hunger, saturation = FOODS[name]
    if name in PRECIOUS_FOODS:
        return 30.0 if health <= 8 else -1.0  # golden apples are for emergencies
    if name in BAD_FOODS:
        return 0.5 if food <= 6 else -1.0  # only when starving
    missing = max(0.0, 20.0 - food)
    useful = min(float(hunger), missing)
    # what fills the bar counts, what would be wasted on a full bar counts against it
    return useful + saturation * 0.25 - (hunger - useful)


@dataclass
class Stack:
    name: str
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "count": self.count}


def _stack(entry: Any) -> Stack | None:
    if isinstance(entry, Stack):
        return Stack(entry.name, entry.count)
    if isinstance(entry, dict) and entry.get("name"):
        return Stack(short(entry["name"]), max(1, int(entry.get("count") or 1)))
    if isinstance(entry, str) and entry:
        return Stack(short(entry), 1)
    return None


class InventoryError(ValueError):
    """A move a player could not make (empty slot, full backpack...)."""


# --------------------------------------------------------------------------- the inventory
class Inventory:
    """Hotbar (0-8) + backpack (9-35) + armour + off-hand, with the moves a player makes."""

    def __init__(self) -> None:
        self.slots: list[Stack | None] = [None] * INVENTORY_SIZE
        self.armor: dict[str, Stack | None] = {slot: None for slot in ARMOR_SLOTS}
        self.offhand: Stack | None = None
        self.selected = 0

    # ------------------------------------------------------------- construction
    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any] | None) -> Inventory:
        """Build from a backend snapshot (``slots`` if the backend reports them, else the flat list)."""
        snapshot = snapshot or {}
        inv = cls()
        slots = snapshot.get("slots")
        if isinstance(slots, list) and slots:
            for index, entry in enumerate(slots[:INVENTORY_SIZE]):
                inv.slots[index] = _stack(entry)
        else:
            # Older backends only report the items: lay them out like a fresh pickup would.
            for entry in snapshot.get("inventory") or []:
                stack = _stack(entry)
                if stack:
                    inv.add(stack.name, stack.count)
            held = short(snapshot.get("held_item"))
            if held and inv.find(held, "hotbar") is None and inv.find(held) is not None:
                inv.move(inv.find(held), inv.first_empty("hotbar") or 0)
        armor = snapshot.get("armor_slots")
        if isinstance(armor, dict):
            for slot in ARMOR_SLOTS:
                inv.armor[slot] = _stack(armor.get(slot))
        else:
            for name in snapshot.get("armor") or []:
                slot = armor_slot_for(str(name))
                if slot and inv.armor[slot] is None:
                    inv.armor[slot] = _stack(name)
        inv.offhand = _stack(snapshot.get("offhand"))
        try:
            inv.selected = max(0, min(HOTBAR_SIZE - 1, int(snapshot.get("selected_slot") or 0)))
        except (TypeError, ValueError):
            inv.selected = 0
        return inv

    def copy(self) -> Inventory:
        return copy.deepcopy(self)

    def to_snapshot(self) -> dict[str, Any]:
        """The fields a backend snapshot carries (``inventory`` stays the flat list older code reads)."""
        held = self.held
        return {
            "slots": [stack.to_dict() if stack else None for stack in self.slots],
            "inventory": [stack.to_dict() for stack in self.slots if stack],
            "selected_slot": self.selected,
            "held_item": held.name if held else "",
            "armor_slots": {slot: (stack.to_dict() if stack else None) for slot, stack in self.armor.items()},
            "armor": [stack.name for stack in self.armor.values() if stack],
            "offhand": self.offhand.name if self.offhand else "",
        }

    # ------------------------------------------------------------- reading
    @property
    def held(self) -> Stack | None:
        return self.slots[self.selected]

    def items(self) -> list[tuple[int, Stack]]:
        return [(index, stack) for index, stack in enumerate(self.slots) if stack]

    def names(self) -> set[str]:
        return {stack.name for _, stack in self.items()}

    def count(self, name: str) -> int:
        name = short(name)
        return sum(stack.count for _, stack in self.items() if stack.name == name)

    @staticmethod
    def _range(where: str) -> range:
        if where == "hotbar":
            return range(0, HOTBAR_SIZE)
        if where in ("backpack", "main", "inventory"):
            return range(HOTBAR_SIZE, INVENTORY_SIZE)
        return range(0, INVENTORY_SIZE)

    def find(self, name: str, where: str = "any") -> int | None:
        """Slot of ``name`` (exact id; the hotbar is searched first, like a player looks)."""
        name = short(name)
        order = list(self._range(where))
        if where == "any":
            order = list(range(HOTBAR_SIZE)) + list(range(HOTBAR_SIZE, INVENTORY_SIZE))
        for index in order:
            stack = self.slots[index]
            if stack and stack.name == name:
                return index
        return None

    def first_empty(self, where: str = "any") -> int | None:
        for index in self._range(where):
            if self.slots[index] is None:
                return index
        return None

    def best(self, key, where: str = "any") -> int | None:
        """Slot with the highest positive ``key(name)``; ties prefer the hotbar / selected slot."""
        best_index, best_score = None, 0.0
        for index in self._range(where):
            stack = self.slots[index]
            if not stack:
                continue
            score = key(stack.name)
            bonus = (0.02 if index == self.selected else 0.01 if index < HOTBAR_SIZE else 0.0)
            if score > 0 and score + bonus > best_score:
                best_index, best_score = index, score + bonus
        return best_index

    def best_tool(self, block: str) -> int | None:
        kind = tool_for_block(block)
        if kind is None:
            return None
        return self.best(lambda name: tier(name) if tool_kind(name) == kind or (kind == "shears" and name == "shears") else 0.0)

    def best_weapon(self) -> int | None:
        return self.best(weapon_score)

    def best_food(self, food: float = 20.0, health: float = 20.0) -> int | None:
        return self.best(lambda name: food_score(name, food, health))

    def building_block(self) -> int | None:
        """The block stack a player would build with (the biggest pile of plain blocks)."""
        return self.best(lambda name: (2.0 if name in BUILDING_BLOCKS else 1.0) * self.count(name) if is_building_block(name) else 0.0)

    def armor_upgrades(self) -> dict[str, int]:
        """Armour slot -> inventory slot of a better piece than the one being worn."""
        upgrades: dict[str, int] = {}
        for slot in ARMOR_SLOTS:
            worn = self.armor[slot]
            worn_score = armor_score(worn.name) if worn else 0.0
            index = self.best(lambda name, slot=slot: armor_score(name) if armor_slot_for(name) == slot else 0.0)
            if index is not None and armor_score(self.slots[index].name) > worn_score:
                upgrades[slot] = index
        return upgrades

    def resolve(self, query: str, food: float = 20.0, health: float = 20.0) -> int | None:
        """Slot for what someone means: an exact id, or a word like "sword", "food", "blocks", "pickaxe"."""
        query = short(query)
        if not query:
            return None
        exact = self.find(query)
        if exact is not None:
            return exact
        words = {
            "sword": self.best(lambda n: tier(n) if tool_kind(n) == "sword" else 0.0),
            "weapon": self.best_weapon(),
            "food": self.best_food(min(food, 14.0), health),
            "block": self.building_block(),
            "blocks": self.building_block(),
        }
        if query in words:
            return words[query]
        for kind in ("pickaxe", "axe", "shovel", "hoe"):
            if query in (kind, kind + "s", "pick" if kind == "pickaxe" else kind):
                return self.best(lambda n, kind=kind: tier(n) if tool_kind(n) == kind else 0.0)
        for index, stack in self.items():  # "oak" -> oak_log, "steak" -> none
            if query in stack.name:
                return index
        return None

    # ------------------------------------------------------------- moves
    def _check(self, index: int) -> None:
        if not 0 <= int(index) < INVENTORY_SIZE:
            raise InventoryError(f"slot {index} does not exist (0-35)")

    def add(self, name: str, count: int = 1) -> int:
        """Pick items up like Minecraft does: top up stacks, then the first free hotbar slot, then the
        backpack. Returns what did not fit."""
        name = short(name)
        limit = max_stack(name)
        left = max(0, int(count))
        for _, stack in self.items():
            if left and stack.name == name and stack.count < limit:
                take = min(limit - stack.count, left)
                stack.count += take
                left -= take
        while left:
            index = self.first_empty("hotbar")
            if index is None:
                index = self.first_empty("backpack")
            if index is None:
                break
            take = min(limit, left)
            self.slots[index] = Stack(name, take)
            left -= take
        return left

    def remove(self, name: str, count: int = 1) -> int:
        """Take ``count`` of ``name`` out (selected slot first). Returns how many were removed."""
        name = short(name)
        wanted = max(0, int(count))
        removed = 0
        order = [self.selected] + [i for i in range(INVENTORY_SIZE) if i != self.selected]
        for index in order:
            stack = self.slots[index]
            if not stack or stack.name != name or removed >= wanted:
                continue
            take = min(stack.count, wanted - removed)
            stack.count -= take
            removed += take
            if stack.count <= 0:
                self.slots[index] = None
        return removed

    def move(self, source: int, target: int) -> None:
        """Click a slot, click another: swap them, or merge into a stack of the same item."""
        self._check(source)
        self._check(target)
        if source == target:
            return
        a, b = self.slots[source], self.slots[target]
        if a is None:
            raise InventoryError(f"slot {source} is empty")
        if b is not None and b.name == a.name and b.count < max_stack(a.name):
            take = min(max_stack(a.name) - b.count, a.count)
            b.count += take
            a.count -= take
            self.slots[source] = a if a.count > 0 else None
            return
        self.slots[source], self.slots[target] = b, a

    def select(self, slot: int) -> None:
        if not 0 <= int(slot) < HOTBAR_SIZE:
            raise InventoryError("hotbar slots are 1-9")
        self.selected = int(slot)

    def equip(self, index: int, destination: str) -> None:
        """Put the item in ``index`` into the hand, the off-hand or an armour slot (swapping)."""
        self._check(index)
        stack = self.slots[index]
        if stack is None:
            raise InventoryError(f"slot {index} is empty")
        if destination == "hand":
            if index < HOTBAR_SIZE:
                self.selected = index
            else:
                self.move(index, self.selected)
            return
        if destination == "offhand":
            self.slots[index], self.offhand = self.offhand, stack
            return
        if destination in ARMOR_SLOTS:
            if armor_slot_for(stack.name) != destination:
                raise InventoryError(f"{stack.name} is not worn on the {destination}")
            self.slots[index], self.armor[destination] = self.armor[destination], stack
            return
        raise InventoryError(f"unknown destination {destination!r} (hand, offhand, head, chest, legs, feet)")

    def unequip(self, destination: str) -> int:
        """Take armour / the off-hand item off into a free slot. Returns that slot."""
        if destination not in ("offhand", *ARMOR_SLOTS):
            raise InventoryError("unequip: offhand, head, chest, legs or feet")
        stack = self.offhand if destination == "offhand" else self.armor[destination]
        if stack is None:
            raise InventoryError(f"nothing on the {destination}")
        index = self.first_empty("backpack")
        if index is None:
            index = self.first_empty("hotbar")
        if index is None:
            raise InventoryError("the inventory is full")
        self.slots[index] = stack
        if destination == "offhand":
            self.offhand = None
        else:
            self.armor[destination] = None
        return index

    def swap_hands(self) -> None:
        """The F key: main hand <-> off-hand."""
        self.slots[self.selected], self.offhand = self.offhand, self.slots[self.selected]

    # ------------------------------------------------------------- organising
    def hotbar_roles(self, food: float = 20.0, health: float = 20.0) -> dict[int, int]:
        """Hotbar slot -> inventory slot of the item a tidy player keeps there."""
        picks: dict[int, int] = {}
        used: set[int] = set()

        def take(hotbar_slot: int, index: int | None) -> None:
            if index is not None and index not in used and hotbar_slot not in picks:
                picks[hotbar_slot] = index
                used.add(index)

        def best_of(key) -> int | None:
            candidates = [(key(stack.name), i) for i, stack in self.items() if i not in used and key(stack.name) > 0]
            return max(candidates)[1] if candidates else None

        take(0, best_of(lambda n: weapon_score(n) if tool_kind(n) in ("sword", "mace", "trident") or (tool_kind(n) == "axe" and not any(tool_kind(s.name) == "sword" for _, s in self.items())) else 0.0))
        take(1, best_of(lambda n: tier(n) if tool_kind(n) == "pickaxe" else 0.0))
        take(2, best_of(lambda n: tier(n) if tool_kind(n) == "axe" else 0.0))
        take(3, best_of(lambda n: tier(n) if tool_kind(n) == "shovel" else 0.0))
        take(4, best_of(lambda n: self.count(n) * (2.0 if n in BUILDING_BLOCKS else 1.0) if is_building_block(n) else 0.0))
        take(5, best_of(lambda n: 1.0 if n in ("torch", "soul_torch", "lantern") else 0.0))
        take(6, best_of(lambda n: 2.0 if n in RANGED else 1.0 if n == "water_bucket" else 0.0))
        take(8, best_of(lambda n: max(0.0, food_score(n, min(food, 12.0), health))))
        return picks

    def plan_hotbar(self, food: float = 20.0, health: float = 20.0) -> list[tuple[int, int]]:
        """The (source, target) clicks that lay the hotbar out like a tidy player's. Applies nothing."""
        wanted = {slot: self.slots[index].name for slot, index in self.hotbar_roles(food, health).items()}
        work = self.copy()
        moves: list[tuple[int, int]] = []
        arranged: set[int] = set()
        for slot in sorted(wanted):
            name = wanted[slot]
            current = work.slots[slot]
            if not (current and current.name == name):
                source = next(
                    (i for i, stack in work.items() if stack.name == name and i not in arranged and i != slot),
                    None,
                )
                if source is None:
                    continue
                work.move(source, slot)
                moves.append((source, slot))
            arranged.add(slot)
        return moves

    def hotbar_is_messy(self, food: float = 20.0, health: float = 20.0) -> bool:
        """Worth tidying: the weapon, pickaxe or food is not in the hotbar at all, or 3+ moves are needed."""
        moves = self.plan_hotbar(food, health)
        if len(moves) >= 3:
            return True
        roles = self.hotbar_roles(food, health)
        return any(roles.get(slot, 0) >= HOTBAR_SIZE for slot in (0, 1, 8) if slot in roles)

    # ------------------------------------------------------------- words
    def describe(self, food: float = 20.0, health: float = 20.0) -> list[str]:
        """The inventory as a player sees it, for the LLM prompt."""
        lines = []
        bar = []
        for index in range(HOTBAR_SIZE):
            stack = self.slots[index]
            text = f"{stack.name}" + (f" x{stack.count}" if stack and stack.count > 1 else "") if stack else "-"
            bar.append(f"[{index + 1}{'*' if index == self.selected else ''}] {text}")
        held = self.held
        lines.append(f"Hotbar (* = in your hand: {held.name if held else 'nothing'}): " + "  ".join(bar))
        worn = [f"{slot} {stack.name}" for slot, stack in self.armor.items() if stack]
        lines.append(
            "Wearing: " + (", ".join(worn) if worn else "no armour")
            + f". Off-hand: {self.offhand.name if self.offhand else 'empty'}."
        )
        backpack: dict[str, int] = {}
        for index in range(HOTBAR_SIZE, INVENTORY_SIZE):
            stack = self.slots[index]
            if stack:
                backpack[stack.name] = backpack.get(stack.name, 0) + stack.count
        free = sum(1 for index in range(HOTBAR_SIZE, INVENTORY_SIZE) if self.slots[index] is None)
        if backpack:
            listed = ", ".join(f"{name} x{count}" for name, count in sorted(backpack.items(), key=lambda kv: -kv[1])[:12])
            lines.append(f"Backpack ({27 - free}/27 slots used): {listed}.")
        else:
            lines.append("Backpack: empty.")
        hints = []
        upgrades = self.armor_upgrades()
        if upgrades:
            hints.append("better armour in your inventory (" + ", ".join(self.slots[i].name for i in upgrades.values()) + ") - wear_armor")
        if self.best_food(food, health) is None:
            hints.append("no food")
        if not any(tool_kind(stack.name) in ("pickaxe", "axe") for _, stack in self.items()):
            hints.append("no tools")
        if self.items() and self.hotbar_is_messy(food, health):
            hints.append("hotbar is messy - organize")
        if free == 0:
            hints.append("backpack full - drop junk")
        if hints:
            lines.append("Inventory notes: " + "; ".join(hints) + ".")
        return lines
