"""Small, deterministic, inventory-driven early-game dependency planner.

These fixed vanilla recipes are intentionally limited. The caller owns safety checks,
permission gates, attempt limits, cancellation and post-action verification. A proposed
step is not evidence that it succeeded or that the world contains the target block.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymc_bot.inventory import Inventory


@dataclass(frozen=True)
class Recipe:
    item: str
    cobblestone: int
    sticks: int
    planks: int = 0


RECIPES = {
    "wooden_pickaxe": Recipe("wooden_pickaxe", cobblestone=0, sticks=2, planks=3),
    "stone_pickaxe": Recipe("stone_pickaxe", cobblestone=3, sticks=2),
    "stone_sword": Recipe("stone_sword", cobblestone=2, sticks=1),
    "furnace": Recipe("furnace", cobblestone=8, sticks=0),
}
PICKAXES = frozenset(f"{kind}_pickaxe" for kind in ("wooden", "stone", "iron", "golden", "diamond", "netherite"))
LOGS = frozenset(f"{kind}_log" for kind in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry"))

# action, params, item whose inventory increase must verify the step
PlanStep = tuple[str, dict[str, Any], str]


def _mine(item: str) -> PlanStep:
    # The goal controller decides between mining the mapped block and collecting a drop.
    return "", {}, item


def _craft(item: str, count: int = 1, *, no_station: bool = False) -> PlanStep:
    return "craft", {"item": item, "count": count, "no_station": no_station}, item


def _plank_stack(inv: Inventory) -> int:
    """Largest single kind: never assume differently coloured planks form one recipe."""
    return max((inv.count(name) for name in inv.names() if name.endswith("_planks")), default=0)


def _ensure_planks(inv: Inventory, minimum: int) -> PlanStep | None:
    if _plank_stack(inv) >= minimum:
        return None
    # Prefer making the type already in inventory; avoids getting stuck with many one-plank stacks.
    logs = sorted((name for name in inv.names() if name in LOGS and inv.count(name)),
                  key=lambda name: (-inv.count(name.removesuffix("_log") + "_planks"), name))
    if logs:
        return _craft(logs[0].removesuffix("_log") + "_planks", 4, no_station=True)
    return _mine("oak_log")


def _ensure_sticks(inv: Inventory, minimum: int) -> PlanStep | None:
    if inv.count("stick") >= minimum:
        return None
    missing_planks = _ensure_planks(inv, 2)
    return missing_planks or _craft("stick", 4, no_station=True)


def _station_planks(inv: Inventory, for_recipe: int = 0) -> int:
    # A crafting table may already exist in the world, but without a reliable
    # observation we reserve enough planks to make one. An inventory table is enough.
    return for_recipe + (0 if inv.count("crafting_table") else 4)


def next_tech_step(snapshot: dict[str, Any], target: str) -> PlanStep | str:
    """Return one prerequisite/recipe step; caller must replan from a fresh snapshot.

    Table placement is allowed only after the caller has checked building and movement
    permissions. Unknown/modded recipes, tool durability and stone alternatives are not
    guessed. A failed step is bounded by the calling goal manager.
    """
    recipe = RECIPES.get(target)
    if recipe is None:
        return "unsupported tech target"
    try:
        inv = Inventory.from_snapshot(snapshot)
        # Each pickaxe is unstackable and one is needed for each output. This plans
        # the next output only, not a speculative batch.
        if recipe.cobblestone and inv.count("cobblestone") < recipe.cobblestone:
            if not (inv.names() & PICKAXES):
                sticks = _ensure_sticks(inv, 2)
                if sticks:
                    return sticks
                planks = _ensure_planks(inv, _station_planks(inv, 3))
                if planks:
                    return planks
                return _craft("wooden_pickaxe")
            return _mine("cobblestone")
        sticks = _ensure_sticks(inv, recipe.sticks)
        if sticks:
            return sticks
        planks = _ensure_planks(inv, _station_planks(inv, recipe.planks))
        if planks:
            return planks
        return _craft(recipe.item)
    except (TypeError, ValueError, OverflowError):
        return "waiting for a valid inventory observation"
