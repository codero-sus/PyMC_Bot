"""Bounded, observation-driven gathering goals for one bot (no generated code)."""
from __future__ import annotations

import math
import threading
from typing import Any

from pymc_bot.inventory import Inventory, short

# Explicit Minecraft item -> block mappings. No recipe invention or destructive arbitrary block IDs.
GATHERABLE = {
    **{f"{wood}_log": f"{wood}_log" for wood in ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry")},
    "cobblestone": "stone", "cobbled_deepslate": "deepslate", "coal": "coal_ore", "raw_iron": "iron_ore",
}
# The bridge searches for each named block; these variants may not exist on older servers.
ALTERNATES = {"coal": ("coal_ore", "deepslate_coal_ore"), "raw_iron": ("iron_ore", "deepslate_iron_ore")}


class GatherGoal:
    """A single in-memory gather goal; only a fresh inventory snapshot can prove completion."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._goal: dict[str, Any] | None = None
        self._inflight = False

    def start(self, item: str, count: int, identity: tuple[Any, ...]) -> dict[str, Any]:
        item = str(item).strip().lower().removeprefix("minecraft:")
        if item not in GATHERABLE and item != "torch":
            raise ValueError("unsupported item; choose a listed log, cobblestone, cobbled_deepslate, coal, raw_iron or torch")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
            raise ValueError("count must be an integer between 1 and 64")
        with self._lock:
            if self._inflight:
                raise RuntimeError("a goal action is still running; wait for it to finish")
            self._generation += 1
            self._goal = {"item": item, "count": count, "observed": None, "status": "active", "detail": "waiting for a fresh observation",
                          "attempts": 0, "failures": 0, "no_progress": 0, "variant": 0,
                          "last_action": "", "progress_item": item, "progress_before": None, "identity": identity}
            return self.snapshot()

    def cancel(self) -> bool:
        with self._lock:
            active = bool(self._goal and self._goal["status"] in {"active", "paused"})
            self._generation += 1
            if active:
                self._goal["status"] = "cancelled"
                self._goal["detail"] = "cancel requested; already-sent actions may still finish"
            return active

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._goal:
                return None
            return {k: v for k, v in self._goal.items() if k != "identity"} | {"in_flight": self._inflight}

    def next(self, state: dict[str, Any], identity: tuple[Any, ...], *, mining: bool, movement: bool,
             inventory: bool = True) -> tuple[str, dict[str, Any] | None]:
        """Return (kind, payload): none, wait, or one claimed action with a generation token."""
        with self._lock:
            g = self._goal
            if not g or g["status"] not in {"active", "paused"}:
                return "none", None
            if self._inflight:
                return "wait", None
            if g["identity"] != identity:
                g.update(status="failed", detail="bot or server changed; create a new goal")
                return "wait", None
            if not isinstance(state, dict) or state.get("status", "connected") != "connected":
                g.update(status="paused", detail="waiting for connection")
                return "wait", None
            count = _inventory_count(state, g["item"])
            if count is None:
                g.update(status="paused", detail="waiting for inventory observation")
                return "wait", None
            if g["observed"] is not None and count > g["observed"]:
                g["no_progress"] = 0
            g["observed"] = count
            if count >= g["count"]:
                g.update(status="completed", detail="target verified in inventory")
                return "wait", None
            if g["item"] == "torch" and not inventory:
                g.update(status="paused", detail="torch plan requires AI inventory permission")
                return "wait", None
            if not mining or not movement:
                g.update(status="paused", detail="requires AI mining and movement permissions")
                return "wait", None
            try:
                health = float(state["health"])
            except (KeyError, TypeError, ValueError):
                health = math.nan
            hostile = any(e.get("hostile") and not e.get("player") and _near(e.get("dist"), 8)
                          for e in (state.get("entities") or []) if isinstance(e, dict))
            oxygen = state.get("oxygen")
            if not math.isfinite(health) or health <= 8 or hostile or state.get("on_fire") or (
                state.get("in_water") and (not _near(oxygen, 20) or _near(oxygen, 8))
            ):
                g.update(status="paused", detail="waiting for a safer situation")
                return "wait", None
            if g["attempts"] >= (64 if g["item"] == "torch" else 12) or g["no_progress"] >= 4:
                g.update(status="failed", detail="no verified progress; inspect the area and try a new goal")
                return "wait", None
            target = g["item"]
            action = ""
            params: dict[str, Any] = {}
            if target == "torch":
                plan = _torch_next(state, g["count"] - count)
                if isinstance(plan, str):
                    g.update(status="paused", detail=plan)
                    return "wait", None
                action, params, target = plan
            if not action:
                drops = state.get("ground_items") or []
                matching_drop = any(isinstance(d, dict) and short(d.get("item")) == target and _near(d.get("dist"), 8) for d in drops)
                action = "collect" if matching_drop else "mine"
                variants = ALTERNATES.get(target, (GATHERABLE[target],))
                params = {"item": target} if matching_drop else {"block": variants[g["variant"] % len(variants)]}
            self._inflight = True
            g.update(status="active", detail=f"{action} {params.get('item') or params.get('block')} in progress",
                     last_action=action, progress_item=target, progress_before=_inventory_count(state, target),
                     attempts=g["attempts"] + 1)
            return "action", {"token": self._generation, "action": action, "params": params}

    def finish(self, token: int, ok: bool, detail: str, state: dict[str, Any] | None) -> None:
        with self._lock:
            self._inflight = False
            g = self._goal
            if not g or token != self._generation or g["status"] == "cancelled":
                return
            if not ok:
                if g["last_action"] == "mine":
                    g["variant"] += 1
                g["failures"] += 1
                if g["failures"] >= 2:
                    g.update(status="failed", detail=f"two actions failed: {str(detail)[:160]}")
                else:
                    g["detail"] = f"action failed; will reassess: {str(detail)[:120]}"
                return
            g["failures"] = 0
            count = _inventory_count(state, g["item"])
            if count is None or state.get("status", "connected") != "connected":
                g.update(status="paused", detail="action returned; waiting for inventory observation")
                return
            progress = _inventory_count(state, g["progress_item"])
            previous = g["progress_before"]
            if progress is None:
                g.update(status="paused", detail="action returned; waiting for inventory observation")
                return
            g["no_progress"] = 0 if previous is None or progress > previous else g["no_progress"] + 1
            if g["last_action"] == "mine" and previous is not None and progress <= previous:
                g["variant"] += 1
            g["observed"] = count
            if count >= g["count"]:
                g.update(status="completed", detail="target verified in inventory")
            else:
                g["detail"] = "action returned; target not yet verified"


def _torch_next(state: dict[str, Any], missing: int) -> tuple[str, dict[str, Any], str] | str:
    """Plan exactly one torch prerequisite; never invent recipes or place stations."""
    try:
        inv = Inventory.from_snapshot(state)
        batches = math.ceil(missing / 4)
        if inv.count("stick") < batches:
            # A stick recipe needs two planks of the same type.
            plank = next((name for name in sorted(inv.names()) if name.endswith("_planks") and inv.count(name) >= 2), None)
            if plank:
                return "craft", {"item": "stick", "count": 4, "no_station": True}, "stick"
            log = next((name for name in sorted(inv.names()) if name in GATHERABLE and name.endswith("_log")
                        and inv.count(name) > 0), None)
            if log:
                planks = log.removesuffix("_log") + "_planks"
                return "craft", {"item": planks, "count": 4, "no_station": True}, planks
            return "", {}, "oak_log"
        if inv.count("coal") + inv.count("charcoal") < batches:
            drops = state.get("ground_items") or []
            coal_drop = any(isinstance(d, dict) and short(d.get("item")) == "coal" and _near(d.get("dist"), 8) for d in drops)
            if not coal_drop and not any(name.endswith("_pickaxe") for name in inv.names()):
                return "torch plan needs a pickaxe to mine coal or coal/charcoal in inventory"
            return "", {}, "coal"
        return "craft", {"item": "torch", "count": 4, "no_station": True}, "torch"
    except (TypeError, ValueError, OverflowError):
        return "waiting for a valid inventory observation"


def _inventory_count(state: Any, item: str) -> int | None:
    if not isinstance(state, dict) or (not isinstance(state.get("inventory"), list) and not isinstance(state.get("slots"), list)):
        return None
    try:
        return Inventory.from_snapshot(state).count(item)
    except (TypeError, ValueError, OverflowError):
        return None


def _near(value: Any, distance: float) -> bool:
    try:
        value = float(value)
        return math.isfinite(value) and 0 <= value <= distance
    except (TypeError, ValueError):
        return False
