"""Bounded synthetic end-to-end goal checks using the local simulated backend.

Not a Minecraft world, server benchmark, model quality measurement or speedrun.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from pymc_bot.agent import AgentLoop
from pymc_bot.backends import SimulatedBackend
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog
from pymc_bot.inventory import Inventory

CASES = (
    ("empty_to_stone_pickaxe", "stone_pickaxe", False),
    ("empty_to_raw_iron", "raw_iron", False),
    ("low_health_pause_then_resume", "wooden_pickaxe", True),
)


def run_sim_bench(repeats: int = 1, max_steps: int = 48) -> dict[str, Any]:
    if not 1 <= repeats <= 10 or not 1 <= max_steps <= 96:
        raise ValueError("repeats or max_steps out of bounds")
    records = []
    with tempfile.TemporaryDirectory(prefix="pymc-sim-bench-") as tmp:
        for repeat in range(repeats):
            for name, goal_item, hazard in CASES:
                started = time.perf_counter()
                record: dict[str, Any] = {"case": name, "repeat": repeat + 1, "target": goal_item,
                                          "completed": False, "paused_safely": None, "steps": 0, "error": None}
                log = EventLog()
                store = ConfigStore(Path(tmp) / f"{repeat}-{name}.json")
                store.load()
                store.update({"minecraft": {"backend": "simulated", "username": "BenchBot"},
                              "agent": {"allow_building": True, "persistent_memory": False, "spatial_memory": False},
                              "search": {"enabled": False}})
                backend = SimulatedBackend(log, ambient_chat=False)
                backend._inv = Inventory()  # noqa: SLF001 - fixed benchmark starting conditions
                backend._ground.clear()  # noqa: SLF001
                bot = MinecraftBot(store, log, auto_reconnect=False, backend=backend)
                try:
                    bot.start()
                    agent = AgentLoop(bot, store, log, prefer_ollama=False)
                    agent.start_gather_goal(goal_item, 1)
                    if hazard:
                        backend._health = 4  # noqa: SLF001
                        agent.run_once()
                        record["paused_safely"] = agent.goal.snapshot()["status"] == "paused"
                        backend._health = 20  # noqa: SLF001
                    for step in range(max_steps):
                        ok, detail = agent.run_once()
                        record["steps"] = step + 1
                        task = agent.goal.snapshot()
                        if task and task["status"] == "completed":
                            record["completed"] = bot.inventory().count(goal_item) >= 1
                            break
                        if not ok or (task and task["status"] in {"failed", "cancelled"}):
                            record["error"] = str(detail)[:200]
                            break
                    else:
                        record["error"] = "step limit reached"
                    if hazard and not record["paused_safely"]:
                        record["completed"] = False
                        record["error"] = "did not pause at low health"
                except Exception as exc:  # noqa: BLE001 - report case failures, never leave a bot running
                    record["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                finally:
                    bot.stop()
                record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
                records.append(record)
    return {"kind": "simulated_goal_integration", "cases": len(records), "repeats": repeats,
            "pass_rate": sum(row["completed"] for row in records) / len(records),
            "mean_elapsed_ms": round(sum(row["elapsed_ms"] for row in records) / len(records), 1),
            "note": "Simulated backend only: no server, real terrain, pathfinding, latency, LLM or game physics.",
            "results": records}
