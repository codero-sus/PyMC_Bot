"""Small read-only action-format benchmark. It does not launch or simulate Minecraft."""
from __future__ import annotations

import json
import time
from typing import Any

from pymc_bot.agent import ACTIONS, parse_decision
from pymc_bot.ollama import OllamaClient

# Fixed, synthetic prompts, not player recordings or training data. These are narrow
# action-choice checks, not survival/parkour/tech-tree benchmarks.
CASES = (
    ("hungry", "You have 6 hunger, 20 health and bread in inventory. No danger nearby. Choose one action.", {"eat"}),
    ("nearby_logs", "You need oak logs and are next to an oak tree. No danger nearby. Choose one action.", {"mine", "chop_tree"}),
    ("stop", "You were asked to stop moving immediately. Choose one action.", {"stop", "wait"}),
)
SYSTEM = ("You are controlling a Minecraft bot. Respond with one JSON object containing an action field. "
          "Actions include eat, mine, chop_tree, stop, wait and other PyMC Bot actions. No prose.")


def benchmark(client: OllamaClient, repeats: int = 1, json_mode: bool = False) -> dict[str, Any]:
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    results = []
    for _ in range(repeats):
        for name, prompt, expected in CASES:
            started = time.perf_counter()
            record: dict[str, Any] = {"case": name, "parsed": False, "strict_json": False, "known_action": False,
                                      "expected_action": False, "action": None, "error": None}
            try:
                reply = client.chat([{"role": "user", "content": prompt}], system=SYSTEM, json_mode=json_mode)
                try:
                    record["strict_json"] = isinstance(json.loads(reply), dict)
                except (ValueError, TypeError):
                    pass
                decision = parse_decision(reply)
                record.update(parsed=True, action=decision.action, known_action=decision.action in ACTIONS,
                              expected_action=decision.action in expected)
            except Exception as exc:  # noqa: BLE001 - an individual bad response must not hide the other cases
                record["error"] = str(exc)[:200]
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            results.append(record)
    n = len(results)
    return {"kind": "synthetic_read_only_action_choices", "model": client.model, "repeats": repeats,
            "cases": n, "parsed_rate": sum(r["parsed"] for r in results) / n,
            "strict_json_rate": sum(r["strict_json"] for r in results) / n,
            "known_action_rate": sum(r["known_action"] for r in results) / n,
            "expected_action_rate": sum(r["expected_action"] for r in results) / n,
            "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1), "results": results}
