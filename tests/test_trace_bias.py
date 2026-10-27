"""Biases a trained Minecraftism Thinking model showed, traced back to its training data.

The model wrote the same scaffold for every question ("The question: ... What I know: ... Key point: ...
Answer directly and briefly."), filled it with made-up facts (a stat card for a piston, a wrong key point), said
"Simple one, answer directly." before code and redstone questions, and told people to use Cherry Planks for a
piston. These tests keep each of those out of the data."""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymc_bot import llm_reasoning as lr  # noqa: E402
from pymc_bot import llm_train  # noqa: E402

QUESTIONS = {
    "What is the exact crafting grid layout to make a Piston in vanilla Minecraft?": "recipe",
    "What ingredients are needed to brew a Potion of Night Vision (3:00)?": "brewing",
    "How does a Redstone Comparator work in Subtract Mode versus Compare Mode?": "compare",
    "Explain how an Observer block detects updates and how many redstone ticks its output pulse lasts.": "explain",
    "Can an Iron Golem spawn naturally in a village with zero villagers present?": "yesno",
    "How far can a redstone signal travel through dust before requiring a repeater or comparator?": "number",
    "If I build an Overworld portal at X: 800, Y: 64, Z: -400, what are the matching nether portal coordinates?": "number",
    "Write a Python function using urllib.request to send a GET request to an API endpoint and parse the JSON payload.": "code",
    "Write a basic bash script to check if a specific process is running and restart it if it is down.": "code",
    "How do I get to the End?": "steps",
}
#: Fixed lines the old traces repeated on thousands of rows.
OLD_LINES = ("Simple one", "Key point", "Answer directly and briefly", "Keep it short and correct", "no fluff",
             "straight away", "I know the exact", "I recall what the wiki says", "Basic Minecraft knowledge",
             "What I know:")


def test_questions_get_the_plan_their_answer_needs():
    for question, kind in QUESTIONS.items():
        assert llm_train.answer_kind(question) == kind, question
    assert llm_train.answer_kind("how do i do this", "```python\nprint(1)\n```") == "code"


def test_rows_without_a_trace_think_about_their_own_question():
    row = {"messages": [{"role": "system", "content": "s"},
                        {"role": "user", "content": "Write a bash script that restarts nginx when it is down."},
                        {"role": "assistant", "content": "```bash\nsystemctl restart nginx\n```"}]}
    text = llm_train.prepare_messages(row, llm_train.PRESETS["minecraftism-thinking"])[-1]["content"]
    trace = text.split("</think>")[0]
    assert "restarts nginx" in trace and "code" in trace.lower()
    assert not any(line.lower() in trace.lower() for line in OLD_LINES)
    assert text.endswith("```bash\nsystemctl restart nginx\n```")
    assert llm_train.think_text("", "hi").startswith("<think>\n") and "Simple one" not in llm_train.think_text("", "hi")


def test_plans_vary_and_never_copy_or_invent_facts():
    plans = [llm_train.answer_plan(q) for q in QUESTIONS]
    assert len({" ".join(p.split()[:4]) for p in plans}) >= 6, "not one scaffold for everything"
    for plan in plans:
        assert "{" not in plan and "}" not in plan and "think>" not in plan
        assert not any(line.lower() in plan.lower() for line in OLD_LINES)
    long_answer = "## Compare mode\nOutput = rear.\n## Subtract mode\nOutput = rear - side.\n" + "x" * 400
    plan = llm_train.answer_plan("How does a comparator work in subtract mode versus compare mode?", long_answer)
    assert "Points to cover: Compare mode; Subtract mode." in plan
    assert "rear - side" not in plan


def test_qa_traces_have_no_scaffold_no_copied_answer_and_mob_notes_only_for_mobs():
    answer = "Use 3 planks, 4 cobblestone, 1 iron ingot and 1 redstone dust. The planks go on top."
    trace = lr.qa_reasoning("How do I craft a Piston?", answer, "knowledge", rng=random.Random(1))
    assert not any(line.lower() in trace.lower() for line in OLD_LINES)
    assert "3 planks" not in trace, "the answer is not copied into the trace as a 'key point'"
    assert "My notes on" not in trace and "HP" not in trace, "no invented stat card for a block"
    mob = lr.qa_reasoning("How much health does an Iron Golem have?", "100 HP (50 hearts).", "knowledge")
    assert "My notes on the iron golem: neutral, 100 HP" in mob and "(" not in mob.split("My notes")[1][:30]
    assert "2 HP is one heart" in mob
    explain = lr.qa_reasoning("Explain how an Observer detects block updates.", "It watches its face...", "wiki")
    assert "briefly" not in explain.lower() and "explanation" in explain.lower() or "mechanism" in explain.lower()


def _data(recipes: dict) -> dict:
    items = [{"id": i, "name": n, "displayName": d, "stackSize": 64} for i, n, d in (
        (1, "oak_planks", "Oak Planks"), (2, "spruce_planks", "Spruce Planks"), (3, "cherry_planks", "Cherry Planks"),
        (4, "cobblestone", "Cobblestone"), (5, "iron_ingot", "Iron Ingot"), (6, "redstone", "Redstone Dust"),
        (7, "piston", "Piston"), (8, "stick", "Stick"), (9, "bamboo", "Bamboo"), (10, "coal", "Coal"),
        (11, "charcoal", "Charcoal"), (12, "torch", "Torch"))]
    return {"version": "1.21.8", "items": items, "recipes": recipes}


def _piston(planks: int) -> dict:
    return {"inShape": [[planks] * 3, [4, 5, 4], [4, 6, 4]], "result": {"id": 7, "count": 1}}


def test_recipes_say_any_planks_instead_of_one_random_wood():
    data = _data({
        "7": [_piston(3), _piston(1), _piston(2)],
        "8": [{"inShape": [[9], [9]], "result": {"id": 8, "count": 1}},  # 2 bamboo -> 1 stick: another recipe
              {"inShape": [[3], [3]], "result": {"id": 8, "count": 4}}, {"inShape": [[1], [1]], "result": {"id": 8, "count": 4}}],
        "12": [{"inShape": [[10], [8]], "result": {"id": 12, "count": 4}}, {"inShape": [[11], [8]], "result": {"id": 12, "count": 4}}],
    })
    facts = {q[0]: a for q, a, _t in llm_train.minecraft_data_facts(data)}
    piston = facts["How do I craft Piston?"]
    assert "Cherry" not in piston and "3 any Planks, 4 Cobblestone, 1 Iron Ingot and 1 Redstone Dust" in piston
    assert "row 1: any Planks, any Planks, any Planks / row 2: Cobblestone, Iron Ingot, Cobblestone" in piston
    assert "don't have to match" in piston and "recipe variants" not in piston
    stick = facts["How do I craft Stick?"]
    assert "2 any Planks" in stick and "You get 4 Stick" in stick and "Bamboo" not in stick
    assert "1 other recipe" in stick
    assert "Charcoal or Coal" in facts["How do I craft Torch?"]


def test_fact_rows_use_more_phrasings_and_plan_instead_of_claiming_certainty():
    data = _data({"7": [_piston(1), _piston(2)]})
    rows = [r for r in llm_train.minecraft_data_rows(data, 100, "SYS", random.Random(0)) if r["messages"][-1]["content"].startswith("Shaped")]
    questions = {row["messages"][1]["content"] for row in rows}
    assert len(questions) == 3 and len(rows) == 3
    for row in rows:
        assert not any(line.lower() in row["reasoning"].lower() for line in OLD_LINES)
        assert "crafting" in row["reasoning"].lower() or "recipe" in row["reasoning"].lower()
    phrasings = set()
    for seed in range(12):
        phrasings |= {r["messages"][1]["content"] for r in llm_train.minecraft_data_rows(data, 100, "S", random.Random(seed))
                      if r["messages"][-1]["content"].startswith("Shaped")}
    assert "What's the crafting grid layout for Piston in vanilla Minecraft?" in phrasings and len(phrasings) == 5


def test_streamed_minecraft_qa_traces_plan_too():
    records = [{"question": "How many hearts does a wither have?", "answer": "A wither has 300 HP, so 150 hearts."}]
    rows = llm_train.minecraft_qa_rows(records, 5, "SYS", random.Random(0))
    assert rows and not any(line.lower() in rows[0]["reasoning"].lower() for line in OLD_LINES)
    assert "value" in rows[0]["reasoning"] or "number" in rows[0]["reasoning"] or "figure" in rows[0]["reasoning"]
