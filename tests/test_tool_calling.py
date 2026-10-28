"""Tool-calling (agentic) rows, the agentic score, and the de-biased bot data."""

from __future__ import annotations

import contextlib
import json
import random
from collections import Counter
from pathlib import Path

from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

from pymc_bot import llm_dataset as ds
from pymc_bot import llm_reasoning, llm_train
from pymc_bot.mobs import load_mobs

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen3_4b_instruct_2507_chat_template.jinja"

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                     "required": ["city"]}}},
    {"type": "function", "function": {"name": "convert_currency", "description": "Converts money, e.g. 10 USD to €.",
                                      "parameters": {"type": "object", "properties": {"amount": {"type": "number"},
                                                                                     "to": {"type": "string"}}}}},
]
CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Pune\"}"}}
CALL2 = {"id": "c2", "type": "function", "function": {"name": "convert_currency",
                                                      "arguments": "{\"amount\": 10, \"to\": \"INR\"}"}}


def record(reasoning: str = "off") -> dict:
    """A chat shaped like nvidia/Nemotron-Post-Training-Dataset-v1:tool_calling."""
    think = (lambda text, trace: f"<think>\n{trace}\n</think>\n\n{text}") if reasoning == "on" else (lambda text, _t: text)
    return {"reasoning": reasoning, "metadata": json.dumps({"tools": TOOLS}), "messages": [
        {"role": "user", "content": "Weather in Pune, and what is 10 USD in rupees?", "tool_calls": []},
        {"role": "assistant", "content": think("", "Two tools: weather and currency."), "tool_calls": [CALL, CALL2]},
        {"role": "tool", "content": "{\"temp_c\": 31}", "tool_calls": []},
        {"role": "tool", "content": "{\"result\": 832.5}", "tool_calls": []},
        {"role": "assistant", "content": think("It's 31 °C in Pune, and 10 USD is about ₹832.50.", "Both came back."),
         "tool_calls": []},
    ]}


def qwen_render(messages: list[dict], tools: list | None = None) -> str:
    """Qwen3-4B-Instruct-2507's real chat template, run the way transformers runs it."""
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
    env.filters["tojson"] = lambda value, indent=None: json.dumps(value, ensure_ascii=False, indent=indent)
    return env.from_string(TEMPLATE.read_text(encoding="utf-8")).render(messages=messages, tools=tools,
                                                                        add_generation_prompt=False)


def test_tool_rows_render_exactly_like_qwens_own_template_does_with_tools():
    rec = record()
    for system in ("You are Minecraftism.", ""):
        native = ([{"role": "system", "content": system}] if system else []) + [
            {k: v for k, v in m.items() if k != "tool_calls" or v} for m in rec["messages"]]
        flat = llm_train.tool_messages(rec["messages"], TOOLS, system, thinking=False)
        assert qwen_render(flat) == qwen_render(native, TOOLS)  # character for character
        assert all(set(m) == {"role", "content"} for m in flat)  # plain turns: any chat template renders them
    tool_turn = flat[2]["content"]
    assert flat[2]["role"] == "assistant" and tool_turn.count("<tool_call>") == 2 and "get_weather" in tool_turn
    assert flat[3] == {"role": "user", "content": "<tool_response>\n{\"temp_c\": 31}\n</tool_response>\n"
                                                  "<tool_response>\n{\"result\": 832.5}\n</tool_response>"}


def test_thinking_editions_keep_a_trace_in_every_assistant_turn_the_others_drop_it():
    flat = llm_train.tool_messages(record("on")["messages"], TOOLS, "", thinking=True)
    answers = [m["content"] for m in flat if m["role"] == "assistant"]
    assert all(a.startswith("<think>\n") and "\n</think>\n\n" in a for a in answers)
    assert "Two tools" in answers[0] and answers[0].endswith("</tool_call>")
    plain = llm_train.tool_messages(record("on")["messages"], TOOLS, "", thinking=False)
    assert not any("<think>" in m["content"] or "Two tools" in m["content"] for m in plain)
    assert llm_train.tool_messages(record("off")["messages"], TOOLS, "", thinking=True) is None  # no trace to train
    preset = llm_train.PRESETS["minecraftism-thinking"]
    row = {"messages": flat, "kind": "tools"}
    assert llm_train.prepare_messages(row, preset) == flat  # not wrapped in a second <think>


def test_tool_rows_hold_out_chats_for_the_score_and_teach_when_no_tool_fits():
    records = [record() for _ in range(80)] + [record("on") for _ in range(20)]
    chats = [{"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": f"Write a poem number {i}"},
                           {"role": "assistant", "content": "Roses..."}], "source": "general"} for i in range(40)]
    rows, held = llm_train.tool_rows(records, 40, "You are Minecraftism.", False, random.Random(1), chats)
    calls = [r for r in rows if r["kind"] == "tools"]
    none = [r for r in rows if r["kind"] == "tools-none"]
    assert len(calls) == 40 and len(none) == 6  # ~15% more where no tool fits
    assert all("<tool_call>" not in r["messages"][-1]["content"] and "# Tools" in r["messages"][0]["content"]
               for r in none)
    systems = Counter(r["messages"][0]["content"].startswith("You are Minecraftism.") for r in calls)
    assert systems[True] and systems[False]  # with our system prompt and with only the tool prompt
    wanted = [h for h in held if h["tool_expect"]]
    assert len(wanted) == llm_train.TOOL_EVAL and wanted[0]["tool_expect"] == [
        {"name": "get_weather", "arguments": {"city": "Pune"}}, {"name": "convert_currency",
                                                                  "arguments": {"amount": 10, "to": "INR"}}]
    assert wanted[0]["messages"][-1]["role"] == "assistant"  # the model is asked for that turn
    assert 0 < len(held) - len(wanted) <= len(llm_train.NO_TOOL_PROMPTS)
    # a tool that might fit the question is never offered as "no tool fits"
    assert llm_train.no_tool_messages("What's the weather today?", "Sunny.", TOOLS, "") is None


def test_the_agentic_score_compares_the_fine_tuned_model_with_its_base():
    want = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "-"}],
            "tool_expect": [{"name": "get_weather", "arguments": {"city": "Pune"}}]}
    none = {"messages": [{"role": "user", "content": "poem"}, {"role": "assistant", "content": "-"}], "tool_expect": []}
    good = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Pune"}}\n</tool_call>'
    assert llm_train.tool_score(want, "<think>hm</think>\n" + good) == {"right_tool": True, "exact": True,
                                                                        "needs_call": True}
    assert llm_train.tool_score(want, good.replace("Pune", "Delhi"))["right_tool"]
    assert not llm_train.tool_score(want, good.replace("Pune", "Delhi"))["exact"]
    assert llm_train.tool_score(none, "Roses are red")["exact"] and not llm_train.tool_score(none, good)["exact"]

    class Model:
        base = False

        @contextlib.contextmanager
        def disable_adapter(self):
            self.base = True
            yield
            self.base = False

    model = Model()

    def generate(row):  # the base model calls a tool for everything; the fine-tune only when one fits
        return good if (model.base or row["tool_expect"]) else "Roses are red"

    report = llm_train.agentic_eval(generate, [want, none, want, none], model, log=lambda _m: None)
    assert report["tuned"]["score"] == 100.0 and report["base"]["score"] == 50.0
    assert report["base"]["no_call_when_none_fits"] == 0.0 and report["tuned"]["exact_calls"] == 100.0
    card = llm_train.model_card("minecraftism", llm_train.PRESETS["minecraftism"],
                                {"quant": "Q4_K_M", "agentic": report, "base_model": "Qwen/Qwen3-4B-Instruct-2507"}, "me/x")
    assert "## Agentic score (tool calling)" in card and "| Qwen3-4B-Instruct-2507 | 50.0% |" in card


def test_the_trainer_takes_tool_rows_from_a_commercial_dataset():
    args = llm_train.build_parser().parse_args(["--tools", "2000", "--commercial"])
    assert args.tool_dataset == llm_train.TOOL_DATASET
    llm_train.resolve_datasets(args)
    assert args.tool_dataset in llm_train.used_datasets(args, llm_train.PRESETS["minecraftism"])
    assert llm_train.commercial_problems(args, llm_train.PRESETS["minecraftism"])[0] == []  # CC BY 4.0


def test_no_bot_action_takes_more_than_15_percent_of_the_game_turns():
    rows = ds.rule_examples(1000, random.Random(5))
    actions = Counter(json.loads(r.messages[-1]["content"]).get("action") for r in rows)
    assert len(rows) == 1000 and max(actions.values()) <= 150 and len(actions) >= 10


def test_a_thinking_trace_starts_with_what_matters_and_always_keeps_the_decision():
    db = load_mobs()
    state = {"health": 20, "food": 20, "dimension": "overworld", "time_of_day": "night", "position": {"y": 64},
             "entities": [{"type": "creeper", "hostile": True, "distance": 3}], "players": []}
    trace = llm_reasoning.turn_reasoning(state, {"action": "flee", "thought": "creeper"}, db=db, rng=random.Random(1))
    assert trace.startswith("A creeper is 3 blocks away") and not trace.startswith("Health")
    calm = dict(state, entities=[])
    openers = {llm_reasoning.turn_reasoning(calm, {"action": "wander"}, db=db, rng=random.Random(i)).split(".")[0]
               for i in range(12)}
    assert len(openers) >= 3  # no fixed scaffold
    chat = [{"username": "Priya", "message": "bot can you follow me? " + "please " * 200}]
    long = llm_reasoning.turn_reasoning(dict(state, health=5), {"action": "flee"}, history=chat, db=db,
                                        rng=random.Random(2))
    assert len(long) <= llm_reasoning.MAX_TRACE and long.startswith("Health 5/20")
    end = llm_reasoning.action_line({"action": "flee"}, dict(state, health=5),
                                    [e for e in state["entities"]], random.Random(2))
    assert end.split(".")[0] in long  # the decision is never the part that gets cut
