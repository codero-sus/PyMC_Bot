"""Training speed: batches sized by tokens (same effective batch), out-of-memory retries and streamed data
shared by every edition of a session."""

from __future__ import annotations

import json
import types

import pytest

from pymc_bot import llm_train

TEXT = {"batch": 2, "accum": 4}


def test_no_budget_keeps_the_preset_batch():
    assert llm_train.plan_batches([500] * 10, TEXT, 1, 3072, 0)["batch"] == 2
    assert llm_train.plan_batches([500] * 10, TEXT, 2, 3072, 0)["accum"] == 2


@pytest.mark.parametrize(("lengths", "world", "budget", "batch", "accum"), [
    ([600] * 100, 1, 12288, 8, 1),  # short rows: the whole effective batch in one pass
    ([600] * 100, 2, 12288, 4, 1),  # two T4s: half each
    ([3000] * 100, 1, 12288, 4, 2),  # long rows: smaller batches, more accumulation
    ([9000] * 100, 1, 12288, 4, 2),  # rows are cut at --max-seq (3072)
    ([3000] * 100, 1, 4096, 2, 4),  # never below the preset's batch
])
def test_token_budget_keeps_the_effective_batch(lengths, world, budget, batch, accum):
    plan = llm_train.plan_batches(lengths, TEXT, world, 3072, budget)
    assert (plan["batch"], plan["accum"]) == (batch, accum)
    assert plan["batch"] * plan["accum"] * world == 8  # what the model learns per step does not change


def test_p95_ignores_a_few_long_rows():
    plan = llm_train.plan_batches([500] * 97 + [3072] * 3, TEXT, 1, 3072, 12288)
    assert plan["p95"] == 500 and plan["batch"] == 8


def test_token_lengths_uses_the_tokenizer_or_estimates():
    def tokenizer(texts, add_special_tokens=False):
        return {"input_ids": [t.split() for t in texts]}

    assert llm_train.token_lengths(tokenizer, ["a b c", "d"], chunk=1) == [3, 1]
    assert llm_train.token_lengths(None, ["x" * 35]) == [10]


def test_speed_options_follow_the_plan():
    args = llm_train.build_parser().parse_args(["--group-by-length", "--token-budget", "12288"])
    plan = {"batch": 8, "accum": 1, "p95": 600, "mean": 400}
    options = llm_train.speed_options(args, TEXT, 1, True, plan=plan)
    assert options["gradient_accumulation_steps"] == 1 and "group_by_length" not in options
    assert llm_train.speed_options(args, TEXT, 1, True)["group_by_length"] is True


def test_out_of_memory_before_the_first_step_halves_the_batch():
    class OutOfMemoryError(RuntimeError):
        pass

    class Trainer:
        state = types.SimpleNamespace(global_step=0)

        def __init__(self, batch):
            self.batch = batch

        def train(self):
            if self.batch > 2:
                raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            return f"trained with {self.batch}"

    sizes = [8]

    def rebuild(exc):
        assert llm_train.is_oom(exc)
        sizes.append(sizes[-1] // 2)
        return Trainer(sizes[-1])

    assert llm_train.run_training(Trainer(8), rebuild, world=1) == "trained with 2" and sizes == [8, 4, 2]
    assert not llm_train.is_oom(ValueError("bad data"))


def test_cache_streams_once_per_session(tmp_path):
    args = types.SimpleNamespace(stream_cache=str(tmp_path / "cache"))
    calls = []

    def load():
        calls.append(1)
        return [{"q": "hi"}]

    assert llm_train.cached(args, "k", load) == [{"q": "hi"}]
    assert llm_train.cached(args, "k", load) == [{"q": "hi"}] and len(calls) == 1
    assert llm_train.cached(args, "other", load) and len(calls) == 2
    assert llm_train.cached(args, "k", load, check=lambda _v: False) and len(calls) == 3  # e.g. pictures deleted
    assert llm_train.cached(types.SimpleNamespace(stream_cache=None), "k", load) and len(calls) == 4
    assert llm_train.cached(args, "odd", lambda: {1, 2}) == {1, 2}  # not JSON: used, just not shared


def test_two_editions_share_the_streamed_chat(tmp_path, monkeypatch):
    calls = []

    def records(name, split, limit, seed, buffer=10_000, factor=3):
        calls.append(name)
        return [{"messages": [{"role": "user", "content": f"question {i}?"},
                              {"role": "assistant", "content": f"A helpful answer number {i}."}]} for i in range(30)]

    monkeypatch.setattr(llm_train, "load_hub_records", records)
    argv = ["--general", "10", "--no-identity", "--stream-cache", str(tmp_path / "cache"), "--commercial"]
    import random

    rows = []
    for preset in ("minecraftism", "minecraftism-vl"):
        args = llm_train.build_parser().parse_args(argv + ["--preset", preset, "--out", str(tmp_path / preset)])
        args.system_prompt = "S"
        llm_train.resolve_datasets(args)
        rows.append(llm_train.extra_rows(args, llm_train.PRESETS[preset], random.Random(1)))
    assert len(calls) == 1 and rows[0] == rows[1] and rows[0]
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
    assert json.loads(next((tmp_path / "cache").glob("*.json")).read_text())[0]["messages"]


def test_pictures_are_reused_only_while_they_exist(tmp_path):
    picture = tmp_path / "a.jpg"
    picture.write_bytes(b"x")
    assert llm_train._pictures_exist([{"images": [str(picture)]}, {"messages": []}])
    picture.unlink()
    assert not llm_train._pictures_exist([{"images": [str(picture)]}])


# --------------------------------------------------------------------------- the notebook
def _notebook():
    from pathlib import Path

    nb = json.loads((Path(__file__).resolve().parents[1] / "pynb/PyMC_Minecraft_LLM_Kaggle.ipynb").read_text("utf-8"))
    return ["".join(cell["source"]) for cell in nb["cells"]]


def _pick(expression):
    cells = _notebook()
    env: dict = {}
    exec(cells[1], env)  # the settings cell defines txt, txt_think, vl, vl_think
    env["editions_to_be_trained"] = eval(expression, env)
    check = cells[3]
    exec("import re\n" + check[check.index("# ---- the editions picked"):check.index("def check(")], env)
    return env["EDITIONS"]


@pytest.mark.parametrize(("expression", "editions"), [
    ("{txt}", ["minecraftism"]),
    ("{txt, txt_think, vl, vl_think}", ["minecraftism-vl-thinking", "minecraftism-vl", "minecraftism-thinking",
                                        "minecraftism"]),  # longest first: both T4s finish at about the same time
    ('{"vl", "txt"}', ["minecraftism-vl", "minecraftism"]),
    ('"all"', ["minecraftism-vl-thinking", "minecraftism-vl", "minecraftism-thinking", "minecraftism"]),
    ('"txt_think, vl"', ["minecraftism-vl", "minecraftism-thinking"]),
    ('["minecraftism-vl", "qwen3-8b", "TXT"]', ["minecraftism-vl", "minecraftism", "qwen3-8b"]),
])
def test_editions_to_be_trained_picks_only_those(expression, editions, capsys):
    assert _pick(expression) == editions
    assert "editions to train (longest first)" in capsys.readouterr().out


def test_unknown_or_no_edition_is_refused_before_any_gpu_time():
    with pytest.raises(ValueError, match="txt, txt_think, vl, vl_think"):
        _pick('{"vl_thinking"}')
    with pytest.raises(AssertionError):
        _pick("set()")


def test_notebook_speed_flags_and_deferred_ggufs():
    cells = _notebook()
    code = "\n".join(cells)
    assert "editions_to_be_trained = {txt}" in cells[1] and "\nEDITIONS = [" not in cells[1]
    assert 'os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")' in code
    assert 'FLAGS += f" --stream-cache {CACHE}"' in code and 'CACHE = f"{WORK}/cache"' in code
    assert 'RUN_FLAGS += " --defer-gguf"' in code and '"--export-only"' in code
    assert '"CUDA_VISIBLE_DEVICES": ""' in code  # the GGUF process never touches a T4
    assert '"--token-budget 0"' in code  # out of memory with the big batches -> the preset's batch
    assert "subprocess.Popen" in cells[4]  # the dry runs run side by side


def test_thinking_edition_fills_up_with_everyday_chat_when_few_traces_fit(tmp_path, monkeypatch, capsys):
    """Nemotron's math traces are mostly far longer than --max-seq (a Kaggle run kept 16 of 12000): the thinking
    edition then gets everyday chat for the rest instead of almost no everyday rows."""
    calls = []

    def records(name, split, limit, seed, buffer=10_000, factor=3):
        calls.append(name)
        if "Nemotron" in name:
            short = [{"messages": [{"role": "user", "content": f"2+{i}?"},
                                   {"role": "assistant", "content": f"<think>add</think>{2 + i}"}]} for i in range(2)]
            return short + [{"messages": [{"role": "user", "content": "prove it"},
                                          {"role": "assistant", "content": "<think>" + "x" * 50_000 + "</think>done"}]}] * 20
        return [{"messages": [{"role": "user", "content": f"question {i}?"},
                              {"role": "assistant", "content": f"A helpful answer number {i}."}]} for i in range(30)]

    monkeypatch.setattr(llm_train, "load_hub_records", records)
    argv = ["--general", "10", "--no-identity", "--stream-cache", str(tmp_path / "cache"), "--commercial"]
    import random

    out = {}
    for preset in ("minecraftism-thinking", "minecraftism"):
        args = llm_train.build_parser().parse_args(argv + ["--preset", preset, "--out", str(tmp_path / preset)])
        args.system_prompt = "S"
        llm_train.resolve_datasets(args)
        out[preset] = llm_train.extra_rows(args, llm_train.PRESETS[preset], random.Random(1))
    thinking = out["minecraftism-thinking"]
    assert sum(row["source"] == "reasoning" for row in thinking) == 2
    assert sum(row["source"] == "general" for row in thinking) == 8 and len(thinking) == 10
    assert "thinking traces: 2 of 22 streamed fit --max-seq" in capsys.readouterr().out
    assert len(calls) == 2  # the text edition reused the chat the thinking edition streamed
    args = llm_train.build_parser().parse_args(argv + ["--preset", "minecraftism-thinking"])
    llm_train.resolve_datasets(args)
    assert llm_train.used_datasets(args, llm_train.PRESETS["minecraftism-thinking"]) == [
        llm_train.COMMERCIAL_REASONING_DATASET, llm_train.COMMERCIAL_GENERAL_DATASET]


def test_qa_rows_with_one_system_prompt_train_as_chats_and_every_answer_stays():
    import random as _random

    from pymc_bot import llm_train as lt

    system = lt.MINECRAFTISM_SYSTEM
    qa = [{"messages": [{"role": "system", "content": system}, {"role": "user", "content": f"Q{i}?"},
                        {"role": "assistant", "content": f"Answer {i % 40}."}], "kind": "mcqa", "source": "minecraft-data"}
          for i in range(80)]  # 40 facts, 2 phrasings each
    other = [{"messages": [{"role": "system", "content": "bot"}, {"role": "user", "content": "state"},
                           {"role": "assistant", "content": "{}"}], "kind": "turn"},
             {"messages": [{"role": "system", "content": "other prompt"}, {"role": "user", "content": "Q?"},
                           {"role": "assistant", "content": "A."}], "kind": "qa"}]
    merged = lt.merge_qa(qa + other, 6, _random.Random(1))
    chats = [r for r in merged if r.get("kind") == "qa-chat"]
    assert other[0] in merged and len(merged) < len(qa) // 4  # game turns are never merged
    answers = [m["content"] for r in merged if r.get("merged") for m in r["messages"] if m["role"] == "assistant"]
    assert sorted(answers) == sorted(r["messages"][2]["content"] for r in qa + other[1:])  # every answer kept
    for chat in chats:
        roles = [m["role"] for m in chat["messages"]]
        assert roles == ["system"] + ["user", "assistant"] * chat["merged"] and chat["merged"] <= 6
        said = [m["content"] for m in chat["messages"] if m["role"] == "assistant"]
        assert len(said) == len(set(said))  # two phrasings of one fact never share a chat
    assert lt.merge_qa(qa, 1) == qa
    before = sum(len(m["content"]) for r in qa for m in r["messages"])
    after = sum(len(m["content"]) for r in chats for m in r["messages"])
    assert after < before / 3  # the system prompt was ~80% of every Q&A row
    args = lt.build_parser().parse_args(["--merge-qa", "6"])
    assert args.merge_qa == 6
