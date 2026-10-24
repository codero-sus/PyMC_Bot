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
