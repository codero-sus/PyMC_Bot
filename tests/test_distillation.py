"""Qwen3-style strong-to-weak distillation in the Kaggle trainer (no GPU: the teacher is faked)."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pymc_bot import llm_train

NOTEBOOK = Path(__file__).resolve().parents[1] / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"
SYSTEM = "You are Minecraftism."


def _rows():
    chat = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"chat {i}?"},
                          {"role": "assistant", "content": f"human answer {i}"}], "source": "general", "kind": "general"}
            for i in range(4)]
    facts = [{"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"How many {i}?"},
                           {"role": "assistant", "content": f"A stack holds {60 + i} items."}],
              "source": "minecraft-data", "kind": "mcqa"} for i in range(3)]
    game = {"messages": [{"role": "system", "content": "bot rules"}, {"role": "user", "content": "creeper near"},
                         {"role": "assistant", "content": '{"action": "flee"}'}], "source": "rules", "kind": "game"}
    trace = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "2+2"},
                          {"role": "assistant", "content": "4"}], "reasoning": "add", "source": "reasoning",
             "kind": "general"}
    picture = {"messages": [{"role": "user", "content": "what biome?"}, {"role": "assistant", "content": "desert"}],
               "images": ["a.png"], "source": "minecraft-raytrace", "kind": "general"}
    return chat + facts + [game, trace, picture]


def _args(tmp_path, **extra):
    parser = llm_train.build_parser()
    args = parser.parse_args(["--preset", "minecraftism", "--out", str(tmp_path / "out")])
    args.stream_cache = str(tmp_path / "cache")
    args.distill_rows = 50
    for key, value in extra.items():
        setattr(args, key, value)
    return args


def _good_answers(jobs):
    """What a well-behaved teacher writes (keeps the reference's numbers)."""
    out = {}
    for job in jobs:
        text = f"teacher says {job['reference'] or job['messages'][-1]['content']}"
        out[job["key"]] = ("<think>\nlet me think {carefully}\n</think>\n\n" + text) if job["thinking"] else text
    return out


def test_parser_defaults_keep_distillation_off():
    args = llm_train.build_parser().parse_args(["--preset", "minecraftism"])
    assert (args.distill_rows, args.onpolicy_steps, args.teacher) == (0, 0, "auto")
    assert args.distill_minutes == 40 and args.onpolicy_batch == 4 and args.onpolicy_tokens == 256


def test_teacher_is_a_bigger_qwen3_with_the_same_tokenizer(tmp_path):
    assert llm_train.teacher_model(_args(tmp_path)) == "unsloth/Qwen3-8B"
    assert llm_train.teacher_model(_args(tmp_path, uncensored=True)) == "huihui-ai/Huihui-Qwen3-8B-abliterated-v2"
    assert llm_train.teacher_model(_args(tmp_path, teacher="unsloth/Qwen3-14B")) == "unsloth/Qwen3-14B"
    assert llm_train.TEACHER_LICENSE[1] is True  # Apache-2.0: a commercial-only run stays commercial


def test_teacher_jobs_pick_chat_and_facts_never_game_turns_traces_or_pictures():
    rows = _rows()
    jobs = llm_train.teacher_jobs(rows, llm_train.PRESETS["minecraftism"], 100, "t")
    kinds = [rows[job["row"]]["kind"] for job in jobs]
    assert kinds[:6] == ["general", "mcqa"] * 3 and sorted(kinds) == ["general"] * 4 + ["mcqa"] * 3
    assert all(rows[job["row"]]["source"] not in ("rules", "reasoning", "minecraft-raytrace") for job in jobs)
    fact = next(job for job in jobs if job["reference"])
    assert fact["reference"] in fact["messages"][0]["content"] and "Never mention the reference" in fact["messages"][0]["content"]
    assert fact["messages"][-1]["role"] == "user"  # the teacher writes the reply itself
    assert rows[fact["row"]]["messages"][0]["content"] == SYSTEM  # the training row keeps its own prompt
    assert len(llm_train.teacher_jobs(rows, llm_train.PRESETS["minecraftism"], 2, "t")) == 2
    thinking = llm_train.teacher_jobs(rows, llm_train.PRESETS["minecraftism-thinking"], 100, "t")
    assert all(job["thinking"] for job in thinking) and thinking[0]["key"] != jobs[0]["key"]
    assert llm_train.teacher_jobs(rows, llm_train.PRESETS["minecraftism"], 100, "other")[0]["key"] != jobs[0]["key"]


def test_teacher_replies_are_parsed_and_checked():
    split = llm_train.split_teacher_reply
    assert split("<think>\nhmm {x}\n</think>\n\nThe answer.", True) == ("hmm (x)", "The answer.")
    assert split("<think>still thinking when the tokens ran out", True) is None
    assert split("<think>\n\n</think>\n\nPlain answer.", False) == ("", "Plain answer.")
    assert split("   ", False) is None
    accept = llm_train.accept_teacher_answer
    fact = {"reference": "A stack holds 64 items and costs 3."}
    assert accept(fact, "", "Stacks go up to 64, it costs 3.", False) == ""
    assert accept(fact, "", "Stacks go up to 16.", False) == "changed a fact"
    assert accept(fact, "the reference says 64 and 3", "64 and 3", False) == "mentioned the reference"
    assert accept({"reference": ""}, "", "I can't help with that.", True) == "refusal"
    assert accept({"reference": ""}, "", "I can't help with that.", False) == ""  # censored runs keep it as is
    assert accept({"reference": ""}, "", "", False) == "empty"


def test_apply_teacher_answers_swaps_answers_and_keeps_originals_otherwise():
    rows = _rows()
    before = json.dumps(rows)
    jobs = llm_train.teacher_jobs(rows, llm_train.PRESETS["minecraftism-thinking"], 100, "teach/er")
    answers = _good_answers(jobs)
    answers[jobs[1]["key"]] = "<think>no end"  # cut off -> original kept
    out, stats = llm_train.apply_teacher_answers(rows, jobs, answers, "teach/er", uncensored=True)
    assert json.dumps(rows) == before  # never edits the caller's rows
    assert stats["asked"] == 7 and stats["used"] == 6 and stats["dropped"] == {"cut off": 1}
    first = out[jobs[0]["row"]]
    assert first["messages"][-1]["content"].startswith("teacher says") and first["reasoning"] == "let me think (carefully)"
    assert first["teacher"] == "teach/er" and "{" not in first["reasoning"]
    assert out[jobs[1]["row"]] is rows[jobs[1]["row"]]
    assert out[7] is rows[7]  # the game turn


def test_offpolicy_distillation_caches_the_answers_for_the_next_edition(tmp_path, capsys):
    rows, calls = _rows(), []

    def generate(teacher, jobs, max_seq, seconds):
        calls.append(len(jobs))
        return _good_answers(jobs)

    args = _args(tmp_path)
    out = llm_train.distill_offpolicy(args, llm_train.PRESETS["minecraftism"], rows, generate=generate)
    assert calls == [7] and sum(1 for row in out if row.get("teacher")) == 7
    assert args.distill_report["used"] == 7 and args.distill_report["teacher"] == "unsloth/Qwen3-8B"
    cache = list((tmp_path / "cache").glob("teacher-*.json"))
    assert len(cache) == 1 and len(json.loads(cache[0].read_text())) == 7
    again = llm_train.distill_offpolicy(_args(tmp_path), llm_train.PRESETS["minecraftism"], rows, generate=generate)
    assert calls == [7] and again == out  # the next edition reuses every answer
    assert "7 cached, 0 to write" in capsys.readouterr().out


def test_offpolicy_distillation_never_costs_the_training(tmp_path, capsys):
    rows = _rows()

    def boom(*_a):
        raise RuntimeError("teacher crashed")

    assert llm_train.distill_offpolicy(_args(tmp_path), llm_train.PRESETS["minecraftism"], rows, generate=boom) == rows
    assert "training on the original answers" in capsys.readouterr().out
    called = []
    dry = _args(tmp_path, dry_run=True)
    assert llm_train.distill_offpolicy(dry, llm_train.PRESETS["minecraftism"], rows,
                                       generate=lambda *a: called.append(1)) is rows
    late = _args(tmp_path, deadline=time.time() + 30 * 60, reserve_minutes=20)  # session nearly over
    assert llm_train.distill_offpolicy(late, llm_train.PRESETS["minecraftism"], rows,
                                       generate=lambda *a: called.append(1)) == rows
    assert called == [] and "not enough session time left for the teacher" in capsys.readouterr().out
    assert llm_train.distill_offpolicy(_args(tmp_path, distill_rows=0), llm_train.PRESETS["minecraftism"], rows) is rows


def test_both_t4s_write_half_of_the_teacher_answers(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMC_RUN_ID", "distill-test")
    real_sleep = time.sleep
    monkeypatch.setattr(llm_train.time, "sleep", lambda _s: real_sleep(0.01))
    shares = {}

    def fake_generate(teacher, jobs, max_seq, seconds, batch=0, log=print):
        shares[threading.current_thread().name] = [job["row"] for job in jobs]
        return _good_answers(jobs)

    monkeypatch.setattr(llm_train, "teacher_generate", fake_generate)
    args = _args(tmp_path)
    out = Path(args.out)
    out.mkdir(parents=True)
    got = {}
    other = threading.Thread(name="rank1", target=lambda: got.update(shared=llm_train.wait_for_rows(
        out, timeout=30, poll=0.01, on_job=lambda: llm_train.serve_distill_job(out, 1, 2))))
    other.start()
    rows = llm_train.distill_offpolicy(args, llm_train.PRESETS["minecraftism"], _rows(), world=2)
    llm_train.share_rows(out, rows, [], "/data")
    other.join(10)
    jobs = llm_train.teacher_jobs(_rows(), llm_train.PRESETS["minecraftism"], 50, "unsloth/Qwen3-8B")
    assert shares["MainThread"] == [job["row"] for job in jobs[0::2]]
    assert shares["rank1"] == [job["row"] for job in jobs[1::2]]
    assert sum(1 for row in rows if row.get("teacher")) == 7 and got["shared"]["rows"] == rows


def test_handover_files_from_an_earlier_launch_are_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("PYMC_RUN_ID", "none")
    monkeypatch.setenv("MASTER_PORT", "29501")
    assert llm_train._rows_file(tmp_path).name == ".ddp-rows-29501.json"  # accelerate's run id "none" is skipped
    monkeypatch.setenv("PYMC_RUN_ID", "vl-123")
    stale = llm_train._rows_file(tmp_path)
    stale.write_text(json.dumps({"rows": [], "eval_rows": [], "data_dir": ""}))
    old = llm_train._STARTED - 3600
    import os
    os.utime(stale, (old, old))
    assert not llm_train._fresh(stale)
    with pytest.raises(TimeoutError):
        llm_train.wait_for_rows(tmp_path, timeout=0.05, poll=0.01)
    assert llm_train._distill_file(tmp_path, 1).name == ".distill-vl-123-1.json"


def test_card_and_sources_credit_the_teacher(tmp_path):
    rows = [dict(row, teacher="unsloth/Qwen3-8B") if row["kind"] == "general" else row for row in _rows()]
    args = _args(tmp_path)
    sources = llm_train.data_sources(rows, args)
    teacher = next(s for s in sources if "teacher answers" in s["name"])
    assert teacher["license"] == "Apache-2.0 (Qwen3)" and teacher["commercial"] and teacher["rows"] == 6
    report = {"train_rows": 9, "data": sources, "commercial": True,
              "distill": {"teacher": "unsloth/Qwen3-8B", "offpolicy": {"used": 6, "general": 6, "mcqa": 0},
                          "onpolicy": {"steps": 60, "replies": 480, "kl_start": 0.41, "kl_end": 0.12}}}
    card = llm_train.model_card("minecraftism", llm_train.PRESETS["minecraftism"], report, "me/Minecraftism-4B-GGUF")
    assert "## Distillation" in card and "strong-to-weak" in card
    assert "[unsloth/Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B)" in card and "0.41 -> 0.12" in card
    plain = llm_train.model_card("minecraftism", llm_train.PRESETS["minecraftism"], {"train_rows": 9}, "me/x")
    assert "## Distillation" not in plain


def test_onpolicy_prompts_and_skips(tmp_path):
    prompts = llm_train.onpolicy_prompts(_rows(), steps=5, batch=4, world=2, seed=1)
    assert len(prompts) == 40 and all(p[-1]["role"] == "user" for p in prompts)
    assert not any(p[-1]["content"] == "creeper near" for p in prompts)  # game turns stay rule-checked
    assert llm_train.onpolicy_prompts([], 5, 4, 2, 1) == []
    args = _args(tmp_path, onpolicy_steps=10)
    # vision editions and steps=0 return before anything heavy is imported
    assert llm_train.onpolicy_distill(args, llm_train.PRESETS["minecraftism-vl"], None, None, _rows()) == {}
    assert llm_train.onpolicy_distill(_args(tmp_path), llm_train.PRESETS["minecraftism"], None, None, _rows()) == {}


def test_train_runs_onpolicy_before_the_gpus_part():
    source = Path(llm_train.__file__).read_text(encoding="utf-8")
    body = source[source.index("def train(args"):source.index("def text_generator")]
    assert body.index("onpolicy_distill(") < body.index("end_ddp()")
    main = source[source.index("def prepare_and_train"):]
    assert main.index("distill_offpolicy(args, preset, rows, dist_info()[1])\n    if share") > main.index("return stop(3)")
    assert "serve_distill_job(Path(args.out), rank, world)" in source


def test_notebook_turns_distillation_on():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for line in ("DISTILL = True", 'TEACHER = "auto"', "DISTILL_ROWS = 1500", "DISTILL_MINUTES = 40", "ONPOLICY_STEPS = 60"):
        assert line in code
    assert "--distill-rows {min(DISTILL_ROWS, 32) if QUICK_TEST else DISTILL_ROWS}" in code
    assert "--onpolicy-steps {min(ONPOLICY_STEPS, 3) if QUICK_TEST else ONPOLICY_STEPS}" in code
    intro = "".join(nb["cells"][0]["source"])
    assert "strong-to-weak" in intro and 'GPU_MODE = "merge"' in intro


# ---------------------------------------------------------------- the maths (needs torch: CI's training job)
def test_reverse_kl_matches_the_formula_and_trains_the_student():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    teacher = torch.randn(7, 11)
    student = torch.randn(7, 11, requires_grad=True)
    assert float(llm_train.reverse_kl(teacher.clone(), teacher)) == pytest.approx(0.0, abs=1e-6)
    p = torch.softmax(student.detach(), -1)
    manual = (p * (torch.log_softmax(student.detach(), -1) - torch.log_softmax(teacher, -1))).sum(-1).mean()
    loss = llm_train.reverse_kl(student, teacher, chunk=3)
    assert float(loss) == pytest.approx(float(manual), rel=1e-5) and float(loss) > 0
    loss.backward()
    with torch.no_grad():
        moved = student - 0.5 * student.grad
    assert float(llm_train.reverse_kl(moved, teacher)) < float(loss)  # a step downhill gets closer to the teacher
    padded = torch.cat([teacher, torch.full((7, 5), -1e4)], dim=-1)  # a padded vocabulary is cut to the shared part
    assert float(llm_train.reverse_kl(teacher.clone(), padded)) == pytest.approx(0.0, abs=1e-5)


def test_token_logits_takes_the_positions_that_predict_the_reply():
    torch = pytest.importorskip("torch")

    class Echo(torch.nn.Module):  # logits at position i = one-hot of the token at i (shows which positions are taken)
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask):
            logits = torch.nn.functional.one_hot(input_ids, 20).float() + self.weight
            return type("Out", (), {"logits": logits * attention_mask[..., None]})()

    rows = llm_train.token_logits(Echo(), [[1, 2, 3, 4], [5, 6, 7]], starts=[2, 1], lengths=[2, 2], pad_id=0)
    assert rows.shape == (4, 20)
    assert rows.argmax(-1).tolist() == [2, 3, 5, 6]  # replies 3,4 and 6,7: read at the positions just before them
