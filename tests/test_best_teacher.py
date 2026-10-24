"""The best teacher for each distillation phase on Kaggle's 2x T4."""

from __future__ import annotations

import json
import sys
import types

from pymc_bot import llm_train
from tests.test_distillation import _args, _fake_vllm, _good_answers, _rows

EIGHT_B, UNCENSORED = llm_train.TEACHER_MODEL, llm_train.UNCENSORED_TEACHER


def _python(tmp_path):
    python = tmp_path / "python"
    python.write_text("")
    return str(python)


def test_the_14b_writes_and_the_8b_grades(tmp_path):
    args = _args(tmp_path, vllm_python=_python(tmp_path))
    assert llm_train.WRITER_MODEL == "Qwen/Qwen3-14B-AWQ"
    assert llm_train.teacher_model(args, "answers") == "Qwen/Qwen3-14B-AWQ"
    assert llm_train.teacher_model(args, "grade") == llm_train.teacher_model(args) == EIGHT_B  # fits beside the student
    no_vllm = _args(tmp_path)  # transformers would be too slow with a 14B: the 8B writes
    assert llm_train.teacher_model(no_vllm, "answers") == EIGHT_B
    hf_only = _args(tmp_path, vllm_python=_python(tmp_path), teacher_engine="hf")
    assert llm_train.teacher_model(hf_only, "answers") == EIGHT_B


def test_uncensored_keeps_the_abliterated_8b_for_both_phases(tmp_path):
    args = _args(tmp_path, vllm_python=_python(tmp_path), uncensored=True)
    assert llm_train.teacher_model(args, "answers") == llm_train.teacher_model(args, "grade") == UNCENSORED


def test_an_explicit_teacher_is_used_for_both_phases(tmp_path):
    args = _args(tmp_path, vllm_python=_python(tmp_path), teacher="unsloth/Qwen3-14B")
    assert llm_train.teacher_model(args, "answers") == llm_train.teacher_model(args, "grade") == "unsloth/Qwen3-14B"
    assert llm_train.smaller_writer(args, "unsloth/Qwen3-14B") is None  # the user's choice is never swapped


def test_prequantized_checkpoints_are_not_quantized_again():
    assert llm_train.prequantized("Qwen/Qwen3-14B-AWQ") and llm_train.prequantized("me/Qwen3-8B-GPTQ-Int4")
    assert not llm_train.prequantized(EIGHT_B) and not llm_train.prequantized("awqward/Qwen3-8B")


def test_one_t4_loads_the_awq_14b_as_it_is(tmp_path, monkeypatch):
    seen = _fake_vllm(monkeypatch, gpus=1)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"teacher": "Qwen/Qwen3-14B-AWQ", "max_seq": 10, "seconds": 600, "eager": False,
                               "jobs": [{"key": "a", "messages": [{"role": "user", "content": "hi"}],
                                         "thinking": False}]}))
    assert llm_train.main(["--vllm-worker", str(job), str(tmp_path / "answers.json")]) == 0
    assert seen["options"]["tensor_parallel_size"] == 1 and "quantization" not in seen["options"]
    job.write_text(job.read_text().replace("Qwen/Qwen3-14B-AWQ", EIGHT_B))  # 16 GB of fp16: 4-bit while loading
    assert llm_train.main(["--vllm-worker", str(job), str(tmp_path / "answers.json")]) == 0
    assert seen["options"]["quantization"] == "bitsandbytes"


def test_the_8b_writes_when_the_14b_does_not_start(tmp_path, monkeypatch):
    used = []

    def fake_vllm(teacher, jobs, max_seq, seconds, python):
        used.append(teacher)
        if teacher == llm_train.WRITER_MODEL:
            raise RuntimeError("vLLM could not run the teacher")
        return _good_answers(jobs)

    monkeypatch.setattr(llm_train, "vllm_generate", fake_vllm)
    monkeypatch.setattr(llm_train, "teacher_generate", lambda *a, **k: (_ for _ in ()).throw(AssertionError("hf")))
    args = _args(tmp_path, vllm_python=_python(tmp_path))
    rows = llm_train.distill_offpolicy(args, llm_train.PRESETS["minecraftism"], _rows())
    assert used == [llm_train.WRITER_MODEL, EIGHT_B]  # the 8B on vLLM, not transformers
    assert args.distill_report["teacher"] == EIGHT_B and args.distill_report["used"] == 7
    assert {row["teacher"] for row in rows if row.get("teacher")} == {EIGHT_B}


def test_the_14b_answers_are_used_and_cached(tmp_path, monkeypatch):
    used = []
    monkeypatch.setattr(llm_train, "vllm_generate",
                        lambda teacher, jobs, *a: used.append(teacher) or _good_answers(jobs))
    args = _args(tmp_path, vllm_python=_python(tmp_path))
    rows = llm_train.distill_offpolicy(args, llm_train.PRESETS["minecraftism"], _rows())
    assert used == ["Qwen/Qwen3-14B-AWQ"] and args.distill_report["teacher"] == "Qwen/Qwen3-14B-AWQ"
    assert {row["teacher"] for row in rows if row.get("teacher")} == {"Qwen/Qwen3-14B-AWQ"}
    table = {d["name"]: d for d in llm_train.data_sources(rows, args)}
    assert table["Qwen/Qwen3-14B-AWQ (teacher answers)"]["commercial"]  # Apache-2.0: commercial runs stay commercial
    again = llm_train.distill_offpolicy(_args(tmp_path, vllm_python=_python(tmp_path)),
                                        llm_train.PRESETS["minecraftism"], _rows())
    assert used == ["Qwen/Qwen3-14B-AWQ"] and sum(1 for row in again if row.get("teacher")) == 7  # from the cache


class _Tok:
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) * self.scale for c in text]}


def test_on_policy_needs_a_teacher_that_cuts_text_the_same_way(tmp_path, monkeypatch):
    assert llm_train.same_tokenizer(_Tok(), _Tok()) and not llm_train.same_tokenizer(_Tok(), _Tok(2))
    assert not llm_train.same_tokenizer(_Tok(), object())
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda name: _Tok(2))))
    monkeypatch.setitem(sys.modules, "unsloth", None)  # never reached: nothing is loaded for a mismatched teacher
    args = _args(tmp_path, onpolicy_steps=4)
    rows = [row for row in _rows() if not row.get("images")]
    assert llm_train.onpolicy_distill(args, llm_train.PRESETS["minecraftism"], object(), _Tok(), rows) == {}


def test_model_card_names_the_writer_and_the_grader():
    report = {"train_rows": 9, "distill": {"teacher": "Qwen/Qwen3-14B-AWQ", "grader": EIGHT_B,
                                           "offpolicy": {"used": 6, "general": 6, "mcqa": 0},
                                           "onpolicy": {"steps": 60, "replies": 480, "kl_start": 0.4, "kl_end": 0.1}}}
    card = llm_train.model_card("minecraftism", llm_train.PRESETS["minecraftism"], report, "me/Minecraftism-4B-GGUF")
    assert "[Qwen/Qwen3-14B-AWQ](https://huggingface.co/Qwen/Qwen3-14B-AWQ) writing the answers" in card
    assert f"[{EIGHT_B}](https://huggingface.co/{EIGHT_B}) grading the student's own replies" in card
