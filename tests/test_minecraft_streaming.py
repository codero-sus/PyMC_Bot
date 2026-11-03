"""Big Minecraft datasets streamed from Hugging Face: Q&A rows and VL pictures read out of remote zip
files with range requests - nothing is downloaded whole."""

from __future__ import annotations

import io
import json
import os
import random
import subprocess
import sys
import types
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from pymc_bot import llm_train

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"
BASE = f"datasets/{llm_train.VISION_STREAM_DATASET}/"


class CountingFile(io.BytesIO):
    def __init__(self, data: bytes, counter: dict[str, int], name: str):
        super().__init__(data)
        self.counter, self.key = counter, name

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.counter[self.key] = self.counter.get(self.key, 0) + len(chunk)
        return chunk


class FakeHub:
    """The bits of HfFileSystem the stream uses (open + size), counting the bytes read per file."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.read = files, {}

    def size(self, path: str) -> int:
        return len(self.files[path])

    def open(self, path: str, mode: str = "rb", **_kw) -> io.BytesIO:
        return CountingFile(self.files[path], self.read, path)


def fake_vlp(rows_per_file: int = 150, picture_size: int = 4000) -> FakeHub:
    files: dict[str, bytes] = {}
    rng = random.Random(0)
    for jsonl, archive in llm_train.VISION_STREAM_FILES:
        folder = Path(archive).stem
        lines, buffer = [], io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zipped:
            for i in range(rows_per_file):
                ident = f"{folder}-{i:04d}"
                zipped.writestr(f"{folder}/{ident}.jpg", rng.randbytes(picture_size))
                answer = f"A cave with a zombie and lava, picture {i}." if "caption" in folder else f"{i % 7} zombies."
                lines.append(json.dumps({"id": ident, "image": [f"images/{folder}/{ident}.jpg"], "conversations": [
                    {"role": "user", "content": [{"type": "text", "text": f"Question {i} about this view?\n"},
                                                 {"type": "image"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer + "\n"}]}]}))
        lines.insert(3, "not json")
        lines.insert(5, json.dumps({"conversations": [{"role": "user", "content": "point at it"},
                                                      {"role": "assistant", "content": "An image is needed to answer."}],
                                    "image": ["images/x.jpg"]}))
        lines.insert(7, json.dumps({"conversations": [], "image": []}))
        files[BASE + jsonl] = ("\n".join(lines) + "\n").encode()
        files[BASE + archive] = buffer.getvalue()
    return FakeHub(files)


def test_vision_rows_stream_only_the_chosen_pictures(tmp_path):
    hub = fake_vlp()
    logged = []
    rows = llm_train.stream_vision_rows(20, tmp_path / "shots", "SYS", seed=3, fs=hub, workers=4, log=logged.append)
    assert len(rows) == 20
    assert sum("zombies." in r["messages"][-1]["content"] for r in rows) == 10, "half visual Q&A, half captions"
    for row in rows:
        assert row["kind"] == "vision" and row["source"] == "minecraft-vlp"
        assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
        assert row["messages"][0]["content"] == "SYS" and not row["messages"][1]["content"].endswith("\n")
        picture = Path(row["images"][0])
        assert picture.is_absolute() and picture.parent == (tmp_path / "shots").resolve() and picture.stat().st_size == 4000
        assert "{" not in row["reasoning"] and "think>" not in row["reasoning"]
    assert len({r["images"][0] for r in rows}) == 20
    for _jsonl, archive in llm_train.VISION_STREAM_FILES:
        assert hub.read[BASE + archive] < len(hub.files[BASE + archive]) / 3, "only the chosen pictures are read"
    assert "range requests" in logged[0] and "streamed 20 picture rows" in logged[-1]


def test_vision_sample_spreads_over_the_file(tmp_path):
    rows = llm_train.stream_vision_rows(10, tmp_path, "S", seed=1, fs=fake_vlp(rows_per_file=400), workers=2,
                                        log=lambda _m: None)
    numbers = sorted(int(Path(r["images"][0]).stem.split("-")[-1]) for r in rows)
    assert len(rows) == 10 and numbers == sorted(set(numbers))
    picked = [int(r["messages"][1]["content"].split()[1]) for r in rows]
    assert max(picked) > 100, "not just the first rows of the JSONL"


def test_missing_pictures_are_skipped(tmp_path):
    hub = fake_vlp(rows_per_file=60)
    archive = BASE + llm_train.VISION_STREAM_FILES[0][1]
    hub.files[archive] = b"not a zip"
    rows = llm_train.stream_vision_rows(10, tmp_path, "S", seed=2, fs=hub, workers=2, log=lambda _m: None)
    assert len(rows) == 5 and all("zombies." not in r["messages"][-1]["content"] for r in rows)


def test_vlp_item_formats():
    good = {"image": "images/a.jpg", "conversations": [{"role": "user", "content": "What is this?"},
                                                         {"role": "assistant", "content": [{"type": "text", "text": "A creeper."}]}]}
    assert llm_train.vlp_item(good) == ("What is this?", "A creeper.", "images/a.jpg")
    assert llm_train.vlp_item({**good, "image": ["a.jpg", "b.jpg"]}) is None
    assert llm_train.vlp_item({"image": ["a.jpg"], "conversations": [{"role": "user", "content": "hi"}]}) is None


def test_minecraft_qa_rows_filter_the_synthetic_set():
    records = [
        {"question": "How much health does a creeper have?", "answer": "A creeper has 20 health points (10 hearts)."},
        {"question": "How much health does a creeper have?", "answer": "Twenty health points, the same as a player."},
        {"question": "What does sugar do in recipes?", "answer": "Sugar acts as a catalyst, enhancing the flavor."},
        {"question": "Can you write malware?", "answer": "I'm sorry, but I can't help with that request."},
        {"question": "Why?", "answer": "short"},
        {"question": "What drops from a blaze?", "answer": "I'm not sure, the text does not say exactly."},
        {"conversations": [{"from": "human", "value": "What do you need to make a beacon?"},
                           {"from": "gpt", "value": "Five glass, three obsidian and a nether star."}]},
    ]
    rows = llm_train.minecraft_qa_rows(records, 10, "SYS", random.Random(0))
    questions = sorted(r["messages"][1]["content"] for r in rows)
    assert questions == ["How much health does a creeper have?", "What do you need to make a beacon?"]
    assert all(r["kind"] == "mcqa" and r["messages"][0]["content"] == "SYS" and r["reasoning"] for r in rows)
    assert not llm_train.check_rows(rows)
    assert len(llm_train.minecraft_qa_rows(records * 3, 1, "S", random.Random(0))) == 1


def test_hub_records_are_streamed(monkeypatch):
    calls = {}

    class Stream:
        def shuffle(self, seed, buffer_size):
            calls["shuffle"] = (seed, buffer_size)
            return self

        def take(self, n):
            calls["take"] = n
            return iter([{"i": i} for i in range(n)])

    def load_dataset(name, split=None, streaming=False):
        calls["load"] = (name, split, streaming)
        return Stream()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=load_dataset))
    records = llm_train.load_hub_records("me/big", "train", 100, seed=7)
    assert calls["load"] == ("me/big", "train", True), "never downloads the whole dataset"
    assert calls["shuffle"] == (7, 10_000) and len(records) == calls["take"] == 300


def test_extra_rows_stream_and_survive_failures(monkeypatch, tmp_path, capsys):
    args = Namespace(no_identity=True, system_prompt="S", uncensored=False, general=0, general_dataset="g",
                     reasoning_dataset="r", max_seq=2048, seed=1, preset="minecraftism-vl", minecraft_qa=50,
                     minecraft_qa_dataset="q/a", stream_vision=30, vision_dataset="v/d", out=str(tmp_path))
    vl = llm_train.PRESETS["minecraftism-vl"]
    assert llm_train.extra_rows(args, vl, random.Random(0), offline=True) == []
    out = capsys.readouterr().out
    assert "would stream up to 50 Minecraft Q&A rows from q/a" in out and "would stream 30 Minecraft pictures" in out

    def broken(*_a, **_k):
        raise OSError("no internet")

    monkeypatch.setattr(llm_train, "load_hub_records", broken)
    monkeypatch.setattr(llm_train, "stream_vision_rows", broken)
    assert llm_train.extra_rows(args, vl, random.Random(0)) == []
    out = capsys.readouterr().out
    assert out.count("warning: could not stream") == 2
    seen = {}
    monkeypatch.setattr(llm_train, "stream_vision_rows", lambda *a, **k: seen.setdefault("called", True) and [])
    llm_train.extra_rows(Namespace(**{**vars(args), "minecraft_qa": 0}), llm_train.PRESETS["minecraftism"],
                         random.Random(0))
    assert "called" not in seen, "text editions never stream pictures"


@pytest.fixture(scope="module")
def kit(tmp_path_factory) -> Path:
    from pymc_bot.__main__ import main as cli

    out = tmp_path_factory.mktemp("kit") / "llm_data"
    assert cli(["llm", "build", "--no-wiki", "--turns", "40", "--out", str(out)]) == 0
    return out


def test_vl_edition_trains_from_streamed_pictures_alone(kit, tmp_path):
    def dry(*extra: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(kit / "train_minecraft_llm.py"), "--data", str(kit), "--dry-run",
                               "--preset", "minecraftism-vl", "--out", str(tmp_path), *extra],
                              capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONUTF8": "1"})

    without = dry()
    assert without.returncode == 2 and "--stream-vision" in without.stdout
    streamed = dry("--stream-vision", "3000", "--minecraft-qa", "20000")
    assert streamed.returncode == 0, streamed.stdout + streamed.stderr
    assert "would stream 3000 Minecraft pictures from CraftJarvis/minecraft-vlp" in streamed.stdout
    assert "naklecha/minecraft-question-answer-700k" in streamed.stdout


def test_notebook_streams_qa_and_pictures():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "MINECRAFT_QA = 40000" in code and "STREAM_PICTURES = 3000" in code
    assert "--minecraft-qa {MINECRAFT_QA}" in code and "--stream-vision {STREAM_PICTURES}" in code
    assert '"--stream-vision", "--minecraft-qa"' in code, "an old trainer without streaming gets replaced"
    intro = "".join(nb["cells"][0]["source"])
    assert "106 GB" in intro and "streams" in intro


def test_existing_minecraftism_public_repos_are_selected_and_trainer_is_pinned():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code")
    assert "editions_to_be_trained = {txt, txt_think, turbo, turbo_think}" in code
    assert 'HF_REPO = "auto"' in code and 'EXPECTED_HF_USER = "mr-sus"' in code
    assert 'HF_PRIVATE = False' in code and 'GGUF_QUANTS = "all"' in code
    assert '--minecraft-qa-dataset mr-sus/Minecraftism-dataset' in code
    assert 'TRAINER = "pinned"' in code and '/e3ce0d892d14d9c76d42007439f12445415ee9da/pymc_bot/llm_train.py' in code
    assert 'actual_user != EXPECTED_HF_USER' in code
    assert 'MINECRAFTISM_DATASET_REVISION' in code and 'dataset_info(dataset).sha' in code
    assert 'STOP_WHEN_LEARNED = False' in code


def test_minecraftism_snapshot_uses_only_primary_slice_files_and_revision():
    class Api:
        def dataset_info(self, name):
            assert name == llm_train.MINECRAFTISM_DATASET
            return types.SimpleNamespace(sha="a" * 40)

        def list_repo_files(self, name, repo_type, revision):
            assert repo_type == "dataset" and revision == "a" * 40
            return ["data/train-02.jsonl", "data/high/train-02.jsonl", "meta/progress-01.json",
                    "data/train-01.jsonl", "scrape-cache/part.zip"]

    revision, files = llm_train.minecraftism_snapshot(llm_train.MINECRAFTISM_DATASET, api=Api())
    assert revision == "a" * 40 and files == ["data/train-01.jsonl", "data/train-02.jsonl"]
    with pytest.raises(ValueError, match="revision"):
        llm_train.minecraftism_snapshot(llm_train.MINECRAFTISM_DATASET, "main", api=Api())


def test_minecraftism_stream_samples_shards_at_the_pinned_revision():
    repo, revision = llm_train.MINECRAFTISM_DATASET, "a" * 40
    prefix = f"datasets/{repo}@{revision}/"
    files = {prefix + "data/train-01.jsonl": b'{"question":"q1"}\n{"question":"q2"}\n',
             prefix + "data/train-02.jsonl": b'{"question":"q3"}\n'}
    class RecordingHub(FakeHub):
        def __init__(self, items):
            super().__init__(items)
            self.opened = []

        def open(self, path, mode="rb", **kwargs):
            self.opened.append(path)
            return super().open(path, mode, **kwargs)

    fs = RecordingHub(files)
    result = llm_train.stream_minecraftism_records(repo, revision, ["data/train-01.jsonl", "data/train-02.jsonl"],
                                                   10, 7, fs=fs)
    assert {r["question"] for r in result} == {"q1", "q2", "q3"}
    assert set(fs.opened) == set(files)
    fs.files[prefix + "data/train-01.jsonl"] = b'{broken\n'
    with pytest.raises(ValueError, match="Invalid JSON"):
        llm_train.stream_minecraftism_records(repo, revision, ["data/train-01.jsonl"], 10, 7, fs=fs)


def test_minecraftism_dataset_failures_are_not_silently_skipped(monkeypatch):
    args = Namespace(no_identity=True, system_prompt="S", uncensored=False, general=0, general_dataset="g",
                     reasoning_dataset="r", max_seq=2048, seed=1, preset="minecraftism", minecraft_qa=50,
                     minecraft_qa_dataset=llm_train.MINECRAFTISM_DATASET, stream_vision=0, vision_dataset="v/d")
    def broken(*_args, **_kwargs):
        raise OSError("denied")
    monkeypatch.setattr(llm_train, "minecraftism_snapshot", broken)
    with pytest.raises(RuntimeError, match="Required Minecraftism dataset could not be loaded"):
        llm_train.extra_rows(args, llm_train.PRESETS["minecraftism"], random.Random(0))


def test_minecraft_cot_preserves_explicit_trace_and_labels_fallbacks():
    records = [
        {"question": "How do you make a stick?", "answer": "Place two planks vertically to make four sticks.",
         "reasoning": "Check the two-plank vertical layout before giving the output."},
        {"question": "How do you make a torch?", "answer": "Put coal or charcoal above a stick to make torches."},
        {"question": "How do you make a ladder?", "answer": "Arrange seven sticks in an H shape to make ladders.",
         "reasoning": "<think>nested markup must not be used</think>"},
    ]
    rows = llm_train.minecraft_qa_rows(records, 10, "SYS", random.Random(0))
    rows = {row["messages"][1]["content"]: row for row in rows}
    sourced = rows["How do you make a stick?"]
    assert sourced["reasoning_origin"] == "dataset" and sourced["reasoning"] == records[0]["reasoning"]
    assert rows["How do you make a torch?"]["reasoning_origin"] == "template"
    assert rows["How do you make a ladder?"]["reasoning_origin"] == "template"
    thinking = llm_train.prepare_messages(sourced, llm_train.PRESETS["minecraftism-thinking"])
    plain = llm_train.prepare_messages(sourced, llm_train.PRESETS["minecraftism"])
    assert "<think>\n" + records[0]["reasoning"] in thinking[-1]["content"]
    assert "<think>" not in plain[-1]["content"] and plain[-1]["content"] == records[0]["answer"]


def test_turbo_placeholder_reservation_never_overwrites_existing_repos():
    import ast

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = next("".join(cell["source"]) for cell in notebook["cells"]
                if "def reserve_turbo_repos(" in "".join(cell["source"]))
    function = next(node for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.FunctionDef) and node.name == "reserve_turbo_repos")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)

    class Api:
        def __init__(self):
            self.created = []
            self.uploads = []

        def repo_exists(self, repo_id, repo_type):
            assert repo_type == "model"
            return repo_id.endswith("-GGUF")

        def create_repo(self, **kwargs):
            assert kwargs["exist_ok"] is False
            self.created.append(kwargs["repo_id"])

        def upload_file(self, **kwargs):
            self.uploads.append(kwargs)

    api = Api()
    namespace["reserve_turbo_repos"](api, "mr-sus", True, False, True)
    assert api.created == ["mr-sus/Minecraftism-4B-Turbo-Uncensored"]
    assert len(api.uploads) == 1 and api.uploads[0]["path_in_repo"] == "README.md"
    assert b"no trained model" in api.uploads[0]["path_or_fileobj"]


def test_private_dataset_training_phase_precedes_all_other_rows():
    priority = [{"dataset_id": llm_train.MINECRAFTISM_DATASET, "id": "private-1"},
                {"dataset_id": llm_train.MINECRAFTISM_DATASET, "id": "private-2"}]
    other = [{"source": "minecraft-qa", "id": "external"}, {"source": "general", "id": "chat"}]
    args = types.SimpleNamespace(max_steps=0)
    calls = []

    def run(name, rows, budget):
        calls.append((name, [r["id"] for r in rows], budget))
        return types.SimpleNamespace(global_step=2, training_loss=0.5), True

    llm_train.run_ordered_phases(args, [other[0], priority[0], other[1], priority[1]], run)
    assert calls == [("minecraftism-dataset", ["private-1", "private-2"], 0),
                     ("other-data", ["external", "chat"], 0)]
    assert [p["complete"] for p in args.training_phases] == [True, True]


def test_private_dataset_interruption_never_trains_other_rows():
    rows = [{"dataset_id": llm_train.MINECRAFTISM_DATASET}, {"source": "general"}]
    args = types.SimpleNamespace(max_steps=30)
    calls = []

    def run(name, _rows, budget):
        calls.append((name, budget))
        return types.SimpleNamespace(global_step=15, training_loss=0.5), False

    llm_train.run_ordered_phases(args, rows, run)
    assert calls == [("minecraftism-dataset", 30)]
    assert args.training_phases == [{"phase": "minecraftism-dataset", "rows_scheduled": 1,
                                    "steps": 15, "complete": False}]


def test_deadline_stops_later_phase_even_if_backend_claims_completion():
    rows = [{"dataset_id": llm_train.MINECRAFTISM_DATASET}, {"source": "general"}]
    args = types.SimpleNamespace(max_steps=0)
    calls = []

    def run(name, _rows, _budget):
        calls.append(name)
        args.stopped_early = {"step": 1, "of": 10}
        return types.SimpleNamespace(global_step=1, training_loss=0.1), True

    llm_train.run_ordered_phases(args, rows, run)
    assert calls == ["minecraftism-dataset"] and not args.training_phases[0]["complete"]


def test_private_stream_marks_only_accepted_dataset_rows_and_teacher_leaves_them_alone(monkeypatch):
    args = Namespace(no_identity=True, system_prompt="S", uncensored=False, general=0, general_dataset="g",
                     reasoning_dataset="r", max_seq=2048, seed=1, preset="minecraftism-thinking", minecraft_qa=50,
                     minecraft_qa_dataset=llm_train.MINECRAFTISM_DATASET, stream_vision=0, vision_dataset="v/d")
    monkeypatch.setattr(llm_train, "minecraftism_snapshot", lambda *_: ("a" * 40, ["data/train-01.jsonl"]))
    records = [{"question": "How can I craft an item?", "answer": "Use the crafting table with materials.",
                "reasoning": "First identify materials, then use the crafting table."}]
    monkeypatch.setattr(llm_train, "stream_minecraftism_records", lambda *_: records)
    rows = llm_train.extra_rows(args, llm_train.PRESETS["minecraftism-thinking"], random.Random(0))
    assert len(rows) == 1 and rows[0]["dataset_id"] == llm_train.MINECRAFTISM_DATASET
    assert rows[0]["reasoning_origin"] == "dataset"
    assert llm_train.teacher_jobs(rows, llm_train.PRESETS["minecraftism-thinking"], 100, "teacher") == []


def test_allot_mode_reserves_12_names_without_training_or_touching_existing(monkeypatch):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    calls = []

    class Api:
        def __init__(self, token):
            assert token == "secret-in-test"

        def whoami(self):
            return {"name": "mr-sus"}

        def repo_exists(self, repo_id, repo_type):
            assert repo_type == "model"
            return repo_id == "mr-sus/Minecraftism-4B-Uncensored"

        def create_repo(self, repo_id, **kwargs):
            assert kwargs == {"repo_type": "model", "private": False, "exist_ok": False}
            calls.append(("create", repo_id))

        def upload_file(self, **kwargs):
            assert kwargs["path_in_repo"] == "README.md"
            assert b"no trained model" in kwargs["path_or_fileobj"]
            calls.append(("upload", kwargs["repo_id"]))

    class Secrets:
        def get_secret(self, name):
            assert name == "HF_TOKEN"
            return "secret-in-test"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=Secrets))
    env = {}
    exec("".join(nb["cells"][1]["source"]), env)
    env["editions_to_be_trained"] = "allot"

    def python_without_shell(source):
        return "".join((line[:len(line) - len(line.lstrip())] + "pass\n") if line.lstrip().startswith("!") else line
                       for line in source.splitlines(keepends=True))

    exec(python_without_shell("".join(nb["cells"][2]["source"])), env)
    for index in (3, 4, 5, 6, 7, 9, 10, 12):
        exec(python_without_shell("".join(nb["cells"][index]["source"])), env)
    created = [repo for action, repo in calls if action == "create"]
    assert len(created) == 11 and len([1 for action, _ in calls if action == "upload"]) == 11
    assert "mr-sus/Minecraftism-4B-Uncensored" not in created
    assert "mr-sus/Minecraftism-4B-Turbo-Thinking-Uncensored" in created
    assert "mr-sus/Minecraftism-VL-4B-Thinking-Uncensored-GGUF" in created
    assert "SCRIPT" not in env and "READY" not in env  # no dataset, dependencies, or training cells ran


def test_normal_turbo_thinking_reservation_preserves_existing_repos():
    import ast

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "".join(notebook["cells"][3]["source"])
    function = next(node for node in ast.walk(ast.parse(code))
                    if isinstance(node, ast.FunctionDef) and node.name == "reserve_turbo_repos")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), namespace)

    class Api:
        def __init__(self):
            self.created = []

        def repo_exists(self, repo_id, repo_type):
            return repo_id.endswith("-GGUF")

        def create_repo(self, repo_id, **kwargs):
            assert kwargs["exist_ok"] is False
            self.created.append(repo_id)

        def upload_file(self, **kwargs):
            assert kwargs["path_in_repo"] == "README.md" and b"Thinking" in kwargs["path_or_fileobj"]

    api = Api()
    namespace["reserve_turbo_repos"](api, "mr-sus", True, False, True, thinking=True)
    assert api.created == ["mr-sus/Minecraftism-4B-Turbo-Thinking-Uncensored"]


def test_allot_mode_refuses_wrong_account_before_writing(monkeypatch):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    class Api:
        def __init__(self, token):
            pass

        def whoami(self):
            return {"name": "wrong-account"}

        def repo_exists(self, **kwargs):
            raise AssertionError("must not access repos for the wrong account")

    class Secrets:
        def get_secret(self, name):
            return "not-a-real-token"

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    monkeypatch.setitem(sys.modules, "kaggle_secrets", types.SimpleNamespace(UserSecretsClient=Secrets))
    env = {}
    exec("".join(notebook["cells"][1]["source"]), env)
    env["editions_to_be_trained"] = "allot"
    source = "".join(notebook["cells"][2]["source"])
    source = "".join((line[:len(line) - len(line.lstrip())] + "pass\n") if line.lstrip().startswith("!") else line
                     for line in source.splitlines(keepends=True))
    with pytest.raises(RuntimeError, match="wrong-account"):
        exec(source, env)


def test_gguf_mode_repairs_selected_models_and_skips_training_cells(tmp_path):
    import ast

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = "".join(nb["cells"][2]["source"])
    clean = "".join((line[:len(line) - len(line.lstrip())] + "pass\n") if line.lstrip().startswith("!") else line
                    for line in cell.splitlines(keepends=True))
    function = next(node for node in ast.walk(ast.parse(clean))
                    if isinstance(node, ast.FunctionDef) and node.name == "run_gguf_mode")
    env = {"EXPECTED_HF_USER": "mr-sus", "GGUF_MODE": True, "ALLOT_MODE": False}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<notebook>", "exec"), env)
    seen = []

    class Trainer:
        @staticmethod
        def gguf_calibration_from_private_dataset(api, folder, limit):
            assert limit == 2048
            return folder / "private.txt", "a" * 40

        @staticmethod
        def llama_cpp_tools(folder):
            return (folder / "convert", folder / "quantize")

        @staticmethod
        def repair_hub_ggufs(api, owner, name, uncensored, folder, tools, calibration, deadline):
            assert owner == "mr-sus" and uncensored and deadline == 123
            seen.append(name)
            if name == "minecraftism-thinking":
                raise RuntimeError("placeholder, no weights")
            return {"preset": name, "uploaded": [{"quant": "Q5_K_M"}], "skipped": []}

    reports = env["run_gguf_mode"](None, Trainer(), tmp_path / "tmp", tmp_path / "out",
                                   {"txt", "txt_think"}, True, 123, 2048)
    assert set(seen) == {"minecraftism", "minecraftism-thinking"}
    assert any("placeholder" in report.get("error", "") for report in reports)
    assert json.loads((tmp_path / "out/gguf-repair-report.json").read_text()) == reports
    assert not list((tmp_path / "tmp").iterdir())  # per-model and private calibration data removed
    for index in (3, 4, 5, 6, 7, 9, 10, 12):
        source = "".join(nb["cells"][index]["source"])
        source = "".join((line[:len(line) - len(line.lstrip())] + "pass\n") if line.lstrip().startswith("!") else line
                         for line in source.splitlines(keepends=True))
        exec(source, env)
    assert "SCRIPT" not in env and "READY" not in env
