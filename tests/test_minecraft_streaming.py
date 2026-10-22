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
    assert "MINECRAFT_QA = 20000" in code and "STREAM_PICTURES = 3000" in code
    assert "--minecraft-qa {MINECRAFT_QA}" in code and "--stream-vision {STREAM_PICTURES}" in code
    assert '"--stream-vision", "--minecraft-qa"' in code, "an old trainer without streaming gets replaced"
    intro = "".join(nb["cells"][0]["source"])
    assert "106 GB" in intro and "streams" in intro
