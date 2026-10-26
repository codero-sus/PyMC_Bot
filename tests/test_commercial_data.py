"""--commercial: only training data whose licence allows commercial use (OpenAssistant chat, NVIDIA Nemotron
traces, Q&A generated from minecraft-data, minecraft-raytrace pictures) - all tested offline with fakes."""

from __future__ import annotations

import gzip
import io
import json
import random
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest

from pymc_bot import llm_train

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "pynb" / "PyMC_Minecraft_LLM_Kaggle.ipynb"


def trainer_args(**over) -> Namespace:
    args = llm_train.build_parser().parse_args([])
    for key, value in over.items():
        setattr(args, key, value)
    return args


# ------------------------------------------------------------------ licences
def test_commercial_swaps_only_the_default_datasets():
    args = trainer_args(commercial=True)
    llm_train.resolve_datasets(args)
    assert args.general_dataset == "OpenAssistant/oasst_top1_2023-08-25"
    assert args.reasoning_dataset == "nvidia/Nemotron-Post-Training-Dataset-v1:math"
    assert args.minecraft_qa_dataset == llm_train.MINECRAFT_DATA == "minecraft-data"
    assert args.vision_dataset == "shinohararin/minecraft-raytrace"
    mine = trainer_args(commercial=True, general_dataset="me/my-chat")
    llm_train.resolve_datasets(mine)
    assert mine.general_dataset == "me/my-chat", "a dataset you chose yourself is kept"
    regular = trainer_args()
    llm_train.resolve_datasets(regular)
    assert regular.minecraft_qa_dataset == llm_train.MINECRAFT_QA_DATASET and regular.general_dataset == llm_train.GENERAL_DATASET


def test_every_commercial_default_is_licensed_for_commercial_use():
    for spec in (llm_train.COMMERCIAL_GENERAL_DATASET, llm_train.COMMERCIAL_REASONING_DATASET, llm_train.MINECRAFT_DATA,
                 llm_train.COMMERCIAL_VISION_DATASET):
        licence, commercial, url = llm_train.dataset_license(spec)
        assert commercial and url.startswith("https://"), spec
    for spec in (llm_train.GENERAL_DATASET, llm_train.MINECRAFT_QA_DATASET, llm_train.VISION_STREAM_DATASET,
                 "minecraft.wiki", "lparkourer10/minecraft-alpacadataset"):
        assert llm_train.dataset_license(spec)[1] is False, f"{spec} is not for commercial use"
    assert llm_train.hub_split("nvidia/x:math", "cot") == ("nvidia/x", "math")
    assert llm_train.hub_split("a/b", "train") == ("a/b", "train")


def test_commercial_problems_block_non_commercial_sets_and_warn_on_unknown_ones():
    vl = llm_train.PRESETS["minecraftism-vl"]
    args = trainer_args(commercial=True, general=100, minecraft_qa=50, stream_vision=10)
    llm_train.resolve_datasets(args)
    assert llm_train.commercial_problems(args, vl) == ([], [])
    args.minecraft_qa_dataset = llm_train.MINECRAFT_QA_DATASET
    args.vision_dataset = "someone/pictures"
    problems, warnings = llm_train.commercial_problems(args, vl)
    assert len(problems) == 1 and "naklecha" in problems[0] and "CC BY-NC-SA" in problems[0]
    assert len(warnings) == 1 and "someone/pictures" in warnings[0]


def test_wiki_rows_are_dropped_and_sources_are_credited():
    rows = [{"source": "wiki"}, {"source": "turn"}, {"source": "mobs"}, {"source": "minecraft-data"},
            {"source": "reasoning"}, {"source": "wiki"}]
    kept, dropped = llm_train.drop_noncommercial(rows)
    assert dropped == {"wiki": 2} and len(kept) == 4
    args = trainer_args(commercial=True)
    llm_train.resolve_datasets(args)
    sources = {s["name"]: s for s in llm_train.data_sources(kept, args)}
    assert set(sources) == {"pymc-bot", "minecraft-data", "nvidia/Nemotron-Post-Training-Dataset-v1"}
    assert sources["pymc-bot"]["rows"] == 2 and all(s["commercial"] for s in sources.values())
    assert sources["nvidia/Nemotron-Post-Training-Dataset-v1"]["license"] == "CC BY 4.0"


# ------------------------------------------------------------------ chat + reasoning formats
def test_openassistant_chatml_text_becomes_chat_rows():
    record = {"text": "<|im_start|>user\nName 3 dog names<|im_end|>\n<|im_start|>assistant\nRex, Max, Luna<|im_end|>\n"
                      "<|im_start|>user\nOne more?<|im_end|>\n<|im_start|>assistant\nBuddy.<|im_end|>\n"}
    assert [t["role"] for t in llm_train.chatml_turns(record["text"])] == ["user", "assistant", "user", "assistant"]
    rows = llm_train.general_rows([record, {"text": "no chat here"}], 5, "SYS", uncensored=True)
    assert len(rows) == 1
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user", "assistant", "user", "assistant"]
    assert rows[0]["messages"][0]["content"] == "SYS" and rows[0]["messages"][-1]["content"] == "Buddy."


def test_nemotron_records_give_real_think_traces():
    good = {"reasoning": "on", "messages": [
        {"role": "user", "content": "What is 2+3?", "tool_calls": []},
        {"role": "assistant", "content": "<think>\n2 plus 3 is 5.\n</think>\n\nThe answer is 5.", "tool_calls": []}]}
    redacted = {"reasoning": "on", "messages": [{"role": "user", "content": ""},
                                                {"role": "assistant", "content": "<think>x</think>y"}]}
    off = {"reasoning": "off", "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]}
    rows = llm_train.reasoning_rows([redacted, off, good], 5, "SYS")
    assert len(rows) == 1
    assert rows[0]["reasoning"] == "2 plus 3 is 5." and rows[0]["messages"][-1]["content"] == "The answer is 5."
    assert rows[0]["messages"][1]["content"] == "What is 2+3?"


# ------------------------------------------------------------------ minecraft-data facts
def fake_minecraft_data() -> dict:
    items = [{"id": 1, "name": "stick", "displayName": "Stick", "stackSize": 64},
             {"id": 2, "name": "oak_planks", "displayName": "Oak Planks", "stackSize": 64},
             {"id": 3, "name": "diamond", "displayName": "Diamond", "stackSize": 64},
             {"id": 4, "name": "diamond_ore", "displayName": "Diamond Ore", "stackSize": 64},
             {"id": 5, "name": "wooden_pickaxe", "displayName": "Wooden Pickaxe", "stackSize": 1, "maxDurability": 59},
             {"id": 6, "name": "iron_pickaxe", "displayName": "Iron Pickaxe", "stackSize": 1, "maxDurability": 250,
              "repairWith": ["iron_ingot"]},
             {"id": 7, "name": "diamond_pickaxe", "displayName": "Diamond Pickaxe", "stackSize": 1, "maxDurability": 1561},
             {"id": 8, "name": "iron_ingot", "displayName": "Iron Ingot", "stackSize": 64},
             {"id": 9, "name": "ender_pearl", "displayName": "Ender Pearl", "stackSize": 16},
             {"id": 10, "name": "glass", "displayName": "Glass", "stackSize": 64},
             {"id": 11, "name": "glowstone", "displayName": "Glowstone", "stackSize": 64},
             {"id": 12, "name": "cooked_beef", "displayName": "Steak", "stackSize": 64},
             {"id": 13, "name": "rotten_flesh", "displayName": "Rotten Flesh", "stackSize": 64},
             {"id": 14, "name": "cow_spawn_egg", "displayName": "Cow Spawn Egg", "stackSize": 64}]
    blocks = [{"name": "diamond_ore", "displayName": "Diamond Ore", "hardness": 3.0, "resistance": 3.0, "diggable": True,
               "harvestTools": {"6": True, "7": True}, "drops": [3]},
              {"name": "glass", "displayName": "Glass", "hardness": 0.3, "resistance": 0.3, "diggable": True, "drops": []},
              {"name": "glowstone", "displayName": "Glowstone", "hardness": 0.3, "resistance": 0.3, "diggable": True,
               "emitLight": 15, "drops": [11]},
              {"name": "wall_torch", "displayName": "Wall Torch", "hardness": 0, "resistance": 0, "emitLight": 14}]
    return {"version": "1.21.8", "items": items, "blocks": blocks,
            "foods": [{"name": "cooked_beef", "displayName": "Steak", "foodPoints": 8.0, "saturation": 204.8}],
            "recipes": {"5": [{"inShape": [[2, 2, 2], [None, 1, None], [None, 1, None]], "result": {"id": 5, "count": 1}},
                              {"inShape": [[2, 2, 2], [None, 1, None], [None, 1, None]], "result": {"id": 5, "count": 1}}],
                        "1": [{"inShape": [[2], [2]], "result": {"id": 1, "count": 4}}]},
            "entities": [{"name": "zombie", "displayName": "Zombie", "width": 0.6, "height": 1.95, "type": "hostile"},
                         {"name": "arrow", "displayName": "Arrow", "width": 0.5, "height": 0.5, "type": "projectile"}],
            "entityLoot": [{"entity": "zombie", "drops": [
                {"item": "rotten_flesh", "dropChance": 1, "stackSizeRange": [0, 2]},
                {"item": "iron_ingot", "dropChance": 0.008333, "stackSizeRange": [1, 1], "playerKill": True}]}],
            "enchantments": [{"name": "mending", "displayName": "Mending", "maxLevel": 1, "treasureOnly": True,
                              "exclude": ["infinity"], "category": "durability"},
                             {"name": "infinity", "displayName": "Infinity", "maxLevel": 1, "exclude": ["mending"],
                              "category": "bow"},
                             {"name": "sharpness", "displayName": "Sharpness", "maxLevel": 5, "category": "sharp_weapon"}],
            "effects": [{"name": "Poison", "displayName": "Poison", "type": "bad"}],
            "biomes": [{"name": "desert", "displayName": "Desert", "has_precipitation": False, "temperature": 2.0,
                        "dimension": "overworld"},
                       {"name": "snowy_plains", "displayName": "Snowy Plains", "has_precipitation": True,
                        "temperature": 0.0, "dimension": "overworld"}]}


def test_minecraft_data_facts_are_exact_game_data():
    facts = {questions[0]: answer for questions, answer, _topic in llm_train.minecraft_data_facts(fake_minecraft_data())}
    assert facts["What do I need to mine Diamond Ore?"].startswith("You need an iron pickaxe or better")
    assert facts["What does Diamond Ore drop when mined?"].startswith("Mining Diamond Ore drops Diamond")
    assert "Silk Touch" in facts["What does Glass drop when mined?"]
    assert facts["Does Glowstone give off light?"] == "Yes, Glowstone gives off light level 15."
    assert facts["How much durability does Iron Pickaxe have?"] == ("Iron Pickaxe has 250 durability. Repair it with "
                                                                    "Iron Ingot on an anvil.")
    assert facts["What's the max stack size of Ender Pearl?"] == "Ender Pearl stacks up to 16 per slot."
    assert facts["How much hunger does Steak restore?"] == "Steak restores 8 hunger points (4 drumsticks)."
    craft = facts["How do I craft Wooden Pickaxe?"]
    assert "3 Oak Planks and 2 Stick" in craft and "needs a crafting table" in craft and "one of 2 recipe variants" in craft
    assert "2x2 inventory grid works" in facts["How do I craft Stick?"] and "You get 4 Stick" in facts["How do I craft Stick?"]
    assert facts["What does a Zombie drop?"] == ("A Zombie drops Rotten Flesh and Iron Ingot (0.83% chance) when killed "
                                                 "by a player.")
    assert "fits through a 1x2 doorway" in facts["How big is a Zombie?"]
    assert facts["What's the max level of Sharpness?"] == "Sharpness goes up to level V."
    assert facts["Can I combine Mending with Infinity?"].startswith("No")
    assert "treasure enchantment" in facts["Can I get Mending from an enchanting table?"]
    assert facts["Is Poison a good or a bad effect?"] == "Poison is a harmful status effect."
    assert facts["Does it rain in Desert?"].startswith("No") and "snows" in facts["Does it rain in Snowy Plains?"]
    text = " ".join(facts)
    assert "Wall Torch" not in text and "Spawn Egg" not in text and "Arrow" not in text, "technical things are skipped"


def test_minecraft_data_rows_reuse_facts_with_new_phrasings():
    data = fake_minecraft_data()
    facts = llm_train.minecraft_data_facts(data)
    rows = llm_train.minecraft_data_rows(data, 10_000, "SYS", random.Random(3))
    questions = [row["messages"][1]["content"] for row in rows]
    assert len(facts) < len(rows) <= 3 * len(facts) and len(set(questions)) == len(questions)
    assert all(row["source"] == "minecraft-data" and row["messages"][0]["content"] == "SYS" for row in rows)
    assert all("{" not in row["reasoning"] and "think>" not in row["reasoning"] for row in rows)
    assert len(llm_train.minecraft_data_rows(data, 5, "SYS", random.Random(3))) == 5


def test_load_minecraft_data_from_a_folder_or_github(tmp_path):
    data = fake_minecraft_data()
    root = tmp_path / "minecraft-data" / "data"
    (root / "pc" / "1.21.8").mkdir(parents=True)
    (root / "pc" / "1.20").mkdir(parents=True)
    paths = {name: "pc/1.21.8" for name in llm_train.MINECRAFT_DATA_FILES}
    paths["entityLoot"] = "pc/1.20"
    (root / "dataPaths.json").write_text(json.dumps({"pc": {"1.21.8": paths}}), encoding="utf-8")
    for name in llm_train.MINECRAFT_DATA_FILES:
        (root / paths[name] / f"{name}.json").write_text(json.dumps(data[name]), encoding="utf-8")
    loaded = llm_train.load_minecraft_data("1.21.8", str(tmp_path / "minecraft-data"))
    assert loaded["recipes"] == data["recipes"] and loaded["entityLoot"] == data["entityLoot"]
    asked = []

    def fetch(rel):
        asked.append(rel)
        return {"pc": {"1.21.8": paths}} if rel == "dataPaths.json" else data[rel.rsplit("/", 1)[1][:-5]]

    assert llm_train.load_minecraft_data("1.21.8", fetch=fetch)["items"] == data["items"]
    assert "pc/1.20/entityLoot.json" in asked
    with pytest.raises(ValueError, match="no Java version 9.9"):
        llm_train.load_minecraft_data("9.9", fetch=fetch)


# ------------------------------------------------------------------ minecraft-raytrace pictures
def png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\nfake-frame"


def capture(index: int, *, mobs=True, dark=False) -> dict:
    cells = [{"blockId": "minecraft:grass_block", "depth": 9.0} for _ in range(60)]
    cells += [{"kind": "sky"} for _ in range(20)]
    cells[34] = {"blockId": "minecraft:oak_log", "depth": 4.4}  # the middle of a 10x8 grid
    labels = {"width": 10, "height": 8, "cells": cells}
    entities = [{"type": "minecraft:cow", "hitCells": [[3, 3]]}, {"type": "minecraft:cow", "hitCells": [[4, 3]]},
                {"type": "minecraft:creeper", "hitCells": []}, {"type": "minecraft:item", "hitCells": [[1, 1]]}]
    return {"capture_id": f"cap-{index}", "frame_png": png_bytes(), "labels_gz": gzip.compress(json.dumps(labels).encode()),
            "entities_json": json.dumps(entities if mobs else []), "meta_json": json.dumps({"weather": "clear"}),
            "biome": "minecraft:plains", "lighting": "natural", "world_time": 6000 + 24000 * index,
            "raining": False, "mean_luminance": 5.0 if dark else 80.0}


def test_raytrace_questions_come_from_the_labels():
    answers = dict(llm_train.raytrace_qa(capture(1), random.Random(0)))
    by_topic = {q: a for q, a in answers.items()}
    joined = " | ".join(f"{q} -> {a}" for q, a in by_topic.items())
    assert "This is a plains biome." in by_topic.values()
    assert "It's daytime." in by_topic.values()
    assert "No, the weather is clear." in by_topic.values()
    assert "Oak log, about 4 blocks away." in by_topic.values(), joined
    assert "Yes: 2 cows." in by_topic.values(), "creeper with no hit cells and item entities are not mobs in view"
    assert any(a.startswith("Mostly grass block (about 74% of the view), then oak log, with open sky above") for a in
               by_topic.values()), joined
    assert any(a.startswith("A first-person view in a plains biome, daytime") for a in by_topic.values())
    night = dict(llm_train.raytrace_qa({**capture(2, mobs=False), "world_time": 18000, "lighting": "nightvision",
                                        "raining": True, "biome": "minecraft:snowy_taiga"}, random.Random(0)))
    assert "It's night - the picture is lit up by night vision." in night.values()
    assert "Yes, it's snowing." in night.values() and "No, there are no mobs in view." in night.values()
    assert llm_train.visible_entities([{"type": "cow"}]) is None, "unknown entity layout -> no mob question"
    assert llm_train.label_cells([[{"blockId": "a"}, {"blockId": "b"}]])[1][:3] == (1.0, 0.0, "b")


class FakeParquetFile:
    def __init__(self, handle):
        self.records = json.loads(handle.read())["records"]
        self.schema_arrow = types.SimpleNamespace(names=list(llm_train._RAYTRACE_COLUMNS) + ["region_gz"])
        self.num_row_groups = 1
        self.columns_read = None

    def read_row_group(self, group, columns=None):
        FakeParquetFile.columns = columns
        return types.SimpleNamespace(to_pylist=lambda: [{k: r[k] for k in columns if k in r} for r in self.records])


class FakeShards:
    """HfFileSystem's ls + open for parquet shards (records stored as JSON, bytes as latin-1 text)."""

    def __init__(self, shards: dict[str, list[dict]]):
        self.shards = shards
        self.opened: list[str] = []

    def ls(self, path, detail=True):
        return [{"name": f"{path}/{name}", "size": 1000 * len(records)} for name, records in self.shards.items()]

    def open(self, path, mode="rb", block_size=None):
        self.opened.append(Path(path).name)
        records = [{k: (v.decode("latin-1") if isinstance(v, bytes) else v) for k, v in r.items()}
                   for r in self.shards[Path(path).name]]
        return io.BytesIO(json.dumps({"records": records}).encode())


@pytest.fixture
def fake_pyarrow(monkeypatch):
    def decode(handle):
        parquet = FakeParquetFile(handle)
        for record in parquet.records:
            for key in ("frame_png", "labels_gz"):
                record[key] = record[key].encode("latin-1")
        return parquet

    parquet = types.SimpleNamespace(ParquetFile=decode)
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace(parquet=parquet))
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    monkeypatch.setitem(sys.modules, "PIL", None)  # keep the original PNG bytes
    return parquet


def test_raytrace_streams_one_shard_at_a_time(fake_pyarrow, tmp_path):
    hub = FakeShards({"captures-000.parquet": [capture(i) for i in range(30)] + [capture(99, dark=True)],
                      "captures-001.parquet": [capture(i) for i in range(30, 60)],
                      "captures-017.parquet": [capture(i) for i in range(60, 64)],
                      "README.md": []})
    logged = []
    rows = llm_train.stream_raytrace_rows(50, tmp_path / "shots", "SYS", seed=1, fs=hub, log=logged.append)
    assert len(rows) == 50 and all(row["source"] == "minecraft-raytrace" and row["kind"] == "vision" for row in rows)
    assert "region_gz" not in FakeParquetFile.columns, "the big voxel dumps are never read"
    assert len({row["images"][0] for row in rows}) == 25, "two questions per frame"
    assert all(Path(row["images"][0]).read_bytes() == png_bytes() for row in rows)
    assert len(hub.opened) <= 2 and "README.md" not in hub.opened
    assert any("streamed 50 picture rows" in line for line in logged)
    small = FakeShards(hub.shards)
    quick = llm_train.stream_raytrace_rows(6, tmp_path / "quick", "SYS", seed=1, fs=small, log=lambda _m: None)
    assert small.opened == ["captures-017.parquet"] and len(quick) == 6, "a quick test reads the smallest shard"


# ------------------------------------------------------------------ trainer wiring
def test_extra_rows_use_the_commercial_sources(monkeypatch, capsys, tmp_path):
    args = trainer_args(commercial=True, no_identity=True, general=0, minecraft_qa=40, stream_vision=10,
                        out=str(tmp_path), system_prompt="S")
    llm_train.resolve_datasets(args)
    vl = llm_train.PRESETS["minecraftism-vl"]
    assert llm_train.extra_rows(args, vl, random.Random(0), offline=True) == []
    out = capsys.readouterr().out
    assert "would generate up to 40 Minecraft Q&A rows from minecraft-data 1.21.8 (MIT)" in out
    assert "would stream 10 Minecraft pictures from shinohararin/minecraft-raytrace (one parquet shard" in out
    monkeypatch.setattr(llm_train, "load_minecraft_data", lambda version, folder=None: fake_minecraft_data())
    monkeypatch.setattr(llm_train, "stream_raytrace_rows", lambda limit, *a, **k: [{"source": "minecraft-raytrace"}] * limit)
    rows = llm_train.extra_rows(args, vl, random.Random(0))
    assert sum(r["source"] == "minecraft-data" for r in rows) == 40 and len(rows) == 50


def write_kit(folder: Path) -> Path:
    def row(source, question):
        return {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": question},
                             {"role": "assistant", "content": "an answer"}], "source": source, "kind": "qa", "check": {}}

    folder.mkdir(parents=True, exist_ok=True)
    rows = [row("wiki", f"wiki {i}") for i in range(5)] + [row("mobs", f"mob {i}") for i in range(7)]
    (folder / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return folder


def test_dry_run_with_commercial_leaves_out_the_wiki(tmp_path, capsys):
    kit = write_kit(tmp_path / "kit")
    common = ["--data", str(kit), "--out", str(tmp_path / "out"), "--dry-run", "--no-identity"]
    assert llm_train.main(common + ["--commercial", "--minecraft-qa", "100"]) == 0
    out = capsys.readouterr().out
    assert "commercial: left out 5 rows of Minecraft Wiki text (CC BY-NC-SA 3.0)" in out
    assert "'wiki'" not in out.split("sources=")[1].split("\n")[0]
    assert "minecraft-data 1.21.8 (MIT)" in out
    assert llm_train.main(common + ["--commercial", "--minecraft-qa", "100", "--minecraft-qa-dataset",
                                    llm_train.MINECRAFT_QA_DATASET]) == 2
    assert "not for commercial use" in capsys.readouterr().out
    assert llm_train.main(common) == 0
    assert "'wiki': 5" in capsys.readouterr().out, "without --commercial the kit is used as it is"


def test_model_card_lists_every_source_and_licence():
    preset = llm_train.PRESETS["minecraftism"]
    args = trainer_args(commercial=True)
    llm_train.resolve_datasets(args)
    data = llm_train.data_sources([{"source": "turn"}, {"source": "general"}, {"source": "minecraft-data"}], args)
    card = llm_train.model_card("minecraftism", preset, {"quant": "Q4_K_M", "commercial": True, "data": data}, "me/x")
    assert "## Training data" in card and "| [minecraft-data](https://github.com/PrismarineJS/minecraft-data) | MIT | yes | 1 |" in card
    assert "OpenAssistant/oasst_top1_2023-08-25" in card and "every training source allows commercial use" in card
    assert "Minecraft Wiki text (CC BY-NC-SA" not in card
    old = llm_train.model_card("minecraftism", preset, {"quant": "Q4_K_M"}, "me/x")
    assert "Minecraft Wiki text (CC BY-NC-SA 3.0" in old and "## Training data" not in old


def test_notebook_defaults_to_commercial_data():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "COMMERCIAL_ONLY = True" in code and 'FLAGS += " --commercial" if COMMERCIAL_ONLY else ""' in code
    assert '"--minecraft-qa", "--commercial",' in code, "an old trainer without --commercial gets replaced"
    intro = "".join(nb["cells"][0]["source"])
    assert "COMMERCIAL_ONLY = True" in intro and "minecraft-raytrace" in intro and "Apache-2.0" in intro
