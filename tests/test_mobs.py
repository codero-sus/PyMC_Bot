"""The mob database (pymc_bot/data/mobs.json): schema, lookups, versions, the AI and the API/CLI."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from pymc_bot.config import SearchSettings
from pymc_bot.knowledge import MOB_FACTS, KnowledgeBase
from pymc_bot.mobs import (
    CATEGORIES,
    DB_PATH,
    MobDatabase,
    is_version,
    load_mobs,
    validate,
    version_at_least,
    version_key,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def db() -> MobDatabase:
    return load_mobs()


def no_web(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must never be called
    raise AssertionError(f"the web was asked: {request.url}")


# ------------------------------------------------------------------ the file itself
def test_the_json_file_is_valid_and_complete(db):
    data = json.loads(DB_PATH.read_text(encoding="utf-8"))
    assert validate(data) == []
    assert len(db) >= 85
    assert data["edition"] == "java" and "minecraft.wiki" in data["source"]
    assert any("Difficulty only scales damage dealt to players" in note for note in data["notes"])
    for mob_id in ("zombie", "skeleton", "creeper", "spider", "enderman", "witch", "warden", "piglin", "blaze",
                   "ghast", "wither", "ender_dragon", "breeze", "creaking", "happy_ghast", "copper_golem",
                   "nautilus", "parched", "camel_husk", "sulfur_cube", "cow", "villager", "wolf", "iron_golem"):
        assert mob_id in db.mobs, mob_id
    assert {m["category"] for m in db.mobs.values()} == set(CATEGORIES)
    assert set(MOB_FACTS) == set(db.mobs)  # the prompt facts come from the database


def test_validate_reports_problems():
    assert validate({"mobs": {}}) == ["'mobs' must be a non-empty object"]
    bad = {"mobs": {"Bad Id": {"name": "X", "category": "scary", "hostile": "yes", "behavior": "", "health": -1,
                               "added_in": "someday", "spawn": {"when": "never", "dimensions": ["moon"]},
                               "attacks": [{"type": "bite", "easy": 1, "normal": [3, 2], "hard": "lots"}],
                               "fact": "{oops}"}}}
    problems = " | ".join(validate(bad))
    for text in ("lower_snake_case", "bad category", "hostile must be", "bad health", "bad added_in", "spawn needs",
                 "spawn.dimensions", "bad normal damage", "bad hard damage", "plain sentence"):
        assert text in problems, text
    assert "missing fact" in validate({"mobs": {"x": {"name": "X"}}})[-1]


def test_verified_stats(db):
    creeper = db.get("creeper")
    assert creeper["hostile"] and creeper["health"] == 20 and creeper["added_in"] == "Classic"
    blast = creeper["attacks"][0]
    assert (blast["easy"], blast["normal"], blast["hard"]) == (22.5, 43, 64.5)
    assert db.to_mobs(blast) == 43  # difficulty only scales damage to players
    warden = db.get("warden")
    assert warden["health"] == 500 and db.max_damage(warden, "hard") == 45 and warden.get("avoid")
    assert db.get("zombie")["attacks"][0]["normal"] == 3 and db.get("zombie")["spawn"]["when"] == "dark"
    assert db.get("skeleton")["attacks"][0]["hard"] == [4, 8]
    assert db.max_damage(db.get("iron_golem")) == 21.5
    assert db.get("enderman")["category"] == "neutral" and not db.get("enderman")["hostile"]
    assert db.get("cow")["attacks"] == [] and db.max_damage(db.get("cow")) == 0
    assert db.to_mobs(db.get("ghast")["attacks"][0]) == 1000  # a returned fireball one-shots a ghast
    assert db.get("happy_ghast")["added_in"] == "1.21.6" and db.get("parched")["added_in"] == "1.21.11"
    assert db.get("zombie_horse")["health"] == 25


# ------------------------------------------------------------------ names
@pytest.mark.parametrize("name, expected", [
    ("sulphur cube", "sulfur_cube"), ("Sulfur Cubes", "sulfur_cube"), ("minecraft:sulfur_cube", "sulfur_cube"),
    ("SULPHUR_CUBES", "sulfur_cube"), ("zombie pigman", "zombified_piglin"), ("Zombie Pigmen", "zombified_piglin"),
    ("endermen", "enderman"), ("wolves", "wolf"), ("gurgle", "drowned"), ("snifflet", "sniffer"),
    ("minecraft:cave_spider", "cave_spider"), ("Ender Dragon", "ender_dragon"), ("snowman", "snow_golem"),
    ("witches", "witch"), ("creepers", "creeper"),
])
def test_names_aliases_and_plurals(db, name, expected):
    assert db.get(name)["id"] == expected


def test_unknown_names(db):
    assert db.get("frostling") is None and db.get("") is None and db.get("blaze rod") is None


def test_longest_name_wins_in_a_sentence(db):
    found = [mob_id for mob_id, _ in db.find_in_text("a cave spider and a zombie horse near a spider")]
    assert found == ["cave_spider", "zombie_horse", "spider"]
    assert db.find_in_text("combat tips") == []  # "bat" only as a whole word


# ------------------------------------------------------------------ versions
def test_version_ordering():
    order = ["Classic", "Indev", "Infdev", "Alpha 1.2.0", "Beta 1.2", "Beta 1.8", "1.0.0", "1.2.1", "1.9", "1.16.2",
             "1.21", "1.21.4", "1.21.11", "26.1", "26.2", "26.3"]
    assert sorted(order, key=version_key) == order
    assert version_at_least("1.21.11", "1.21.4") and not version_at_least("1.21.4", "1.21.11")
    assert version_at_least("1.21", "1.21.0") and version_at_least("26.2", "26.2")
    assert is_version("1.21.4") and is_version("26.2") and not is_version("auto") and not is_version("1")
    with pytest.raises(ValueError):
        version_key("someday")


def test_sulfur_cube_is_newer_than_mineflayer(db):
    cube = db.get("sulphur cube")
    assert cube["added_in"] == "26.2" and cube["category"] == "passive" and not cube["hostile"]
    assert cube["health"] == 8 and cube["spawn"]["when"] == "any" and "sulfur caves" in cube["spawn"]["where"]
    assert not db.supported_by_mineflayer(cube)
    assert db.with_derived(cube)["mineflayer_supported"] is False
    assert [m["id"] for m in db.newer_than_mineflayer()] == ["sulfur_cube"]
    assert db.newer_than_mineflayer("26.1") == [] and db.newer_than_mineflayer("26.3")[0]["id"] == "sulfur_cube"
    assert db.exists_in(cube, "26.2") and not db.exists_in(cube, "1.21.11")
    for mob_id in ("creeper", "parched", "happy_ghast", "copper_golem", "nautilus"):
        assert db.supported_by_mineflayer(db.get(mob_id)), mob_id
    text = db.describe(cube, "1.21.4")
    assert "Added in Java 26.2" in text and "newer than mineflayer" in text and "In 1.21.4: no" in text


def test_search_filters(db):
    nether_hostiles = {m["id"] for m in db.search(hostile=True, dimension="nether")}
    assert {"blaze", "ghast", "wither_skeleton", "hoglin"} <= nether_hostiles and "cow" not in nether_hostiles
    assert {m["id"] for m in db.search(added_since="1.21.5")} == {
        "happy_ghast", "copper_golem", "nautilus", "zombie_nautilus", "camel_husk", "parched", "sulfur_cube"}
    assert "sulfur_cube" not in {m["id"] for m in db.search(exists_in="26.1")}
    assert all(m["category"] == "boss" for m in db.search(category="boss"))
    assert [m["id"] for m in db.search(query="sulphur")] == ["sulfur_cube"]


# ------------------------------------------------------------------ answering questions
@pytest.mark.parametrize("question, mob_id", [
    ("how much damage does a warden do", "warden"), ("is the sulphur cube hostile?", "sulfur_cube"),
    ("does the sulfur cube exist in 1.21.4", "sulfur_cube"), ("when was the camel added", "camel"),
    ("where do husks spawn", "husk"), ("how to tame a wolf", "wolf"), ("creeper", "creeper"),
    ("Zombie Pigmen", "zombified_piglin"), ("what does an enderman drop", "enderman"),
])
def test_questions_the_database_answers(db, question, mob_id):
    assert db.answerable(question)["id"] == mob_id


@pytest.mark.parametrize("question", [
    "what drops blaze rods", "blaze rods", "piglin behavior", "how to tame a fox", "warden farm design",
    "how do i craft a bed", "",
])
def test_questions_left_to_the_web(db, question):
    assert db.answerable(question) is None


def test_lookup_answers_mob_questions_offline_first():
    kb = KnowledgeBase(SearchSettings(), None, transport=httpx.MockTransport(no_web))
    result = kb.lookup("is the sulphur cube hostile?")
    assert result.ok and result.source == "mob_database" and result.title == "Sulfur Cube"
    assert "passive mob" in result.text and "Added in Java 26.2" in result.text
    assert result.url == "https://minecraft.wiki/w/Sulfur_Cube"
    versioned = kb.lookup("is there a sulfur cube in 1.21.11?")
    assert "In 1.21.11: no, it does not exist yet" in versioned.text
    damage = kb.lookup("how much damage does a warden do")
    assert "30" in damage.text and "16/30/45" in damage.text
    assert kb.stats["mob_database"] == 3 and kb.stats["web_requests"] == 0
    assert kb.recent[-1].source == "mob_database" and kb.status()["cached_answers"] == 0
    assert kb.status()["mob_database"]["mobs"] == len(load_mobs())
    # also when web search is off - the database is local
    kb.settings = SearchSettings(enabled=False)
    assert kb.lookup("creeper").source == "mob_database"
    kb.settings = SearchSettings(enabled=False, use_mob_database=False)
    assert kb.lookup("creeper").error == "web search is switched off"


def test_prompt_lines_and_unknown_entities():
    kb = KnowledgeBase(SearchSettings(auto_lookup_mobs=True), None, transport=httpx.MockTransport(no_web))
    lines = kb.facts_for(["minecraft:sulfur_cube", "warden", "unknown"])
    assert lines[0].startswith("sulfur cube (passive, 8 HP, up to 43 dmg): passive slime-like cube")
    assert lines[1].startswith("warden (hostile, 500 HP, up to 30 dmg):")
    assert lines[2].startswith("unknown creature: probably a mob newer than mineflayer - sulfur cube")
    assert kb.facts_for(["unknown"], server_version="26.1") == []  # nothing newer exists there
    assert not kb.prefetch_mob("minecraft:sulfur_cube") and not kb.prefetch_mob("unknown")


def test_the_agent_sees_new_mobs(live_bot, store, log):
    from pymc_bot.agent import AgentLoop

    store.update({"agent": {"mode": "ollama", "human_chat": False}, "minecraft": {"version": "26.2"}})
    agent = AgentLoop(live_bot, store, log)
    state = {"health": 20, "food": 20, "position": {"x": 0, "y": 64, "z": 0}, "players": [],
             "entities": [{"type": "unknown", "x": 2, "y": 64, "z": 0, "dist": 2}]}
    prompt = agent._build_prompt(store.settings, state)
    assert "probably a mob newer than mineflayer - sulfur cube (passive" in prompt
    result = agent.look_up("sulphur cube spawn")
    assert result.ok and result.source == "mob_database" and "sulfur caves" in result.text
    assert "sulfur caves" in agent.mind.snapshot()["lookups"][-1]["answer"]


# ------------------------------------------------------------------ API, CLI, packaging
def test_mobs_api(store):
    from pymc_bot.server import create_app

    client = TestClient(create_app(config_path=store.path))
    listing = client.get("/api/mobs").json()
    assert listing["count"] == len(load_mobs()) and listing["meta"]["mineflayer_max_version"] == "26.1"
    cube = next(m for m in listing["mobs"] if m["id"] == "sulfur_cube")
    assert cube == {"id": "sulfur_cube", "name": "Sulfur Cube", "category": "passive", "hostile": False, "health": 8,
                    "max_damage_normal": 43.0, "spawn_time": "any", "added_in": "26.2", "mineflayer_supported": False}
    hostile_end = client.get("/api/mobs", params={"hostile": "true", "dimension": "end"}).json()["mobs"]
    assert {m["id"] for m in hostile_end} >= {"shulker", "ender_dragon"} and all(m["hostile"] for m in hostile_end)
    new = client.get("/api/mobs", params={"added_since": "26.2", "full": "true"}).json()["mobs"]
    assert [m["id"] for m in new] == ["sulfur_cube"] and new[0]["attacks"][1]["to_mobs"] == 43
    for bad in ({"category": "scary"}, {"dimension": "moon"}, {"spawn_time": "noon"}, {"added_since": "soon"}):
        assert client.get("/api/mobs", params=bad).status_code == 422, bad
    entry = client.get("/api/mobs/sulphur cubes").json()
    assert entry["id"] == "sulfur_cube" and "Added in Java 26.2" in entry["summary"]
    assert client.get("/api/mobs/zombie pigman").json()["id"] == "zombified_piglin"
    assert client.get("/api/mobs/frostling").status_code == 404
    assert client.get("/api/knowledge").json()["mob_database"]["enabled"] is True


def test_mob_cli(capsys):
    from pymc_bot.__main__ import main

    assert main(["mob", "sulphur", "cube", "--version", "1.21.11"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Sulfur Cube: passive mob") and "In 1.21.11: no" in out
    assert main(["mob", "creeper", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["id"] == "creeper" and data["mineflayer_supported"] and data["attacks"][0]["to_mobs"] == 43
    assert main(["mob", "--list", "--since", "26.2"]) == 0
    out = capsys.readouterr().out
    assert "Sulfur Cube" in out and "(newer than mineflayer)" in out and out.rstrip().endswith(f"1 of {len(load_mobs())} mobs")
    assert main(["mob", "--list", "--category", "boss", "--json"]) == 0
    assert {m["id"] for m in json.loads(capsys.readouterr().out)} == {"wither", "ender_dragon"}
    assert main(["mob", "frostling"]) == 1
    assert main(["mob", "--list", "--since", "soon"]) == 2


def test_search_cli_can_skip_the_database(store, capsys):
    from pymc_bot.__main__ import main

    assert main(["search", "is", "a", "creeper", "hostile", "--config", str(store.path)]) == 0
    assert "[mob_database]" in capsys.readouterr().out.splitlines()[0]
    # --no-mob-db goes to the web, which the test suite keeps offline
    assert main(["search", "creeper", "--no-mob-db", "--no-cache", "--config", str(store.path)]) == 1


def test_the_database_ships_with_the_package():
    lines = (ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines()
    package_data = lines[lines.index("[tool.setuptools.package-data]") + 1]
    assert package_data.startswith("pymc_bot = [") and '"data/*.json"' in package_data
    assert DB_PATH == ROOT / "pymc_bot" / "data" / "mobs.json"
