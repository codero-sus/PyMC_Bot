"""Minecraft knowledge: built-in mob facts, web lookups (mocked) and the AI's search-then-act turn."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from pymc_bot.agent import ACTIONS, AgentLoop, Decision
from pymc_bot.config import SearchSettings
from pymc_bot.knowledge import (
    MOB_FACTS,
    KnowledgeBase,
    html_to_text,
    mob_fact,
    normalize_query,
    shared_knowledge,
    shorten,
)
from pymc_bot.mind import ACTION_REFERENCE, Mind, build_system_prompt, build_turn_prompt

WOLF_LEAD = (
    '<div class="mw-parser-output"><div class="hatnote">For the dog, see Dog.</div>'
    '<table class="infobox"><tr><td>Health 8</td></tr></table>'
    "<p>A <b>wolf</b> is a neutral mob{weird}.<sup>[1]</sup> It can be tamed.</p>"
    '<div class="navbox">Mobs | Animals</div></div>'
)
WOLF_TAMING = "<h2>Taming<span class=\"mw-editsection\">[edit]</span></h2><p>Wolves are tamed with <a>bones</a>.</p><ul><li>Tamed wolves sit.</li><li>They follow you.</li></ul>"


def wiki_handler(calls: list[httpx.Request] | None = None, hits: list[str] | None = None):
    hits = ["Wolf", "Dog"] if hits is None else hits

    def handle(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        params = dict(request.url.params)
        if request.url.host != "minecraft.wiki":
            return httpx.Response(404)
        assert request.headers["user-agent"].startswith("PyMC_Bot/")
        if params.get("list") == "search":
            return httpx.Response(200, json={"query": {"search": [{"title": title} for title in hits]}})
        if params.get("prop") == "sections":
            return httpx.Response(200, json={"parse": {"title": "Wolf", "sections": [
                {"line": "Spawning", "index": "1"}, {"line": "Behavior", "index": "2"}, {"line": "Taming", "index": "3"},
                {"line": "Drops", "index": "4"}]}})
        if params.get("section") == "0":
            return httpx.Response(200, json={"parse": {"text": WOLF_LEAD}})
        if params.get("section") == "3":
            return httpx.Response(200, json={"parse": {"text": WOLF_TAMING}})
        if params.get("section") == "4":
            return httpx.Response(200, json={"parse": {"text": "<p>Wolves drop nothing but experience.</p>"}})
        return httpx.Response(200, json={"parse": {"text": "<p>Other section.</p>"}})

    return handle


def kb_with(handler, **settings) -> KnowledgeBase:
    return KnowledgeBase(SearchSettings(**settings), None, transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ built-in facts
def test_every_dangerous_mob_has_a_fact():
    for mob in ("creeper", "skeleton", "zombie", "enderman", "spider", "witch", "warden", "piglin", "blaze", "phantom"):
        assert mob_fact(mob), mob
    assert "explode" in MOB_FACTS["creeper"].lower()
    assert "look" in MOB_FACTS["enderman"]
    assert mob_fact("minecraft:creeper") == MOB_FACTS["creeper"]
    assert mob_fact("some_modded_beast") is None
    assert all("{" not in fact and "}" not in fact for fact in MOB_FACTS.values())


def test_html_to_text_drops_tables_navboxes_and_footnotes():
    text = html_to_text(WOLF_LEAD + WOLF_TAMING)
    assert "A wolf is a neutral mob(weird). It can be tamed." in text
    assert "Health" not in text and "Mobs | Animals" not in text and "[1]" not in text and "For the dog" not in text
    assert "[edit]" not in text
    assert "Wolves are tamed with bones.\n- Tamed wolves sit.\n- They follow you." in text


def test_shorten_cuts_at_a_sentence():
    text = "First sentence here. Second sentence is longer than the rest of it."
    assert shorten(text, 30) == "First sentence here."
    assert shorten("short", 30) == "short"
    assert normalize_query("  How do I tame a WOLF?? ") == "how do i tame a wolf"


# --------------------------------------------------------------------- providers
def test_minecraft_wiki_reads_the_lead_and_the_section_that_fits():
    calls: list[httpx.Request] = []
    result = kb_with(wiki_handler(calls)).lookup("how do i tame a wolf")
    assert result.ok, result.error
    assert result.title == "Wolf" and result.source == "minecraft_wiki"
    assert result.url == "https://minecraft.wiki/w/Wolf"
    assert "neutral mob" in result.text and "Taming: Wolves are tamed with bones." in result.text
    assert "Related pages: Dog" in result.text
    assert dict(calls[0].url.params)["srsearch"] == "how do i tame a wolf"

    drops = kb_with(wiki_handler()).lookup("what does a wolf drop")
    assert "Drops: Wolves drop nothing" in drops.text
    lead_only = kb_with(wiki_handler()).lookup("wolf")
    assert "Taming:" not in lead_only.text and "neutral mob" in lead_only.text


def test_falls_back_to_duckduckgo_when_the_wiki_has_nothing():
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.host == "minecraft.wiki":
            return httpx.Response(200, json={"query": {"search": []}})
        assert request.url.host == "api.duckduckgo.com"
        assert dict(request.url.params)["q"] == "minecraft sniffer egg"
        return httpx.Response(200, json={"Heading": "Sniffer", "AbstractText": "A sniffer egg hatches into a sniffer.",
                                         "AbstractURL": "https://example.org/sniffer"})

    result = kb_with(handle).lookup("sniffer egg")
    assert result.ok and result.source == "duckduckgo" and result.title == "Sniffer"

    no_fallback = kb_with(handle, fallback=False).lookup("sniffer egg")
    assert not no_fallback.ok and "nothing on the wiki" in no_fallback.error


def test_duckduckgo_related_topics_and_searxng_and_brave(monkeypatch):
    def ddg(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"AbstractText": "", "RelatedTopics": [
            {"Text": "Creeper - a mob that explodes", "FirstURL": "https://x/creeper"},
            {"Name": "Group", "Topics": [{"Text": "Charged creeper", "FirstURL": "https://x/charged"}]}]})

    result = kb_with(ddg, provider="duckduckgo").lookup("creeper")
    assert result.ok and "- Creeper - a mob that explodes" in result.text and "Charged creeper" in result.text

    def searx(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search" and dict(request.url.params)["format"] == "json"
        return httpx.Response(200, json={"results": [{"title": "Piglin", "url": "https://w/piglin", "content": "Wear <b>gold</b>."}]})

    result = kb_with(searx, provider="searxng", searxng_url="http://searx.local").lookup("piglin")
    assert result.ok and result.text == "- Piglin: Wear gold." and result.url == "https://w/piglin"
    assert not kb_with(searx, provider="searxng", fallback=False).lookup("piglin").ok  # no URL configured

    seen = {}

    def brave(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers.get("x-subscription-token")
        return httpx.Response(200, json={"web": {"results": [{"title": "Blaze", "url": "https://w/blaze",
                                                              "description": "Drops <strong>blaze rods</strong>."}]}})

    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    missing = kb_with(brave, provider="brave", fallback=False).lookup("blaze")
    assert not missing.ok and "BRAVE_API_KEY" in missing.error
    monkeypatch.setenv("BRAVE_API_KEY", "secret-key")
    result = kb_with(brave, provider="brave").lookup("blaze")
    assert result.ok and "blaze rods" in result.text and seen["token"] == "secret-key"
    assert "secret-key" not in json.dumps(result.to_dict())


def test_errors_never_raise():
    def broken(request: httpx.Request) -> httpx.Response:
        if request.url.host == "minecraft.wiki":
            return httpx.Response(503)
        return httpx.Response(200, text="<html>blocked</html>")

    result = kb_with(broken).lookup("creeper")
    assert not result.ok and "HTTP 503" in result.error and "not JSON" in result.error

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    assert "ConnectError" in kb_with(boom, fallback=False).lookup("creeper").error
    assert kb_with(boom).lookup("   ").error == "empty question"


# ------------------------------------------------------------- cache and limits
def test_answers_are_cached_on_disk_and_requests_are_capped(tmp_path):
    calls: list[httpx.Request] = []
    path = tmp_path / "knowledge_cache.json"
    kb = KnowledgeBase(SearchSettings(max_per_hour=1), path, transport=httpx.MockTransport(wiki_handler(calls)))
    first = kb.lookup("tame wolf")
    assert first.ok and not first.cached
    n = len(calls)
    again = kb.lookup("Tame  WOLF?")
    assert again.cached and again.text == first.text and len(calls) == n
    assert json.loads(path.read_text())["tame wolf"]["title"] == "Wolf"

    limited = kb.lookup("wolf drops")
    assert not limited.ok and "limit reached (1/hour)" in limited.error and len(calls) == n

    reloaded = KnowledgeBase(SearchSettings(), path, transport=httpx.MockTransport(wiki_handler([])))
    assert reloaded.lookup("tame wolf").cached
    assert reloaded.status()["cached_answers"] == 1
    assert reloaded.clear_cache() == 1 and reloaded.cached("tame wolf") is None


def test_cache_expires_and_disabled_search_uses_only_the_cache(tmp_path):
    kb = kb_with(wiki_handler(), cache_days=1)
    assert kb.lookup("wolf").ok
    kb._cache["wolf"]["at"] = time.time() - 2 * 86400
    assert kb.cached("wolf") is None
    kb.settings = SearchSettings(enabled=False, cache_days=1)
    assert kb.lookup("wolf").error == "web search is switched off"


def test_unknown_mobs_are_looked_up_in_the_background():
    calls: list[httpx.Request] = []
    kb = kb_with(wiki_handler(calls, hits=["Frostling"]))
    assert not kb.prefetch_mob("creeper")  # built in
    assert not kb.prefetch_mob("player")
    assert kb.prefetch_mob("othermod:frostling")
    deadline = time.time() + 5
    while kb.cached("mob frostling") is None and time.time() < deadline:
        time.sleep(0.02)
    assert kb.cached("mob frostling") is not None
    assert not kb.prefetch_mob("frostling")  # already known now
    facts = kb.facts_for(["minecraft:creeper", "frostling", "player", "nothing_known"])
    assert facts[0].startswith("creeper: ") and facts[1].startswith("frostling: ")
    assert len(facts) == 2
    assert kb.recent == type(kb.recent)([], maxlen=kb.recent.maxlen)  # background lookups stay out of "recent"


def test_shared_knowledge_is_one_per_cache_file(tmp_path):
    a = shared_knowledge(SearchSettings(), tmp_path)
    b = shared_knowledge(SearchSettings(max_per_hour=5), tmp_path)
    assert a is b and b.settings.max_per_hour == 5
    assert a.cache_path == tmp_path / "knowledge_cache.json"
    assert shared_knowledge(SearchSettings(cache_file=""), tmp_path).cache_path is None


# --------------------------------------------------------------------- prompt
def test_prompt_mentions_search_and_shows_facts_and_lookups():
    assert "search" in ACTIONS and '"query"' in ACTION_REFERENCE
    system = build_system_prompt("Bot")
    assert '"search"' in system and "Don't guess about the game" in system
    off = build_system_prompt("Bot", allowed={"search": False})
    assert "Web search is switched off" in off

    mind = Mind()
    mind.note_lookup("piglin behavior", "Piglins are neutral if you wear gold {armor}.")
    mind.note_lookup("Piglin behavior", "Piglins are neutral if you wear gold.")  # replaces, no duplicate
    state = {"health": 20, "food": 20, "position": {"x": 0, "y": 64, "z": 0}, "players": [], "entities": []}
    prompt = build_turn_prompt(state, "Bot", mind, [], {"search": True},
                               knowledge=["creeper: " + MOB_FACTS["creeper"]])
    narrative = prompt.split("EXACT DATA (JSON):")[0]
    assert "What you know about the mobs around: creeper:" in narrative
    assert "WHAT YOU LOOKED UP" in narrative and narrative.count("piglin behavior") + narrative.count("Piglin behavior") == 1
    assert "{" not in narrative
    assert mind.snapshot()["lookups"][0]["found"] is True


# ---------------------------------------------------------------------- agent
class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.base_url = self.model = ""
        self.temperature = self.timeout = 0

    def chat(self, messages, system=None):
        self.prompts.append(messages[-1]["content"])
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


@pytest.fixture()
def wiki(monkeypatch, store):
    """Route the agent's shared knowledge base to the mocked wiki."""
    calls: list[httpx.Request] = []
    kb = shared_knowledge(store.settings.search, store.path.parent)
    kb._transport = httpx.MockTransport(wiki_handler(calls))
    return calls


def test_llm_searches_then_acts_in_the_same_turn(live_bot, store, log, wiki):
    store.update({"agent": {"mode": "ollama", "human_chat": False}})
    fake = FakeLLM([
        '{"thought": "how do wolves work", "action": "search", "query": "how to tame a wolf"}',
        '{"thought": "bones it is", "goal": "tame a wolf", "action": "wander"}',
    ])
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    ok, detail = agent.run_once()
    assert ok, detail
    assert len(fake.prompts) == 2
    assert "WHAT YOU LOOKED UP" not in fake.prompts[0]
    assert "WHAT YOU LOOKED UP" in fake.prompts[1] and "tamed with bones" in fake.prompts[1]
    assert "do not search for the same thing again" in fake.prompts[1]
    assert agent.status["last_decision"]["action"] == "wander"
    messages = [event["message"] for event in log.tail(40)]
    assert any(message.startswith('🔎 looked up "how to tame a wolf" (Wolf') for message in messages)
    # the next turn still remembers the answer
    agent.run_once()
    assert "tamed with bones" in fake.prompts[-1]


def test_nearby_mob_facts_reach_the_prompt(live_bot, store, log, wiki):
    store.update({"agent": {"mode": "ollama", "human_chat": False}})
    fake = FakeLLM(['{"action": "wait", "seconds": 1}'])
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    state = {"health": 20, "food": 20, "position": {"x": 0, "y": 64, "z": 0}, "players": [],
             "entities": [{"type": "minecraft:creeper", "x": 3, "y": 64, "z": 0, "dist": 3, "hostile": True},
                          {"type": "minecraft:player", "player": True, "dist": 2}]}
    prompt = agent._build_prompt(store.settings, state)
    assert "creeper: walks up silently" in prompt


def test_search_rounds_are_limited_and_can_be_switched_off(live_bot, store, log, wiki):
    store.update({"agent": {"mode": "ollama", "human_chat": False}, "search": {"max_per_turn": 1}})
    fake = FakeLLM(['{"action": "search", "query": "wolf"}'])  # a model that only ever searches
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    ok, detail = agent.run_once()
    assert len(fake.prompts) == 2  # one research round, then the second search is executed as the action
    assert ok and detail.startswith("looked up wolf")

    store.update({"agent": {"allow_web_search": False}})
    fake = FakeLLM(['{"action": "search", "query": "wolf"}'])
    agent = AgentLoop(live_bot, store, log, ollama=fake)
    ok, detail = agent.run_once()
    assert len(fake.prompts) == 1 and not ok and "web search is disabled" in detail
    assert "Web search is switched off" in agent.system_prompt(store.settings, store.settings.ollama.system_prompt)
    assert agent.allowed(store.settings)["search"] is False


def test_manual_search_action_and_aliases(live_bot, store, log, wiki):
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.execute(Decision("lookup", {"query": "tame a wolf"}, "manual"), enforce_permissions=False)
    assert ok and "Wolves are tamed with bones" in detail
    ok, detail = agent.execute(Decision("search", {}, "manual"))
    assert not ok and detail == "search needs a query"
    assert agent.mind.snapshot()["lookups"][-1]["query"] == "tame a wolf"


# ------------------------------------------------------------------- server/CLI
def test_knowledge_api(store, wiki):
    from pymc_bot.server import create_app

    client = TestClient(create_app(config_path=store.path))
    status = client.get("/api/knowledge").json()
    assert status["enabled"] and status["provider"] == "minecraft_wiki" and status["built_in_mobs"] == len(MOB_FACTS)
    result = client.post("/api/knowledge/search", json={"query": "tame wolf"}).json()
    assert result["ok"] and result["title"] == "Wolf"
    assert client.post("/api/knowledge/search", json={"query": ""}).status_code == 422
    assert client.get("/api/knowledge").json()["recent"][-1]["query"] == "tame wolf"
    assert client.post("/api/knowledge/clear").json() == {"cleared": 1}
    contract = client.get("/api/contract").json()
    assert "search" in json.dumps(contract)


def test_search_cli(store, wiki, capsys):
    from pymc_bot.__main__ import main

    assert main(["search", "tame", "wolf", "--config", str(store.path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Wolf  [minecraft_wiki]") and "https://minecraft.wiki/w/Wolf" in out
    assert main(["search", "tame", "wolf", "--json", "--config", str(store.path)]) == 0
    assert json.loads(capsys.readouterr().out)["cached"] is True
    # --no-cache builds a fresh base without the mock -> offline in tests
    assert main(["search", "tame", "wolf", "--no-cache", "--config", str(store.path)]) == 1
    assert "No answer" in capsys.readouterr().out
