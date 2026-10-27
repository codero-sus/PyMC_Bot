"""The Minecraftism dataset collector (pymc_bot/llm_webdata.py) and its Kaggle notebook."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import types
import urllib.parse
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pymc_bot import llm_webdata as w  # noqa: E402

NOTEBOOK = ROOT / "pynb" / "Minecraftism_Dataset_Kaggle.ipynb"
FANDOM = "https://minecraft.fandom.com/wiki/Lead"

PAGE_HTML = """
<div class="mw-parser-output">
<aside class="portable-infobox"><h2>Lead</h2><div>Stackable 64</div></aside>
<p>A <b>lead</b> is an item used to leash and lead passive mobs.<sup class="reference">[1]</sup></p>
<h2>Obtaining<span class="mw-editsection">[edit]</span></h2>
<p>Leads can be crafted from 5 string. Crafting a lead gives 2 leads.</p>
<table class="wikitable"><tr><th>Ingredients</th><th>Result</th></tr><tr><td>String</td><td>Lead</td></tr></table>
<div class="navbox"><p>Items navigation that should not appear</p></div>
<script>var x = 1;</script>
<p>Hostile mobs cannot be leashed, except zoglins.</p>
</div>
"""


def page_text() -> str:
    return w.html_to_text(PAGE_HTML)


# ------------------------------------------------------------------ sources and licences
def test_only_commercial_friendly_sources_are_used():
    assert w.source_policy(FANDOM) == ("text", "CC BY-SA 3.0")
    assert w.source_policy("https://en.wikipedia.org/wiki/Minecraft")[0] == "text"
    assert w.source_policy("https://www.minecraft.net/en-us/article/minecraft-java-edition-1-21-6")[0] == "check"
    for url in ("https://minecraft.wiki/w/Lead", "https://zh.minecraft.wiki/w/Lead",
                "https://www.reddit.com/r/Minecraft/", "https://minecraft.fandom.com/de/wiki/Leine",
                "https://minecraft.fandom.com/wiki/Talk:Lead", "https://minecraft.fandom.com/wiki/Special:Search"):
        assert w.source_policy(url) is None, url
    with pytest.raises(ValueError):
        w.fetch_page("https://minecraft.wiki/w/Lead", get=lambda *a: pytest.fail("never downloaded"))


def test_html_to_text_keeps_the_article_and_drops_the_clutter():
    text = page_text()
    assert "Leads can be crafted from 5 string." in text and "Hostile mobs cannot be leashed" in text
    assert "Ingredients | Result" in text and "String | Lead" in text
    for junk in ("[1]", "[edit]", "navigation that should not appear", "var x", "Stackable 64"):
        assert junk not in text


def test_fetch_page_reads_the_wiki_api_and_caches(tmp_path):
    calls = []

    def get(url, timeout=30, accept=""):
        calls.append(url)
        return json.dumps({"parse": {"text": {"*": PAGE_HTML}}}).encode()

    text = w.fetch_page(FANDOM, tmp_path, get)
    assert "5 string" in text and calls[0].startswith("https://minecraft.fandom.com/api.php?action=parse&page=Lead")
    assert w.fetch_page(FANDOM, tmp_path, get) == text and len(calls) == 1


def test_web_search_keeps_allowed_pages_and_falls_back_to_the_wiki_search():
    class Hits:
        def text(self, query, max_results=8):
            return [{"href": "https://minecraft.wiki/w/Lead", "title": "x"}, {"href": FANDOM + "#Obtaining", "title": "Lead"},
                    {"href": "https://example.com/lead", "title": "y"}]

    assert w.web_search("lead", ddgs=Hits(), delay=0) == [{"url": FANDOM, "title": "Lead"}]

    class Broken:
        def text(self, query, max_results=8):
            raise RuntimeError("rate limited")

    def get(url, timeout=30, accept=""):
        assert "srsearch=lead+recipe" in url and "site" not in url
        return json.dumps({"query": {"search": [{"title": "Lead"}]}}).encode()

    assert w.web_search("lead recipe site:minecraft.fandom.com", ddgs=Broken(), get=get, delay=0)[0]["url"] == FANDOM


# ------------------------------------------------------------------ checks
def test_quotes_must_be_on_the_page_and_numbers_in_the_quotes():
    page = page_text()
    assert w.quote_found("Leads can be crafted from 5 string.", page)
    assert w.quote_found("leads can be crafted from 5 string", page)  # case and punctuation
    assert not w.quote_found("Leads are crafted from 4 string and a slimeball.", page)
    assert not w.quote_found("5 string", page), "too short to prove anything"
    assert w.unbacked_numbers("Craft 2 leads from 5 string.", ["Leads can be crafted from 5 string. Crafting a lead gives 2 leads."]) == set()
    assert w.unbacked_numbers("It takes 4 string.", ["Leads can be crafted from 5 string."]) == {"4"}


def _data() -> dict:
    items = [{"id": i, "name": n, "displayName": d} for i, n, d in (
        (1, "string", "String"), (2, "lead", "Lead"), (3, "oak_planks", "Oak Planks"), (4, "spruce_planks", "Spruce Planks"),
        (5, "cobblestone", "Cobblestone"), (6, "iron_ingot", "Iron Ingot"), (7, "redstone", "Redstone Dust"),
        (8, "piston", "Piston"))]
    piston = [{"inShape": [[p, p, p], [5, 6, 5], [5, 7, 5]], "result": {"id": 8, "count": 1}} for p in (3, 4)]
    return {"items": items, "recipes": {"2": [{"inShape": [[1, 1, None], [1, 1, None], [None, None, 1]],
                                               "result": {"id": 2, "count": 2}}], "8": piston},
            "entities": [{"name": "creeper", "displayName": "Creeper", "type": "hostile"},
                         {"name": "arrow", "displayName": "Arrow", "type": "projectile"}],
            "enchantments": [{"name": "mending", "displayName": "Mending"}]}


def test_crafting_answers_must_match_the_game_recipe():
    book = w.recipe_book(_data())
    assert book["piston"]["ingredients"] == {"any Planks": 3, "Cobblestone": 4, "Iron Ingot": 1, "Redstone Dust": 1}
    assert w.data_check("How do you craft a Lead?", "Use 4 string and 1 slimeball.", book)[0] == "contradicts"
    assert w.data_check("How do you craft a Lead?", "5 string in the grid makes 2 leads.", book)[0] == "ok"
    assert w.data_check("Piston recipe?", "Surround redstone with 4 iron ingots.", book)[0] == "contradicts"
    assert w.data_check("What is the crafting recipe for a piston?",
                        "3 planks on top, 4 cobblestone around an iron ingot, redstone dust below.", book)[0] == "ok"
    assert w.data_check("What does a lead do?", "It leashes mobs.", book)[0] == "n/a"


def test_topics_start_with_mechanics_then_mobs():
    topics = w.topics(_data(), random.Random(0))
    assert topics[:len(w.MECHANICS)] == list(w.MECHANICS)
    assert topics[len(w.MECHANICS)] == "Creeper" and "Arrow" not in topics and "Mending (enchantment)" in topics
    assert len({t.lower() for t in topics}) == len(topics)


def test_parse_json_finds_the_object_in_a_chatty_reply():
    assert w.parse_json('<think>hm {"no": 1}</think> Sure! {"verdict": "SUPPORTED", "reason": "a } b"} done') == {
        "verdict": "SUPPORTED", "reason": "a } b"}
    assert w.parse_json("no json") is None


# ------------------------------------------------------------------ the whole loop, with a fake LLM / web
class FakeLLM:
    """Writes one good pair, one with a made-up quote, one with an outdated recipe; judges by the evidence."""

    def __init__(self):
        self.calls = 0
        self.seconds = 0.0

    def chat(self, system, user, schema, max_tokens=700):
        self.calls += 1
        if system == w.PROPOSE_SYSTEM:
            assert "SOURCE TEXT" in user and "5 string" in user
            return {"pairs": [
                {"question": "How do you craft a lead in Minecraft?", "answer": "Craft it from 5 string; you get 2 leads.",
                 "evidence": ["Leads can be crafted from 5 string.", "Crafting a lead gives 2 leads."], "edition": "both"},
                {"question": "Can you leash a creeper?", "answer": "No, hostile mobs cannot be leashed.",
                 "evidence": ["Creepers can be leashed with gold leads."], "edition": "java"},
                {"question": "What is the lead recipe?", "answer": "Use 4 string and a slimeball.",
                 "evidence": ["Leads can be crafted from 5 string."], "edition": "java"},
            ]}
        assert system == w.JUDGE_SYSTEM
        return {"verdict": "SUPPORTED" if "5 string" in user and "4 string" not in user else "NOT_FOUND", "reason": "r"}


def test_collect_keeps_checked_rows_rejects_the_rest_and_resumes(tmp_path):
    folder = tmp_path / "Minecraftism-dataset"
    collector = w.Collector(folder, "Qwen3-4B-Instruct-2507")
    searched = []

    def search(query):
        searched.append(query)
        return [{"url": FANDOM, "title": "Lead"}, {"url": "https://www.minecraft.net/en-us/article/java-1-21-6", "title": "x"}]

    def fetch(url):
        return page_text() if "fandom" in url else "The Slime Ball in the Lead recipe has been replaced: 5 string now."

    uploads = []
    w.collect({"pairs_per_page": 3, "model": "Qwen3-4B-Instruct-2507", "max_topics": 1}, FakeLLM(), collector,
              w.recipe_book(_data()), ["Lead"], search, fetch, deadline=1e18, on_upload=uploads.append)
    assert searched == []  # the wiki page is opened directly, and the game data confirmed it: no search at all
    assert [r["question"] for r in collector.rows] == ["How do you craft a lead in Minecraft?"]
    row = collector.rows[0]
    assert row["trust"] == "high" and row["checks"]["game_data"] == "ok" and row["checks"]["second_look"] == "SUPPORTED"
    assert row["messages"][-1] == {"role": "assistant", "content": row["answer"]}
    assert row["sources"][0] == {"url": FANDOM, "license": "CC BY-SA 3.0"}
    assert len(row["sources"]) == 1 and "independent_page" not in row["checks"]  # the game data confirmed it: no 2nd search
    assert row["kind"] == "mcqa" and row["source"] == "web-verified"
    whys = sorted(r["why"] for r in collector.rejected)
    assert len(whys) == 2 and whys[0].startswith("evidence not on the page") and "numbers not in the evidence" in whys[1]
    assert collector.done == ["Lead"]

    saved = [json.loads(line) for line in (folder / "data" / "train-01.jsonl").read_text().splitlines()]
    assert saved == collector.rows
    card = (folder / "README.md").read_text()
    assert "license: cc-by-sa-4.0" in card and "**1 rows** - 1 `high`" in card and "minecraft.wiki (CC BY-NC-SA) is never used" in card
    assert "Made with ❤️ by [codero-sus]" in card

    again = w.Collector(folder, "x")  # a new session picks up the rows and skips the topic
    assert again.done == ["Lead"] and again.seen("how do you craft a LEAD in minecraft") and len(again.rejected) == 2
    w.collect({"model": "x"}, FakeLLM(), again, {}, ["Lead"], search, fetch, deadline=1e18)
    assert len(again.rows) == 1


def test_collect_stops_at_the_deadline_and_keep_high_drops_medium_rows(tmp_path):
    collector = w.Collector(tmp_path / "d", "m")
    w.collect({}, FakeLLM(), collector, {}, ["Lead"], lambda q: pytest.fail("no search after the deadline"),
              lambda u: "", deadline=0)
    assert collector.rows == [] and collector.done == []
    llm = FakeLLM()
    w.collect({"keep": "high", "corroborate": False}, llm, collector, {}, ["Lead"],
              lambda q: [{"url": FANDOM, "title": "Lead"}], lambda u: page_text(), deadline=1e18)
    assert collector.rows == [] and any(r["why"] == "only medium trust" for r in collector.rejected)


# ------------------------------------------------------------------ Discord
def test_discord_pings_the_role_only_for_important_messages():
    sent = []
    discord = w.Discord("https://discord.com/api/webhooks/1/x", "<@&123>", post=lambda url, body: sent.append(json.loads(body)))
    discord.send("hourly", "Progress")
    discord.send("uploaded", "Uploaded", "good", important=True)
    discord.send("boom", "Notebook error", "bad")
    assert "content" not in sent[0] and sent[0]["allowed_mentions"] == {"parse": []}
    assert sent[1]["content"] == "<@&123>" and sent[1]["allowed_mentions"] == {"parse": [], "roles": ["123"]}
    assert sent[2]["content"] == "<@&123>"
    assert not w.Discord("").send("x") and not w.Discord("not-a-url").send("x")


# ------------------------------------------------------------------ the notebook
def test_notebook_is_built_from_the_module_and_runs_on_cpu():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_webdata_notebook.py"), "--check"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["metadata"]["kaggle"]["isGpuEnabled"] is False and nb["metadata"]["kaggle"]["accelerator"] == "none"
    assert nb["metadata"]["kaggle"]["isInternetEnabled"] is True
    sources = ["".join(c["source"]) for c in nb["cells"]]
    module = (ROOT / "pymc_bot" / "llm_webdata.py").read_text(encoding="utf-8")
    assert any(s.endswith(module.rstrip("\n")) for s in sources), "the collector cell is the module, verbatim"
    settings = sources[1]
    assert 'DATASET_REPO = "Minecraftism-dataset"' in settings and 'HF_SECRET = "HF_TOKEN"' in settings
    assert 'DISCORD_SECRET = "discord_webhook_inform_me1"' in settings and 'PING_ROLE_SECRET = "ping_role"' in settings
    assert "MAX_HOURS = 11.75" in settings and 'DEVICE = "auto"' in settings  # free CPU by default, the GPU if attached
    assert '"/kaggle/working", "/kaggle/tmp"' in sources[2]
    assert "Made with ❤️ by codero-sus" in sources[0] and "minecraft.wiki is CC BY-NC-SA" in sources[0]
    for code in sources[1:]:
        compile(code, "cell", "exec")
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert text == json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def test_notebook_cells_run_in_order_without_kaggle(tmp_path, monkeypatch):
    """Settings + collector + run cell work together (no network: main() is given a dry run)."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"]]
    namespace: dict = {"__name__": "__main__"}
    monkeypatch.setitem(sys.modules, "ipykernel", sys)  # like a notebook kernel: the module's CLI block stays off
    exec(compile(sources[1], "settings", "exec"), namespace)
    exec(compile(sources[3], "collector", "exec"), namespace)
    seen = {}
    result = {"rows": 2, "high": 1, "medium": 1, "rejected": 3, "topics": 1, "link": "https://hf.co/datasets/me/x",
              "samples": [{"question": "Lead?", "answer": "5 string.", "trust": "high"}]}
    namespace["main"] = lambda cfg: seen.update(cfg) or result
    namespace.update(WORK=str(tmp_path), TMP=str(tmp_path / "tmp"))
    exec(compile(sources[4], "run", "exec"), namespace)
    assert seen["dataset_repo"] == "Minecraftism-dataset" and seen["private"] is True and seen["model"] == "auto"
    assert seen["parallel"] == 0 and seen["min_tokens_per_s"] == 0 and seen["speculate"] == "auto"  # 0 = auto
    assert seen["device"] == "auto" and seen["pages_per_topic"] == 0 and seen["pairs_per_page"] == 0
    assert seen["wiki_pages"] is True and seen["sessions"] == 1 and seen["session"] == 1
    assert seen["processes"] is True
    assert seen["max_hours"] == 11.75 and seen["ping_role_secret"] == "ping_role"


FAKE_SERVER = r'''#!PYTHON
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
if args == ["--version"]:
    print("version: 1 (fake)"); sys.exit(0)
port = int(args[args.index("--port") + 1])
assert "--jinja" in args and args[args.index("-t") + 1] == "2"
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        body = json.dumps(obj).encode(); self.send_response(200)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body)))
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send({"status": "ok"})
    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        schema = req["response_format"]["json_schema"]["schema"]
        thinking = req.get("chat_template_kwargs")
        reply = {"verdict": "SUPPORTED", "reason": json.dumps([sorted(schema["properties"]), thinking])}
        self._send({"choices": [{"message": {"content": "<think></think>" + json.dumps(reply)}}]})
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def test_llama_server_starts_answers_with_the_schema_and_stops(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text(FAKE_SERVER.replace("PYTHON", sys.executable))
    binary.chmod(0o755)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    assert w._runs(binary)
    assert w.llama_server_binary(tmp_path, fetch_json=lambda url: pytest.fail("found locally")) == binary
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = w.LlamaServer(binary, model, threads=2, port=port, hybrid=True).start(wait=60)
    try:
        verdict, reason = w.judge(server, "q?", "a", "evidence")
        assert verdict == "SUPPORTED" and json.loads(reason) == [["reason", "verdict"], {"enable_thinking": False}]
        assert server.calls == 1
    finally:
        server.stop()
    assert server.process.poll() is not None


def test_model_cards_carry_the_polyform_required_notice():
    import inspect

    from pymc_bot import llm_train
    source = inspect.getsource(llm_train.model_card)
    assert "\nRequired Notice: Copyright codero-sus" in source
    assert "polyformproject.org/licenses/noncommercial/1.0.0" in source


class FakeHub:
    def __init__(self, public: bool = True, role: str = "write", info_fails: bool = False):
        self.public, self.role, self.info_fails = public, role, info_fails
        self.calls: list = []

    def whoami(self):
        return {"name": "me", "auth": {"accessToken": {"role": self.role}}}

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create", repo_id, kwargs["private"]))

    def repo_info(self, repo_id, **kwargs):
        if self.info_fails:
            raise OSError("hub down")
        return types.SimpleNamespace(private=not self.public)

    def update_repo_settings(self, repo_id, **kwargs):
        self.calls.append(("settings", repo_id, kwargs))
        self.public = not kwargs["private"]

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs["repo_id"], tuple(kwargs["allow_patterns"]),
                           tuple(kwargs.get("delete_patterns") or ())))


def test_private_on_switches_a_public_repo_to_private_before_every_upload(tmp_path):
    hub = FakeHub(public=True)
    assert w.hub_repo_id(hub, "Minecraftism-dataset") == "me/Minecraftism-dataset"
    assert w.hub_make_private(hub, "me/Minecraftism-dataset", True) is True
    assert hub.calls == [("create", "me/Minecraftism-dataset", True),
                         ("settings", "me/Minecraftism-dataset", {"repo_type": "dataset", "private": True})]
    hub.public = True  # someone made it public meanwhile: the next upload makes it private again
    link = w.hub_upload(hub, "me/Minecraftism-dataset", tmp_path, True, "30 rows")
    assert link == "https://huggingface.co/datasets/me/Minecraftism-dataset" and not hub.public
    assert hub.calls[-1][0] == "upload" and "data/train-01.jsonl" in hub.calls[-1][2] and "data/*" not in hub.calls[-1][2]
    assert "data/train.jsonl" in hub.calls[-1][3]  # session 01 removes the file from before the sessions
    w.hub_upload(hub, "me/Minecraftism-dataset", tmp_path, True, "x", ["data/train-02.jsonl", "meta/progress-02.json"])
    assert hub.calls[-1][2] == ("README.md", "data/train-02.jsonl", "meta/progress-02.json") and hub.calls[-1][3] == ()
    unsure = FakeHub(public=False, info_fails=True)  # visibility unknown: made private to be sure
    assert w.hub_make_private(unsure, "me/x", True) and unsure.calls[-1][0] == "settings"
    public = FakeHub(public=True)  # PRIVATE = False leaves it public
    assert w.hub_make_private(public, "me/x", False) is False and [c[0] for c in public.calls] == ["create"]


def test_a_read_only_token_or_a_bad_setting_stops_right_away(tmp_path):
    with pytest.raises(SystemExit, match="write"):
        w.hub_repo_id(FakeHub(role="read"), "Minecraftism-dataset")
    with pytest.raises(SystemExit, match="qwen3-4b"):
        w.main({"model": "llama-70b", "upload": False, "work": str(tmp_path), "tmp": str(tmp_path / "t")})
    with pytest.raises(SystemExit, match="medium"):
        w.main({"keep": "all", "upload": False, "work": str(tmp_path), "tmp": str(tmp_path / "t")})


def test_pages_are_fetched_one_topic_ahead_and_a_stop_keeps_the_rows(tmp_path):
    import threading

    collector = w.Collector(tmp_path / "d", "m")
    threads = []

    def search(query):
        return [{"url": FANDOM, "title": "Lead"}]

    def fetch(url):
        threads.append((url.rsplit("/", 1)[-1], threading.current_thread().name))
        return page_text() + f"\nSee {url}."  # one page per topic

    class Stopping(FakeLLM):
        def chat(self, system, user, schema, max_tokens=700):
            if self.calls >= 2:  # the second topic's first LLM call: the Kaggle stop button
                raise KeyboardInterrupt
            return super().chat(system, user, schema, max_tokens)

    with pytest.raises(KeyboardInterrupt):
        w.collect({"corroborate": False}, Stopping(), collector, w.recipe_book(_data()), ["Lead", "Lead use", "Lead knot"],
                  search, fetch, deadline=1e18)
    assert all(name.startswith("prefetch") for _q, name in threads)
    assert [q for q, _n in threads][:2] == ["Lead", "Lead_use"]  # fetched in the background, one topic ahead
    assert len(collector.rows) == 1  # saved on the way out
    saved = (tmp_path / "d" / "data" / "train-01.jsonl").read_text().splitlines()
    assert len(saved) == 1 and (tmp_path / "d" / "data" / "high" / "train-01.jsonl").read_text().count("\n") == 1
    card = (tmp_path / "d" / "README.md").read_text()
    assert "config_name: high" in card and "path: data/high/train-*.jsonl" in card and 'load_dataset(' in card


def test_parallel_topics_share_the_llm_keep_the_order_and_never_duplicate_a_question(tmp_path):
    import threading

    collector = w.Collector(tmp_path / "d", "m")
    names, busy, most = set(), [0], [0]
    lock = threading.Lock()

    class Slow(FakeLLM):
        def chat(self, system, user, schema, max_tokens=700):
            with lock:
                busy[0] += 1
                most[0] = max(most[0], busy[0])
                names.add(threading.current_thread().name.split("_")[0])
            time.sleep(0.05)
            try:
                return super().chat(system, user, schema, max_tokens)
            finally:
                with lock:
                    busy[0] -= 1

    def search(query):
        if "site:minecraft.fandom.com" in query:
            return [{"url": FANDOM, "title": "Lead"}]
        return [{"url": "https://www.minecraft.net/en-us/article/java-1-21-6", "title": "x"}]

    def fetch(url):
        return page_text() if "fandom" in url else "The Slime Ball in the Lead recipe has been replaced: 5 string now."

    topics = ["Lead", "Lead use", "Lead knot", "Lead fence"]
    w.collect({"parallel": 3}, Slow(), collector, {}, topics, search, fetch, deadline=1e18)
    assert most[0] >= 2 and names == {"topic"}  # several topics talk to the LLM at once
    assert collector.done == topics  # finished in topic order
    assert [r["question"] for r in collector.rows] == ["How do you craft a lead in Minecraft?"]  # once, not 4 times
    row = collector.rows[0]  # no game data here, so an independent page was asked
    assert row["checks"]["independent_page"] == "SUPPORTED" and row["trust"] == "high"
    assert row["sources"][1]["role"] == "confirms" and "minecraft.net" in row["sources"][1]["url"]


def test_llama_server_runs_parallel_slots_and_reports_its_speed():
    server = w.LlamaServer(Path("llama-server"), Path("m.gguf"), threads=4, ctx=4096, parallel=4)
    assert server.parallel == 4 and server.ctx * server.parallel == 16384
    assert w.llm_speed(server) == ""
    server.tokens.update(prompt=900, prompt_ms=20000, cached=100, out=300, out_ms=40000)
    assert w.llm_speed(server) == "LLM reads 45 tok/s (10% from cache), writes 7.5 tok/s per request x 4 at once"
    assert w.VERDICT_SCHEMA["properties"]["reason"]["maxLength"] <= 200  # short judge replies


class Timed:
    """A started server for fast_llm: writes ``speed`` tok/s in the speed test."""

    def __init__(self, key, spec, speed, stopped):
        self.key, self.spec, self.speed, self.stopped = key, spec, speed, stopped
        self.parallel, self.calls = 4, 0
        self.tokens = {"prompt": 0, "prompt_ms": 0.0, "cached": 0, "out": 0, "out_ms": 0.0}

    def chat(self, system, user, schema, max_tokens=700):
        if system == w.JUDGE_SYSTEM:
            return {"verdict": "SUPPORTED", "reason": "r"}
        assert "5 string" in user and "SOURCE TEXT" in user
        self.tokens["out"] += 100
        self.tokens["out_ms"] += 100 / self.speed * 1000 * self.parallel  # 4 side by side: `speed` in total
        return {"pairs": [{"question": "How do you craft a lead?", "answer": "From 5 string.",
                           "evidence": ["Leads are crafted from 5 string"], "edition": "both"}]}

    def stop(self):
        self.stopped.append((self.key, self.spec))


def test_the_speed_test_keeps_the_biggest_model_that_writes_20_tokens_per_second():
    speeds = {("qwen3-30b-a3b", True): 11, ("qwen3-30b-a3b", False): 8, ("qwen3-4b", True): 14,
              ("qwen3-4b", False): 9, ("qwen3-1.7b", True): 16, ("qwen3-1.7b", False): 23}
    stopped = []

    def start(key, spec):
        return Timed(key, spec, speeds[(key, spec)], stopped)

    server, key, speed, spec = w.fast_llm(list(w.MODELS), start, 20)
    assert (key, spec) == ("qwen3-1.7b", False) and round(speed) == 23 and server.speed == 23
    assert ("qwen3-1.7b", False) not in stopped and len(stopped) == 5  # every slower one was stopped

    stopped.clear()  # fast enough at once: no other test, no smaller model
    server, key, speed, spec = w.fast_llm(list(w.MODELS), start, 10)
    assert (key, spec) == ("qwen3-30b-a3b", True) and stopped == []
    server, key, speed, spec = w.fast_llm(["qwen3-30b-a3b"], start, 20)  # MODEL set by you: kept, the faster way
    assert (key, spec) == ("qwen3-30b-a3b", True) and round(speed) == 11
    server, key, speed, spec = w.fast_llm(list(w.MODELS), start, 0, "off")  # 0 = always the biggest
    assert (key, spec) == ("qwen3-30b-a3b", False)

    def no_speculation(key, spec):  # an older llama.cpp without the flag, or broken replies with it
        if spec:
            raise RuntimeError("llama-server stopped: unknown argument --spec-type")
        return start(key, spec)

    assert w.fast_llm(list(w.MODELS), no_speculation, 20)[1:] == ("qwen3-1.7b", 23, False)

    class Broken(Timed):
        def chat(self, system, user, schema, max_tokens=700):
            return {} if system == w.PROPOSE_SYSTEM else super().chat(system, user, schema, max_tokens)

    assert w.benchmark(Broken("x", True, 50, [])) == 0.0  # fast but useless replies count as 0


class PidSearch:
    """Picklable search for the worker processes: notes which process ran it."""

    def __init__(self, log):
        self.log = log

    def __call__(self, query):
        with open(self.log, "a") as handle:
            handle.write(f"{os.getpid()}\n")
        return [{"url": FANDOM, "title": "Lead"}] if "site:" in query else []


class PageText:
    """Picklable page fetcher for the worker processes: notes which process ran it; one page per topic."""

    def __init__(self, log=None):
        self.log = log

    def __call__(self, url):
        if self.log:
            with open(self.log, "a") as handle:
                handle.write(f"{os.getpid()}\n")
        return page_text() + f"\nSee {url}."


def test_the_web_part_runs_in_worker_processes_and_the_rows_stay_right(tmp_path):
    pool = w.process_pool(2)
    assert pool is not None
    try:
        collector = w.Collector(tmp_path / "d", "m")
        log = tmp_path / "pids"
        topics = ["Lead", "Lead use", "Lead knot"]
        w.collect({"parallel": 2, "corroborate": False}, FakeLLM(), collector, w.recipe_book(_data()), topics,
                  PidSearch(tmp_path / "searches"), PageText(log), deadline=1e18, processes=pool)
        pids = set(log.read_text().split())
        assert len(pids) >= 1 and str(os.getpid()) not in pids  # every page was read in a worker process
        assert collector.done == topics and [r["question"] for r in collector.rows] == [
            "How do you craft a lead in Minecraft?"]
        assert pool.submit(w._ready).result() != os.getpid()  # the shared pool is still usable afterwards
    finally:
        pool.shutdown()
    assert pickle_roundtrip(w.WebSearch(5)).results == 5 and pickle_roundtrip(w.PageFetcher(tmp_path)).cache == tmp_path


def pickle_roundtrip(thing):
    import pickle

    return pickle.loads(pickle.dumps(thing))


# ------------------------------------------------------------------ GPU (Accelerator GPU T4 x2)
def test_gpu_info_reads_nvidia_smi_and_picks_the_better_quant_when_it_fits():
    def smi(command, **kwargs):
        assert command[0] == "nvidia-smi"
        return types.SimpleNamespace(returncode=0, stdout="Tesla T4, 15360, 7.5\nTesla T4, 15360, 7.5\n")

    gpus = w.gpu_info(smi)
    assert gpus == [{"name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}] * 2

    def missing(command, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    assert w.gpu_info(missing) == []
    assert w.gpu_quant("qwen3-30b-a3b", 30) == "Q5_K_M"  # 2 x T4: the whole 30B in a better quant
    assert w.gpu_quant("qwen3-30b-a3b", 16) == "Q4_K_M"  # one 16 GB GPU: llama.cpp keeps the rest in RAM
    assert w.gpu_quant("qwen3-4b", 16) == "Q8_0"


FAKE_CUDA_SERVER = r'''#!/bin/sh
if [ "$1" = "--version" ]; then echo "version: 1 (fake)"; exit 0; fi
if [ "$1" = "--list-devices" ]; then
  if [ -f "$(dirname "$0")/libcudart.so.12" ]; then printf "Available devices:\n  CUDA0: Tesla T4 (15095 MiB, 14982 MiB free)\n  CUDA1: Tesla T4 (15095 MiB, 14982 MiB free)\n"; fi
  exit 0
fi
exit 1
'''


def test_the_cuda_build_comes_with_its_runtime_and_must_see_the_gpu(tmp_path):
    import tarfile

    def pack(name, files):
        path = tmp_path / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(path, "w:gz") as tar:
            for member, text in files.items():
                file = tmp_path / "src" / "x"
                file.write_text(text)
                file.chmod(0o755)
                tar.add(file, member)
        return path

    built = {"llama-b9-bin-ubuntu-cuda-12.8-x64.tar.gz": pack("a.tgz", {"llama-b9/llama-server": FAKE_CUDA_SERVER}),
             "cudart-llama-b9-bin-ubuntu-cuda-12.8-x64.tar.gz": pack("b.tgz", {
                 "cudart-llama-b9-bin-ubuntu-cuda-12.8-x64/libcudart.so.12": "lib"}),
             "llama-b9-bin-ubuntu-x64.tar.gz": pack("c.tgz", {"llama-b9/llama-server": "#!/bin/sh\nexit 0\n"})}
    release = [{"tag_name": "b9", "assets": [{"name": n, "browser_download_url": n} for n in built]}]
    fetched = []

    def download(url, target):
        fetched.append(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(built[url].read_bytes())
        return target

    binary = w.llama_server_binary(tmp_path / "tools", lambda url: release, download, cuda_arch="75")
    assert binary.parent.name == "llama-b9" and (binary.parent / "libcudart.so.12").is_file()
    assert w._cuda_devices(binary) == 2 and "cuda" in binary.parts  # kept apart from the CPU build
    assert sorted(fetched) == sorted(n for n in built if "cuda" in n)
    again = w.llama_server_binary(tmp_path / "tools", lambda url: pytest.fail("found locally"), download, cuda_arch="75")
    assert again == binary


def test_main_uses_the_gpu_and_falls_back_to_a_compiled_build_then_the_cpu(tmp_path, monkeypatch):
    calls, started = [], []

    def binary(tools, fetch_json=None, download=None, cuda_arch="", prebuilt=True):
        calls.append((cuda_arch, prebuilt))
        return Path(f"/bin/{'cuda-' + ('pre' if prebuilt else 'built') if cuda_arch else 'cpu'}/llama-server")

    class Server(Timed):
        def __init__(self, binary, model, ctx, hybrid, parallel, speculate):
            super().__init__(model.name, speculate, 50, [])
            self.binary, self.parallel, self.seconds = binary, parallel, 0.0

        def start(self):
            started.append((self.binary.parent.name, self.key, self.parallel))
            if self.binary.parent.name in broken:
                raise RuntimeError("llama-server stopped: the provided PTX was compiled with an unsupported toolchain")
            return self

    monkeypatch.setattr(w, "gpu_info", lambda: [{"name": "Tesla T4", "vram_gb": 15.0, "arch": "75"}] * 2)
    monkeypatch.setattr(w, "llama_server_binary", binary)
    monkeypatch.setattr(w, "LlamaServer", Server)
    monkeypatch.setattr(w, "model_file", lambda spec, folder, token="", quant="": Path(quant or spec["quant"]))
    monkeypatch.setattr(w, "available_ram_gb", lambda: 29.0)
    monkeypatch.setattr(w, "collect", lambda cfg, llm, *a, **k: calls.append(("collect", cfg["pairs_per_page"],
                                                                               cfg["pages_per_topic"], llm.binary)))
    monkeypatch.setattr(w, "load_minecraft_data", lambda version=None: _data())
    monkeypatch.setattr(w, "wiki_titles", lambda cache=None: ["Tutorials/Farming", "Lead"])
    cfg = {"upload": False, "work": str(tmp_path), "tmp": str(tmp_path / "t"), "processes": False, "speculate": "off"}

    broken = {"cuda-pre"}  # the prebuilt CUDA build fails on this driver: compiled for the T4, still the 30B
    w.main(cfg)
    assert calls[:2] == [("75", True), ("75", False)]
    assert calls[-1] == ("collect", 5, 1, Path("/bin/cuda-built/llama-server"))  # 5 pairs per part, whole page
    assert started == [("cuda-pre", "Q5_K_M", 8), ("cuda-built", "Q5_K_M", 8)]  # 30B Q5_K_M, 8 at once, no smaller

    calls.clear(), started.clear()
    broken = {"cuda-pre", "cuda-built"}  # no GPU build works: the CPU, with the CPU settings
    w.main(cfg)
    assert calls[-1] == ("collect", 5, 1, Path("/bin/cpu/llama-server"))
    assert started[-1] == ("cpu", "Q4_K_M", 4)


# ------------------------------------------------------------------ as many rows as possible
def test_a_page_is_read_whole_in_parts_without_the_changelog():
    lines = ["Intro about the villager."]
    for heading in ("Trading", "Breeding", "History", "Gallery", "Trivia"):
        lines.append(f"## {heading}")
        lines += [f"{heading} fact number {i}: villagers do things with emeralds and beds." for i in range(40)]
    parts = w.page_sections("\n".join(lines), size=1200)
    text = "\n".join(parts)
    assert len(parts) >= 6 and all(len(p) <= 1300 for p in parts)
    assert "Trading fact number 39" in text and "Breeding fact number 0" in text and "Trivia fact number 39" in text
    assert "History fact" not in text and "Gallery fact" not in text and "[Trading]" in text
    assert w.page_sections("short", 2800) == [] and len(w.page_sections("\n".join(lines), 1200, limit=2)) == 2
    html = "<p>Intro text here.</p><h2>Obtaining<span class='mw-editsection'>[edit]</span></h2><p>Craft it.</p>"
    assert w.html_to_text(html) == "Intro text here.\n## Obtaining\nCraft it."


def test_the_wiki_page_list_has_every_article_about_the_game(tmp_path):
    pages = {None: (["Lead", "Java Edition 1.21", "Pig", "Villager/DV"], "M"),
             "M": (["Tutorials/Farming", "24w14a", "Minecraft Dungeons:Arch-Illager", "Zombie (disambiguation)",
                    "Redstone circuits", "Pocket Edition Alpha 0.1.0", "Pink Wool"], None)}
    asked = []

    def get(url, timeout=30, accept="*/*"):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        assert query["list"] == ["allpages"] and query["apfilterredir"] == ["nonredirects"]
        key = (query.get("apcontinue") or [None])[0]
        asked.append(key)
        titles, more = pages[key]
        body = {"query": {"allpages": [{"title": t} for t in titles]}}
        if more:
            body["continue"] = {"apcontinue": more}
        return json.dumps(body).encode()

    titles = w.wiki_titles(get, tmp_path / "wiki_titles.json")
    assert asked == [None, "M"]
    assert titles == ["Lead", "Pig", "Tutorials/Farming", "Redstone circuits", "Pink Wool"]
    assert w.wiki_titles(lambda *a, **k: pytest.fail("cached"), tmp_path / "wiki_titles.json") == titles
    data = {"entities": [{"name": "pig", "displayName": "Pig", "type": "passive"}], "items": [], "blocks": []}
    topics = w.all_topics(data, tmp_path, {"wiki_pages": True})
    assert topics[0] == w.MECHANICS[0] and topics.count("Pig") == 1 and topics[-1] == "Pink Wool"
    assert "Redstone circuits" in topics and len(topics) == len(set(t.lower() for t in topics))


def test_sessions_at_the_same_time_write_their_own_files_and_share_what_is_done(tmp_path):
    folder = tmp_path / "Minecraftism-dataset"
    (folder / "data").mkdir(parents=True)
    (folder / "meta").mkdir()
    row = w.make_row("Lead", {"question": "How do you craft a lead?", "answer": "From 5 string.",
                              "evidence": ["5 string"], "edition": "both"}, FANDOM, "CC BY-SA 3.0", {}, "high", "", "m")
    (folder / "data" / "train.jsonl").write_text(json.dumps(row) + "\n")  # a dataset from before the sessions
    (folder / "meta" / "progress.json").write_text(json.dumps({"done_topics": ["Lead"], "done_pages": ["abc"]}))
    one = w.Collector(folder, "m", 1)
    assert len(one.rows) == 1 and one.done == ["Lead"] and not (folder / "data" / "train.jsonl").exists()
    one.save()
    assert (folder / "data" / "train-01.jsonl").is_file() and (folder / "meta" / "progress-01.json").is_file()

    two = w.Collector(folder, "m", 2)  # session 2 starts: sees session 1's rows, topics and pages, owns none of them
    assert two.rows == [] and two.seen("how do you craft a LEAD") and two.finished() == {"Lead"} and "abc" in two.pages
    assert two.stats()["rows"] == 1 and two.stats()["high"] == 1
    two.add({**row, "question": "What do leads attach to?"})
    two.save()
    assert len((folder / "data" / "train-01.jsonl").read_text().splitlines()) == 1  # untouched
    assert len((folder / "data" / "train-02.jsonl").read_text().splitlines()) == 1
    assert two.own_files() == ["data/train-02.jsonl", "data/high/train-02.jsonl", "meta/rejected-02.jsonl",
                               "meta/progress-02.json"]
    card = (folder / "README.md").read_text()
    assert "**2 rows**" in card and "path: data/train-*.jsonl" in card


def test_llama_server_speculation_flags():
    plain = w.LlamaServer(Path("llama-server"), Path("m.gguf"), threads=4, parallel=4).command()
    assert "--spec-type" not in plain and plain[plain.index("-np") + 1] == "4"
    fast = w.LlamaServer(Path("llama-server"), Path("m.gguf"), threads=4, parallel=4, speculate=True).command()
    assert fast[fast.index("--spec-type") + 1] == "ngram-simple" and "--spec-draft-n-max" in fast


def test_auto_picks_the_biggest_model_that_fits_in_the_ram(monkeypatch):
    assert w.pick_model("auto", 29.5)[0] == "qwen3-30b-a3b"  # Kaggle's 30 GB: the 30B mixture of experts
    assert w.pick_model("auto", 13)[0] == "qwen3-4b" and w.pick_model("auto", 3)[0] == "qwen3-1.7b"
    assert w.pick_model("auto", 1)[0] == "qwen3-1.7b"
    key, why = w.pick_model("qwen3-30b-a3b", 8)
    assert key == "qwen3-30b-a3b" and "WARNING" in why and "23 GB" in why  # your choice, with a warning
    assert "WARNING" not in w.pick_model("qwen3-4b", 30)[1]
    assert list(w.MODELS) == sorted(w.MODELS, key=lambda k: -w.MODELS[k]["ram_gb"])  # biggest first
    assert all(spec["license"] == "Apache-2.0" for spec in w.MODELS.values())
    assert w.MODELS["qwen3-30b-a3b"]["repo"] == "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
    assert w.available_ram_gb() > 0
