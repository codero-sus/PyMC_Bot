"""The Minecraftism dataset collector (pymc_bot/llm_webdata.py) and its Kaggle notebook."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import types
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
    assert searched[0] == "Lead site:minecraft.fandom.com"
    assert [r["question"] for r in collector.rows] == ["How do you craft a lead in Minecraft?"]
    row = collector.rows[0]
    assert row["trust"] == "high" and row["checks"]["game_data"] == "ok" and row["checks"]["second_look"] == "SUPPORTED"
    assert row["messages"][-1] == {"role": "assistant", "content": row["answer"]}
    assert row["sources"][0] == {"url": FANDOM, "license": "CC BY-SA 3.0"}
    assert row["sources"][1]["role"] == "confirms" and "minecraft.net" in row["sources"][1]["url"]
    assert row["kind"] == "mcqa" and row["source"] == "web-verified"
    whys = sorted(r["why"] for r in collector.rejected)
    assert len(whys) == 2 and whys[0].startswith("evidence not on the page") and "numbers not in the evidence" in whys[1]
    assert collector.done == ["Lead"]

    saved = [json.loads(line) for line in (folder / "data" / "train.jsonl").read_text().splitlines()]
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
    assert "MAX_HOURS = 11.75" in settings and "cuda" not in "".join(sources[3:]).lower()
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
    assert seen["dataset_repo"] == "Minecraftism-dataset" and seen["private"] is True and seen["model"] == "qwen3-4b"
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
        self.calls.append(("upload", kwargs["repo_id"], tuple(kwargs["allow_patterns"])))


def test_private_on_switches_a_public_repo_to_private_before_every_upload(tmp_path):
    hub = FakeHub(public=True)
    assert w.hub_repo_id(hub, "Minecraftism-dataset") == "me/Minecraftism-dataset"
    assert w.hub_make_private(hub, "me/Minecraftism-dataset", True) is True
    assert hub.calls == [("create", "me/Minecraftism-dataset", True),
                         ("settings", "me/Minecraftism-dataset", {"repo_type": "dataset", "private": True})]
    hub.public = True  # someone made it public meanwhile: the next upload makes it private again
    link = w.hub_upload(hub, "me/Minecraftism-dataset", tmp_path, True, "30 rows")
    assert link == "https://huggingface.co/datasets/me/Minecraftism-dataset" and not hub.public
    assert hub.calls[-1][0] == "upload" and "data/*" in hub.calls[-1][2]
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
        threads.append((query.split(" site:")[0], threading.current_thread().name))
        return [{"url": FANDOM, "title": "Lead"}]

    class Stopping(FakeLLM):
        def chat(self, system, user, schema, max_tokens=700):
            if self.calls >= 2:  # the second topic's first LLM call: the Kaggle stop button
                raise KeyboardInterrupt
            return super().chat(system, user, schema, max_tokens)

    with pytest.raises(KeyboardInterrupt):
        w.collect({"corroborate": False}, Stopping(), collector, w.recipe_book(_data()), ["Lead", "Lead use", "Lead knot"],
                  search, lambda u: page_text(), deadline=1e18)
    assert threads[0] == ("Lead", threads[0][1]) and all(name.startswith("prefetch") for _q, name in threads)
    assert [q for q, _n in threads][:2] == ["Lead", "Lead use"]  # fetched in the background, one topic ahead
    assert len(collector.rows) == 1  # saved on the way out
    saved = (tmp_path / "d" / "data" / "train.jsonl").read_text().splitlines()
    assert len(saved) == 1 and (tmp_path / "d" / "data" / "high" / "train.jsonl").read_text().count("\n") == 1
    card = (tmp_path / "d" / "README.md").read_text()
    assert "config_name: high" in card and "path: data/high/train.jsonl" in card and 'load_dataset(' in card
