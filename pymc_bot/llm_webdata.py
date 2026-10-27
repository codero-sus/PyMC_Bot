"""Minecraftism trusted dataset: web search + a small CPU LLM -> checked Minecraft Q&A -> Hugging Face.

Runs on Kaggle **without a GPU** (Accelerator: None, Internet on): llama.cpp's CPU server runs a small Qwen3
GGUF in RAM, DuckDuckGo (``ddgs``) finds pages, and every answer has to pass these checks before it goes in:

1. **Licence first.** Answers are only written from pages whose licence allows commercial use (Minecraft Fandom
   wiki CC BY-SA 3.0, Wikipedia CC BY-SA 4.0). minecraft.wiki is CC BY-NC-SA (non-commercial), so it is never
   read. Official Mojang pages are only used to *confirm* an answer; nothing but their link is kept.
2. **Grounded.** The LLM writes question/answer pairs from the page text only, and copies the sentences that
   back each answer as evidence.
3. **Evidence is real.** Every evidence quote must actually be on the downloaded page.
4. **Numbers are backed.** Every number in the answer must be in the evidence (or the question).
5. **Second look.** A separate LLM pass sees only the question, the answer and the evidence, and must say
   SUPPORTED.
6. **Game data.** Crafting answers are compared with the game's own recipes (PrismarineJS/minecraft-data, MIT).
   An ingredient that is missing or a wrong count rejects the pair. This catches outdated pages (the lead
   recipe changed in 1.21.6).
7. **Corroboration.** The question is searched again and an independent page (another site, or an official
   one) is asked to confirm. If it contradicts the answer, the pair is rejected.

``trust`` on each row: ``high`` = confirmed by the game data or an independent page, ``medium`` = passed 1-5 from
one source. Rejected pairs and the reason go to ``meta/rejected.jsonl`` (for review, never trained on).

Rows have ``question`` / ``answer`` (what ``train_minecraft_llm.py --minecraft-qa-dataset`` streams) plus
``messages``, the sources and their licences, the evidence and the checks.

The Kaggle notebook ``pynb/Minecraftism_Dataset_Kaggle.ipynb`` carries this file verbatim (rebuild it with
``python scripts/build_webdata_notebook.py``). Only the standard library is imported at the top.
"""

import hashlib
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

UA = "PyMC-Bot-dataset/1.0 (https://github.com/codero-sus/PyMC_Bot)"
DATASET_NAME = "Minecraftism-dataset"
MINECRAFT_DATA_VERSION = "1.21.8"
MINECRAFT_DATA_URL = "https://raw.githubusercontent.com/PrismarineJS/minecraft-data/master/data/"
LLAMA_CPP_REPO = "ggml-org/llama.cpp"

#: Small Qwen3 GGUFs that run on Kaggle's 4 CPU cores (Apache-2.0: the dataset may be used commercially).
MODELS = {
    "qwen3-4b": {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "quant": "Q4_K_M", "name": "Qwen3-4B-Instruct-2507",
                 "hybrid": False, "license": "Apache-2.0"},
    "qwen3-1.7b": {"repo": "unsloth/Qwen3-1.7B-GGUF", "quant": "Q4_K_M", "name": "Qwen3-1.7B", "hybrid": True,
                   "license": "Apache-2.0"},
}

#: Which sites may be used and how. ``text``: answers are written from the page and short quotes are kept (open
#: licence, commercial use allowed). ``check``: only confirms an answer; just the link is kept.
SOURCES = {
    "minecraft.fandom.com": ("text", "CC BY-SA 3.0"),
    "en.wikipedia.org": ("text", "CC BY-SA 4.0"),
    "www.minecraft.net": ("check", "official Mojang page (link only, nothing copied)"),
    "minecraft.net": ("check", "official Mojang page (link only, nothing copied)"),
    "help.minecraft.net": ("check", "official Mojang help (link only, nothing copied)"),
    "feedback.minecraft.net": ("check", "official Mojang changelog (link only, nothing copied)"),
}
#: Never read: the licence does not allow commercial use.
BLOCKED = {"minecraft.wiki": "CC BY-NC-SA 3.0 (non-commercial)"}

#: Mechanics a player asks about (searched first, then mobs, then items / blocks / enchantments / effects / biomes).
MECHANICS = (
    "Redstone comparator", "Redstone repeater", "Redstone dust", "Observer", "Piston", "Sticky piston", "Hopper",
    "Dropper", "Dispenser", "Daylight detector", "Target block", "Sculk sensor", "Nether portal", "End portal",
    "Brewing", "Enchanting table", "Anvil", "Grindstone", "Smithing table", "Netherite upgrade", "Beacon",
    "Conduit", "Elytra", "Totem of Undying", "Villager trading", "Villager breeding", "Zombie villager curing",
    "Piglin bartering", "Raid", "Bad Omen", "Trial chambers", "Trial spawner", "Vault", "Mob spawning",
    "Light level", "Hunger", "Saturation", "Experience", "Fall damage", "Armor", "Shield", "Critical hit",
    "Sweeping edge", "Fishing", "Farming", "Bone meal", "Composter", "Breeding", "Taming", "Slime chunk",
    "Spawner", "Respawn anchor", "Bed", "Ender pearl", "Chorus fruit", "Stronghold", "Ancient city",
    "Woodland mansion", "Ocean monument", "Bastion remnant", "Nether fortress", "End city", "Trident",
    "Mace", "Crossbow", "Bow", "Map", "Compass", "Lodestone", "Recovery compass", "Bundle", "Shulker box",
    "Ender chest", "Crafter", "Copper bulb", "Trial key", "Wind charge", "Wolf armor", "Happy ghast",
)
_STOP = set("the a an of to in on for and or is are do does can how what which when where who why with from by "
            "at as it its this that be you your i my me minecraft vanilla java bedrock edition game".split())


# ---------------------------------------------------------------------------- small helpers
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def norm(text: str) -> str:
    """Lower case, plain quotes and dashes, letters/digits/spaces only - for comparing text."""
    text = html.unescape(str(text or "")).lower()
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                                         "\u2013": "-", "\u2014": "-", "\u00a0": " "}))
    text = re.sub(r"(?<=\d)[.,](?=\d)", "\u2027", text)  # keep 1.5 and 1,000 together
    text = re.sub(r"[^a-z0-9\u2027%]+", " ", text)
    return " ".join(text.replace("\u2027", ".").split())


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", str(text or ""))}


def domain(url: str) -> str:
    return urllib.parse.urlsplit(str(url or "")).netloc.lower().split(":")[0]


def source_policy(url: str) -> tuple[str, str] | None:
    """``("text" | "check", licence)`` for a usable page, ``None`` for anything else (blocked, unknown, not
    an English article)."""
    host = domain(url)
    if any(host == b or host.endswith("." + b) for b in BLOCKED):
        return None
    policy = SOURCES.get(host)
    if policy is None:
        return None
    path = urllib.parse.urlsplit(url).path
    if host.endswith(("fandom.com", "wikipedia.org")) and not path.startswith("/wiki/"):
        return None  # other languages (/de/wiki/...) and non-article pages
    if host.endswith(("fandom.com", "wikipedia.org")) and re.match(
            r"/wiki/(Special|Talk|User|File|Category|Template|Help|Minecraft_Wiki|Wikipedia)(_talk)?:", path):
        return None
    return policy


def kaggle_secret(name: str) -> str:
    """A Kaggle secret ("" when it is not set or this is not Kaggle). Never printed."""
    if not name:
        return ""
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore[import-not-found]

        return str(UserSecretsClient().get_secret(name) or "").strip()
    except Exception:  # noqa: BLE001 - missing secret / not on Kaggle
        return str(os.environ.get(name.upper(), "")).strip()


def http_get(url: str, timeout: float = 30, accept: str = "*/*", limit: int = 4_000_000) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - https only, see callers
        return response.read(limit)


def http_json(url: str, timeout: float = 60) -> Any:
    return json.loads(http_get(url, timeout, "application/json").decode("utf-8"))


def parse_json(text: str) -> Any:
    """The first JSON object in an LLM reply (think blocks and code fences ignored); ``None`` if there is none."""
    text = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    start = text.find("{")
    while start != -1:
        depth, inside, escape = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if inside:
                escape = (char == "\\") and not escape
                if char == '"' and not escape:
                    inside = False
                continue
            if char == '"':
                inside = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except ValueError:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------- pages
class _TextParser(HTMLParser):
    """Readable text from wiki HTML: paragraphs and list items on their own lines, table cells joined with
    " | ", and no scripts, navigation boxes, edit links or reference markers."""

    SKIP_TAGS = {"script", "style", "noscript", "nav", "footer", "aside", "sup", "figure", "form", "button"}
    SKIP_CLASS = re.compile(r"navbox|mw-editsection|reference|toc\b|noprint|infobox-image|portable-infobox-image|"
                            r"mw-empty-elt|gallery|hatnote|metadata|ambox|catlinks|printfooter", re.I)
    BLOCK = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "br", "dd", "dt", "table", "ul", "ol",
             "section", "caption", "figcaption", "blockquote", "pre"}
    VOID = {"br", "img", "hr", "input", "meta", "link", "wbr", "source", "col", "area", "base", "embed", "track"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.stack: list[tuple[str, bool]] = []

    def _skipping(self) -> bool:
        return any(skip for _tag, skip in self.stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        skip = tag in self.SKIP_TAGS or bool(self.SKIP_CLASS.search(values.get("class") or "")) \
            or values.get("style", "").replace(" ", "").find("display:none") != -1
        if tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th") and not self._skipping():
            self.parts.append(" | ")
        if tag not in self.VOID:
            self.stack.append((tag, skip))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break
        if tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping():
            self.parts.append(data)


def html_to_text(markup: str) -> str:
    parser = _TextParser()
    parser.feed(str(markup or ""))
    parser.close()
    lines = []
    for line in "".join(parser.parts).splitlines():
        line = re.sub(r"\[\d+\]|\[edit\]", "", " ".join(line.split())).strip(" |")
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return "\n".join(lines)


def page_api_url(url: str) -> str | None:
    """The MediaWiki API URL that returns a wiki page's article HTML (cleaner than the full page)."""
    host, path = domain(url), urllib.parse.urlsplit(url).path
    if not path.startswith("/wiki/"):
        return None
    title = urllib.parse.unquote(path[len("/wiki/"):])
    api = {"minecraft.fandom.com": "https://minecraft.fandom.com/api.php",
           "en.wikipedia.org": "https://en.wikipedia.org/w/api.php"}.get(host)
    if not api:
        return None
    query = urllib.parse.urlencode({"action": "parse", "page": title, "prop": "text", "format": "json",
                                    "redirects": 1, "disablelimitreport": 1, "disableeditsection": 1})
    return f"{api}?{query}"


def fetch_page(url: str, cache: Path | None = None, get: Callable[..., bytes] = http_get) -> str:
    """A page's readable text (cached on disk). Only allowed sites are fetched."""
    if source_policy(url) is None:
        raise ValueError(f"not an allowed source: {domain(url)}")
    path = None
    if cache is not None:
        path = Path(cache) / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".txt")
        if path.is_file():
            return path.read_text(encoding="utf-8")
    api = page_api_url(url)
    if api:
        data = json.loads(get(api, 30, "application/json").decode("utf-8"))
        markup = ((data.get("parse") or {}).get("text") or {}).get("*", "")
        if not markup:
            raise ValueError(f"no article at {url}")
    else:
        markup = get(url, 30, "text/html").decode("utf-8", "replace")
    text = html_to_text(markup)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return text


def query_terms(query: str) -> list[str]:
    return [w for w in norm(query).split() if len(w) > 2 and w not in _STOP]


def passages(text: str, query: str, size: int = 1400, top: int = 2) -> str:
    """The parts of a page that are about ``query`` (``top`` chunks of about ``size`` characters, page order)."""
    chunks, current = [], ""
    for line in str(text or "").splitlines():
        if current and len(current) + len(line) > size:
            chunks.append(current)
            current = ""
        current += line[:size * 2] + "\n"
    if current.strip():
        chunks.append(current)
    terms = query_terms(query)
    phrase = norm(query)

    def score(index: int) -> float:
        low = norm(chunks[index])
        value = sum(low.count(term) for term in terms) + (5 if phrase and phrase in low else 0)
        return value + (0.5 if index == 0 else 0)  # the intro usually says what it is

    best = sorted(sorted(range(len(chunks)), key=score, reverse=True)[:top])
    return "\n".join(chunks[i].strip() for i in best if score(i) > 0)


# ---------------------------------------------------------------------------- web search
def web_search(query: str, max_results: int = 8, ddgs: Any = None, get: Callable[..., bytes] = http_get,
               delay: float = 1.5) -> list[dict[str, str]]:
    """``[{url, title}]`` of allowed pages for ``query``: DuckDuckGo (``ddgs``), then the Minecraft Fandom wiki's own
    search when DuckDuckGo fails or finds nothing usable."""
    found: list[dict[str, str]] = []
    try:
        if ddgs is None:
            from ddgs import DDGS  # type: ignore[import-not-found]

            ddgs = DDGS(timeout=15)
        for hit in ddgs.text(query, max_results=max_results) or []:
            url = str(hit.get("href") or hit.get("url") or "")
            if source_policy(url) and url not in (f["url"] for f in found):
                found.append({"url": url.split("#")[0], "title": str(hit.get("title") or "")})
    except Exception as exc:  # noqa: BLE001 - rate limits, network: the wiki search still works
        log(f"web search failed ({type(exc).__name__}: {str(exc)[:120]}) - using the Fandom wiki search")
    finally:
        time.sleep(delay)
    if not found:
        bare = re.sub(r"\bsite:\S+", "", query).strip()
        api = "https://minecraft.fandom.com/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": bare, "srlimit": min(max_results, 10), "format": "json"})
        try:
            data = json.loads(get(api, 30, "application/json").decode("utf-8"))
            for hit in (data.get("query") or {}).get("search") or []:
                title = str(hit.get("title") or "")
                found.append({"url": "https://minecraft.fandom.com/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                              "title": title})
        except Exception as exc:  # noqa: BLE001
            log(f"Fandom wiki search failed too ({type(exc).__name__})")
    return found


# ---------------------------------------------------------------------------- game data (topics + recipe check)
def load_minecraft_data(version: str = MINECRAFT_DATA_VERSION, fetch: Callable[[str], Any] | None = None) -> dict:
    fetch = fetch or (lambda rel: http_json(MINECRAFT_DATA_URL + rel))
    paths = (fetch("dataPaths.json").get("pc") or {}).get(version) or {}
    data: dict[str, Any] = {"version": version}
    for name in ("items", "blocks", "entities", "enchantments", "effects", "biomes", "recipes"):
        if paths.get(name):
            try:
                data[name] = fetch(f"{paths[name]}/{name}.json")
            except Exception as exc:  # noqa: BLE001 - one missing file only costs its topics / checks
                log(f"minecraft-data: {name}.json skipped ({type(exc).__name__})")
    return data


def _label(thing: dict) -> str:
    return str(thing.get("displayName") or str(thing.get("name", "")).replace("_", " ").title())


def topics(data: dict, rng: random.Random | None = None) -> list[str]:
    """What to search for: the mechanics, then mobs, then items, blocks, enchantments, effects and biomes mixed."""
    rng = rng or random.Random(0)
    mobs = [_label(e) for e in data.get("entities") or [] if isinstance(e, dict)
            and e.get("type") in ("hostile", "passive", "animal", "ambient", "water_creature", "mob")]
    rest = [_label(x) + suffix for key, suffix in (("items", ""), ("blocks", ""), ("enchantments", " (enchantment)"),
                                                   ("effects", " (status effect)"), ("biomes", " (biome)"))
            for x in data.get(key) or [] if isinstance(x, dict) and not str(x.get("name", "")).endswith("spawn_egg")
            and x.get("name") not in ("air", "cave_air", "void_air")]
    rng.shuffle(mobs)
    rng.shuffle(rest)
    out, seen = [], set()
    for topic in list(MECHANICS) + mobs + rest:
        key = topic.lower()
        if key not in seen:
            seen.add(key)
            out.append(topic)
    return out


def _group(names: list[str]) -> str:
    names = sorted(set(names))
    if len(names) == 1:
        return names[0]
    last = {n.split()[-1] for n in names}
    return f"any {last.pop()}" if len(last) == 1 else " or ".join(names)


def _recipe_cells(recipe: dict) -> list[Any]:
    if isinstance(recipe.get("inShape"), list):
        return [c for row in recipe["inShape"] if isinstance(row, list) for c in row]
    return sorted(recipe.get("ingredients") or [], key=lambda i: (not isinstance(i, int), i if isinstance(i, int) else 0))


def _recipe_signature(recipe: dict) -> str:
    shape = recipe.get("inShape") if isinstance(recipe.get("inShape"), list) else []
    return json.dumps([[[c is None for c in row] for row in shape if isinstance(row, list)],
                       len(recipe.get("ingredients") or []), (recipe.get("result") or {}).get("count", 1)])


def recipe_book(data: dict) -> dict[str, dict[str, Any]]:
    """``{item name (lower): {"name", "count", "ingredients": {name: count}}}`` from minecraft-data, built on the
    recipe most variants share (12 wood types of a piston = "any Planks")."""
    items = {i.get("id"): i for i in data.get("items") or [] if isinstance(i, dict)}
    book: dict[str, dict[str, Any]] = {}
    for result, variants in (data.get("recipes") or {}).items():
        item = items.get(int(result)) if str(result).isdigit() else None
        recipes = [v for v in variants or [] if isinstance(v, dict)] if isinstance(variants, list) else []
        if item is None or not recipes:
            continue
        cells, signature = _recipe_cells, _recipe_signature
        counts: dict[str, int] = {}
        for r in recipes:
            counts[signature(r)] = counts.get(signature(r), 0) + 1
        main = max(recipes, key=lambda r: counts[signature(r)])
        same = [cells(r) for r in recipes if signature(r) == signature(main)]
        if not same[0] or any(len(c) != len(same[0]) for c in same):
            continue
        ingredients: dict[str, int] = {}
        usable = True
        for slot in range(len(same[0])):
            ids = [c[slot] for c in same]
            if all(v is None for v in ids):
                continue
            if any(not isinstance(v, int) or v not in items for v in ids):
                usable = False
                break
            name = _group([_label(items[v]) for v in ids])
            ingredients[name] = ingredients.get(name, 0) + 1
        if usable and ingredients:
            book[_label(item).lower()] = {"name": _label(item), "count": (main.get("result") or {}).get("count", 1),
                                          "ingredients": ingredients}
    return book


def _stems(word: str) -> set[str]:
    word = word.lower()
    out = {word}
    if word.endswith("s") and len(word) > 3:
        out.add(word[:-1])
    else:
        out.add(word + "s")
    return out


def data_check(question: str, answer: str, book: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """``("ok" | "contradicts" | "n/a", note)``: a crafting answer against the game's recipe."""
    if not book or not re.search(r"\b(craft\w*|recipe)\b", question, re.I):
        return "n/a", ""
    q = f" {norm(question)} "
    names = sorted((n for n in book if f" {norm(n)} " in q), key=len, reverse=True)
    if not names:
        return "n/a", ""
    recipe = book[names[0]]
    words = set(norm(answer).split())
    text = norm(answer)
    for ingredient, count in recipe["ingredients"].items():
        options = [o for o in re.split(r" or ", ingredient.replace("any ", "")) if o]
        keys = [w for o in options for w in norm(o).split() if len(w) >= 4] or [norm(options[0])]
        if not any(_stems(k) & words for k in keys):
            return "contradicts", f"the game's recipe for {recipe['name']} needs {ingredient}; the answer leaves it out"
        last = norm(options[0]).split()[-1]
        said = {int(n) for n in re.findall(rf"\b(\d+)\s+(?:x\s+)?(?:[a-z]+\s+){{0,2}}?(?:{'|'.join(map(re.escape, _stems(last)))})\b",
                                           text)}
        if said and count not in said:
            return "contradicts", f"the game's recipe uses {count} {ingredient}, the answer says {sorted(said)[0]}"
    return "ok", f"matches the {MINECRAFT_DATA_VERSION} recipe for {recipe['name']}"


# ---------------------------------------------------------------------------- checks
def quote_found(quote: str, page: str) -> bool:
    """Is ``quote`` really on the page? Exact after normalising, or 90% of its 4-word runs (a small LLM sometimes
    drops a word or a bracket)."""
    q, p = norm(quote), norm(page)
    if len(q.split()) < 3:
        return False
    if q in p:
        return True
    words = q.split()
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return bool(grams) and sum(g in p for g in grams) / len(grams) >= 0.9


def unbacked_numbers(answer: str, evidence: list[str], question: str = "") -> set[str]:
    backed = numbers(question)
    for quote in evidence:
        backed |= numbers(quote)
    return {n for n in numbers(answer) if n not in backed}


# ---------------------------------------------------------------------------- the LLM (llama.cpp server, CPU)
PAIR_SCHEMA = {
    "type": "object",
    "properties": {"pairs": {"type": "array", "maxItems": 6, "items": {
        "type": "object",
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"},
                       "evidence": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}},
                       "edition": {"type": "string", "enum": ["java", "bedrock", "both", "unknown"]}},
        "required": ["question", "answer", "evidence", "edition"]}}},
    "required": ["pairs"],
}
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"verdict": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "NOT_FOUND"]},
                   "reason": {"type": "string"}},
    "required": ["verdict", "reason"],
}
PROPOSE_SYSTEM = (
    "You write question-answer pairs for a Minecraft knowledge dataset. Use ONLY the source text you are given - "
    "never your own memory. Write questions a player would really ask (varied wording: how, what, why, can, how "
    "many). Each answer: 1-4 sentences in your own words, every fact and number taken from the source, nothing "
    "added. For each pair copy 1-3 sentences from the source, word for word, that prove the answer (evidence). If "
    "the source says the fact is for Java or Bedrock Edition only, set edition and say so in the answer. Skip "
    "anything the source is vague about, history or trivia nobody asks, and facts that depend on an old version. "
    "Return JSON only."
)
JUDGE_SYSTEM = (
    "You check a Minecraft answer against evidence text. SUPPORTED: every fact and number in the answer is stated "
    "in the evidence. CONTRADICTED: the evidence says something different. NOT_FOUND: the evidence does not say "
    "it (or only part of it). Be strict: anything extra in the answer that the evidence does not state means "
    "NOT_FOUND. Return JSON only."
)


class LlamaServer:
    """llama.cpp's ``llama-server`` with one GGUF, on CPU; ``chat()`` returns parsed JSON (schema-constrained)."""

    def __init__(self, binary: Path, model: Path, threads: int = 0, ctx: int = 8192, port: int = 8089,
                 hybrid: bool = False) -> None:
        self.binary, self.model, self.port, self.hybrid = Path(binary), Path(model), port, hybrid
        self.threads = threads or max(1, os.cpu_count() or 4)
        self.ctx = ctx
        self.process: subprocess.Popen | None = None
        self.calls = 0
        self.seconds = 0.0

    def start(self, wait: float = 900) -> "LlamaServer":
        env = {**os.environ, "LD_LIBRARY_PATH": f"{self.binary.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
        command = [str(self.binary), "-m", str(self.model), "-c", str(self.ctx), "-t", str(self.threads),
                   "--host", "127.0.0.1", "--port", str(self.port), "--jinja", "-np", "1", "--no-webui"]
        logfile = open(self.model.parent / "llama-server.log", "w")  # noqa: SIM115 - lives as long as the server
        self.process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, env=env)
        end = time.time() + wait
        while time.time() < end:
            if self.process.poll() is not None:
                tail = (self.model.parent / "llama-server.log").read_text(errors="replace")[-1500:]
                raise RuntimeError(f"llama-server stopped (exit {self.process.returncode}):\n{tail}")
            try:
                if json.loads(http_get(f"http://127.0.0.1:{self.port}/health", 5)).get("status") == "ok":
                    return self
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server did not become ready")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(30)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def chat(self, system: str, user: str, schema: dict, max_tokens: int = 700) -> Any:
        body: dict[str, Any] = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.2, "top_p": 0.9, "max_tokens": max_tokens, "cache_prompt": True,
            "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}},
        }
        if self.hybrid:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/chat/completions",
                                         data=json.dumps(body).encode("utf-8"), method="POST",
                                         headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=900) as response:
            reply = json.loads(response.read().decode("utf-8"))
        self.calls += 1
        self.seconds += time.time() - started
        return parse_json(((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _runs(binary: Path) -> bool:
    env = {**os.environ, "LD_LIBRARY_PATH": f"{binary.parent}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    try:
        done = subprocess.run([str(binary), "--version"], capture_output=True, text=True, env=env, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and "GLIBC" not in done.stderr


def _extract(archive: Path, target: Path) -> None:
    import tarfile
    import zipfile

    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not (target / name).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe path in {archive.name}: {name}")
            zf.extractall(target)
        return
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            if not (target / member.name).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in {archive.name}: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tar.extractall(target, filter="tar")
        else:  # Python 3.10
            tar.extractall(target)


def _download(url: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=600) as response, open(target, "wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle, 1 << 20)
    return target


def llama_server_binary(tools: Path, fetch_json: Callable[[str], Any] = http_json,
                        download: Callable[[str, Path], Path] = _download) -> Path:
    """``llama-server`` for CPU: one already in ``tools``, else the newest llama.cpp Linux x64 CPU build, else
    compiled from that release's source (~10 min on 4 cores) when the prebuilt one cannot start here."""
    tools = Path(tools)
    for found in sorted(tools.rglob("llama-server")) if tools.is_dir() else []:
        if found.is_file() and _runs(found):
            return found
    releases = fetch_json(f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=15")
    for release in releases:
        tag = str(release.get("tag_name") or "")
        asset = next((a for a in release.get("assets") or []
                      if re.fullmatch(rf"llama-{re.escape(tag)}-bin-ubuntu-x64\.(tar\.gz|zip)", str(a.get("name")))), None)
        if not asset:
            continue
        log(f"llama.cpp {tag}: downloading the CPU build")
        folder = tools / f"llama.cpp-{tag}"
        _extract(download(asset["browser_download_url"], folder / asset["name"]), folder / "bin")
        binary = next((p for p in sorted((folder / "bin").rglob("llama-server")) if p.is_file()), None)
        if binary:
            binary.chmod(0o755)
            if _runs(binary):
                return binary
        log("the prebuilt llama-server does not start here - compiling it from source (~10 min)")
        src = folder / "src"
        _extract(download(f"https://codeload.github.com/{LLAMA_CPP_REPO}/tar.gz/refs/tags/{tag}", folder / "src.tar.gz"),
                 src)
        top = next(p.parent for p in src.rglob("CMakeLists.txt") if (p.parent / "tools").is_dir() or
                   (p.parent / "examples").is_dir())
        subprocess.run(["cmake", "-S", str(top), "-B", str(top / "build"), "-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF",
                        "-DCMAKE_BUILD_TYPE=Release"], check=True)
        subprocess.run(["cmake", "--build", str(top / "build"), "--target", "llama-server", "-j",
                        str(os.cpu_count() or 4)], check=True)
        return next(p for p in sorted((top / "build").rglob("llama-server")) if p.is_file())
    raise RuntimeError("no llama.cpp Linux x64 CPU release found")


def model_file(spec: dict, folder: Path, token: str = "") -> Path:
    """The GGUF of ``spec`` (downloaded once into ``folder``)."""
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]

    files = HfApi(token=token or None).list_repo_files(spec["repo"])
    wanted = [f for f in files if f.endswith(f"{spec['quant']}.gguf") and "/" not in f]
    if not wanted:
        raise RuntimeError(f"{spec['repo']} has no {spec['quant']} GGUF")
    return Path(hf_hub_download(spec["repo"], wanted[0], local_dir=str(folder), token=token or None))


# ---------------------------------------------------------------------------- the pipeline
def propose(llm: Any, topic: str, url: str, passage: str, k: int = 3) -> list[dict[str, Any]]:
    user = (f"Topic: {topic}\nSource: {url}\n\nSOURCE TEXT:\n<<<\n{passage}\n>>>\n\n"
            f"Write up to {k} question-answer pairs about {topic} that this text fully answers.")
    reply = llm.chat(PROPOSE_SYSTEM, user, PAIR_SCHEMA, max_tokens=260 * k + 120) or {}
    pairs = reply.get("pairs") if isinstance(reply, dict) else None
    out = []
    for pair in pairs or []:
        if not isinstance(pair, dict):
            continue
        question = " ".join(str(pair.get("question") or "").split())
        answer = " ".join(str(pair.get("answer") or "").split())
        evidence = [" ".join(str(e).split()) for e in pair.get("evidence") or [] if str(e).strip()]
        if question and answer and evidence:
            out.append({"question": question, "answer": answer, "evidence": evidence[:3],
                        "edition": str(pair.get("edition") or "unknown")})
    return out[:k]


def judge(llm: Any, question: str, answer: str, evidence: str) -> tuple[str, str]:
    user = f"QUESTION: {question}\nANSWER: {answer}\n\nEVIDENCE:\n<<<\n{evidence}\n>>>"
    reply = llm.chat(JUDGE_SYSTEM, user, VERDICT_SCHEMA, max_tokens=160) or {}
    verdict = str(reply.get("verdict") or "NOT_FOUND").upper() if isinstance(reply, dict) else "NOT_FOUND"
    return (verdict if verdict in ("SUPPORTED", "CONTRADICTED", "NOT_FOUND") else "NOT_FOUND",
            str(reply.get("reason") or "")[:300] if isinstance(reply, dict) else "")


def check_pair(pair: dict, page: str, llm: Any, book: dict) -> tuple[bool, dict[str, Any], str]:
    """Checks 3-6 (see the module doc). ``(passed, checks, why it failed)``."""
    checks: dict[str, Any] = {}
    question, answer = pair["question"], pair["answer"]
    if not (10 <= len(question) <= 300 and 15 <= len(answer) <= 1200):
        return False, checks, "question or answer too short / long"
    if not question.rstrip().endswith("?") and not re.match(r"(?i)(how|what|why|can|does|do|is|are|which|when|where)",
                                                            question):
        return False, checks, "not a question"
    missing = [q for q in pair["evidence"] if not quote_found(q, page)]
    checks["evidence_on_page"] = not missing
    if missing:
        return False, checks, f"evidence not on the page: {missing[0][:80]}"
    extra = unbacked_numbers(answer, pair["evidence"], question)
    checks["numbers_backed"] = not extra
    if extra:
        return False, checks, f"numbers not in the evidence: {', '.join(sorted(extra))}"
    verdict, reason = judge(llm, question, answer, "\n".join(pair["evidence"]))
    checks["second_look"] = verdict
    if verdict != "SUPPORTED":
        return False, checks, f"second look: {verdict} ({reason[:120]})"
    status, note = data_check(question, answer, book)
    checks["game_data"] = status
    if status == "contradicts":
        return False, checks, note
    return True, checks, ""


def corroborate(llm: Any, pair: dict, search: Callable[[str], list[dict]], fetch: Callable[[str], str],
                skip: set[str]) -> tuple[str, str]:
    """``(verdict, url)`` from an independent page: another site, or an official one, or another wiki page."""
    query = pair["question"] if "minecraft" in pair["question"].lower() else pair["question"] + " Minecraft"
    for hit in search(query)[:5]:
        url = hit["url"]
        if url in skip:
            continue
        try:
            text = passages(fetch(url), pair["question"] + " " + pair["answer"], size=1200, top=1)
        except Exception:  # noqa: BLE001 - one page failing is fine
            continue
        if not text:
            continue
        verdict, _reason = judge(llm, pair["question"], pair["answer"], text)
        if verdict in ("SUPPORTED", "CONTRADICTED"):
            return verdict, url
    return "NOT_FOUND", ""


def question_key(question: str) -> str:
    return " ".join(sorted(set(query_terms(question))))


class Collector:
    """The dataset on disk (``/kaggle/working/Minecraftism-dataset``): rows, rejects, progress, card."""

    def __init__(self, folder: Path, model: str) -> None:
        self.folder, self.model = Path(folder), model
        self.rows: list[dict] = []
        self.rejected: list[dict] = []
        self.done: list[str] = []
        self.keys: set[str] = set()
        self.started = time.time()
        self.load()

    def load(self) -> None:
        for name, target in (("data/train.jsonl", self.rows), ("meta/rejected.jsonl", self.rejected)):
            path = self.folder / name
            if path.is_file():
                target[:] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        progress = self.folder / "meta" / "progress.json"
        if progress.is_file():
            self.done = list(json.loads(progress.read_text(encoding="utf-8")).get("done_topics") or [])
        self.keys = {question_key(r["question"]) for r in self.rows}
        self.start_rows = len(self.rows)

    def seen(self, question: str) -> bool:
        return question_key(question) in self.keys

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.keys.add(question_key(row["question"]))

    def save(self) -> None:
        high = [r for r in self.rows if r.get("trust") == "high"]
        for name, items in (("data/train.jsonl", self.rows), ("data/high/train.jsonl", high),
                            ("meta/rejected.jsonl", self.rejected)):
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
        (self.folder / "meta" / "progress.json").write_text(json.dumps(
            {"done_topics": self.done, "rows": len(self.rows), "rejected": len(self.rejected),
             "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1), encoding="utf-8")
        (self.folder / "README.md").write_text(dataset_card(self.rows, self.rejected, self.model), encoding="utf-8")

    def stats(self) -> dict[str, int]:
        high = sum(1 for r in self.rows if r.get("trust") == "high")
        return {"rows": len(self.rows), "high": high, "medium": len(self.rows) - high, "rejected": len(self.rejected),
                "topics": len(self.done)}


def make_row(topic: str, pair: dict, url: str, licence: str, checks: dict, trust: str, confirmed: str,
             model: str) -> dict:
    return {
        "question": pair["question"], "answer": pair["answer"],
        "messages": [{"role": "user", "content": pair["question"]}, {"role": "assistant", "content": pair["answer"]}],
        "topic": topic, "edition": pair.get("edition", "unknown"), "trust": trust,
        "sources": [{"url": url, "license": licence}] + ([{"url": confirmed, "license": source_policy(confirmed)[1],
                                                           "role": "confirms"}] if confirmed and source_policy(confirmed) else []),
        "evidence": pair["evidence"], "checks": checks, "source": "web-verified", "kind": "mcqa", "model": model,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def size_category(count: int) -> str:
    return next(label for limit, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                                           (float("inf"), "100K<n<1M")) if count < limit)


def dataset_card(rows: list[dict], rejected: list[dict], model: str) -> str:
    high = sum(1 for r in rows if r.get("trust") == "high")
    sites: dict[str, int] = {}
    for row in rows:
        for source in row.get("sources") or []:
            sites[domain(source["url"])] = sites.get(domain(source["url"]), 0) + 1
    table = "\n".join(f"| {site} | {SOURCES.get(site, ('', '?'))[1]} | {count} |" for site, count in
                      sorted(sites.items(), key=lambda kv: -kv[1])) or "| - | - | 0 |"
    return f"""---
license: cc-by-sa-4.0
language:
- en
pretty_name: Minecraftism dataset
task_categories:
- question-answering
- text-generation
tags:
- minecraft
- pymc-bot
size_categories:
- {size_category(len(rows))}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
- config_name: high
  data_files:
  - split: train
    path: data/high/train.jsonl
---

# Minecraftism dataset

Checked Minecraft questions and answers for the **Minecraftism** models of
[PyMC Bot](https://github.com/codero-sus/PyMC_Bot), collected by the notebook
`pynb/Minecraftism_Dataset_Kaggle.ipynb` (Kaggle, CPU only): web search + a small LLM ({model}) that may only
write what a page says, and then checks on every answer.

**{len(rows)} rows** - {high} `high` trust (confirmed by the game's own data or an independent page), {len(rows) - high}
`medium` (one source, passed every check). {len(rejected)} rejected pairs (with the reason) are in
`meta/rejected.jsonl` for review - they are not training data.

## How an answer gets in

1. Only pages whose licence allows commercial use are read (Minecraft Fandom wiki CC BY-SA 3.0, Wikipedia
   CC BY-SA 4.0). minecraft.wiki (CC BY-NC-SA) is never used. Official Mojang pages only confirm answers;
   only their links are kept.
2. The LLM writes the pair from the page text alone and copies the sentences that prove it (`evidence`).
3. Every evidence quote must really be on the page; every number in the answer must be in the evidence.
4. A second LLM pass sees only question, answer and evidence and must answer SUPPORTED.
5. Crafting answers must match the game's recipes (minecraft-data {MINECRAFT_DATA_VERSION}, MIT).
6. The question is searched again: an independent page that contradicts the answer rejects it, one that
   confirms it makes it `high`.

## Fields

`question`, `answer`, `messages` (chat format), `topic`, `edition` (java / bedrock / both / unknown), `trust`,
`sources` (URL + licence of each page), `evidence` (the quotes), `checks`, `model`, `created`.

## Use it

```python
from datasets import load_dataset

rows = load_dataset("<you>/{DATASET_NAME}", split="train")          # every checked row
strict = load_dataset("<you>/{DATASET_NAME}", "high", split="train")  # only the confirmed ones
```

Train Minecraftism on it: `train_minecraft_llm.py --minecraft-qa-dataset <you>/{DATASET_NAME}` (it reads
`question` / `answer`). A private dataset needs your token (`HF_TOKEN`) to load.

## Sources and licence

| site | licence | rows |
|---|---|---|
{table}

The answers are written by the model from the pages above, with the quotes and links kept for attribution. As
the text sources are CC BY-SA, this dataset is **CC BY-SA 4.0** (share-alike; commercial use allowed). Game facts
are not copyrightable; Minecraft is a trademark of Mojang Studios - this dataset is not affiliated with Mojang or
Microsoft. Checks are automatic: a `medium` row can still be wrong, so review before relying on it.

Made with ❤️ by [codero-sus](https://the-codero.web.app/) ([codero.is-a.dev](https://codero.is-a.dev)).
"""


class Discord:
    """Messages to a Discord webhook. The ``ping_role`` role only hears the important ones (start, upload, error,
    end). Never prints the webhook."""

    def __init__(self, url: str, role: str = "", post: Callable[[str, bytes], None] | None = None) -> None:
        self.url = url if str(url).startswith("https://") else ""
        digits = re.sub(r"\D", "", str(role or ""))
        self.role = digits
        self.post = post or self._post
        self.dead = False

    def _post(self, url: str, body: bytes) -> None:
        request = urllib.request.Request(url, data=body, method="POST", headers={
            "Content-Type": "application/json", "User-Agent": UA})
        urllib.request.urlopen(request, timeout=15).close()  # noqa: S310 - the webhook from the secret

    def send(self, text: str, title: str = "", level: str = "info", important: bool = False) -> bool:
        if not self.url or self.dead:
            return False
        colors = {"info": 0x3498DB, "good": 0x2ECC71, "warn": 0xF1C40F, "bad": 0xE74C3C}
        payload: dict[str, Any] = {
            "username": "Minecraftism dataset", "allowed_mentions": {"parse": []},
            "embeds": [{"title": title[:256], "description": str(text)[:4000], "color": colors.get(level, colors["info"]),
                        "footer": {"text": "Minecraftism dataset on Kaggle · PyMC Bot by codero-sus"}}],
        }
        if self.role and (important or level == "bad"):
            payload["content"] = f"<@&{self.role}>"
            payload["allowed_mentions"] = {"parse": [], "roles": [self.role]}
        try:
            self.post(self.url, json.dumps(payload).encode("utf-8"))
            return True
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                self.dead = True
                log(f"Discord answered HTTP {exc.code} - check the webhook secret; no more messages")
        except Exception:  # noqa: BLE001 - never stop collecting for a chat message
            pass
        return False


def hub_repo_id(api: Any, name: str) -> str:
    """``<your HF name>/<name>``; stops right away with a clear message when the token cannot write (instead of
    failing at the first upload, half an hour in)."""
    info = api.whoami()
    role = ((info.get("auth") or {}).get("accessToken") or {}).get("role")
    if role == "read":
        raise SystemExit("Your Hugging Face token can only read. Make a *write* token (huggingface.co/settings/tokens) "
                         "and put it in the Kaggle secret, or set UPLOAD = False.")
    return name if "/" in name else f"{info['name']}/{name}"


def hub_make_private(api: Any, repo_id: str, private: bool = True) -> bool:
    """Create the dataset repo (private when asked) and switch an existing public one to private: with
    ``PRIVATE = True`` the dataset is never visible to anyone else. Returns whether it is private."""
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    try:
        is_private = bool(api.repo_info(repo_id, repo_type="dataset").private)
    except Exception:  # noqa: BLE001 - unknown: make sure below
        is_private = False
    if private and not is_private:
        api.update_repo_settings(repo_id, repo_type="dataset", private=True)
        log(f"{repo_id} was public - switched it to private")
        is_private = True
    return is_private


def hub_resume(api: Any, repo_id: str, folder: Path, token: str) -> bool:
    """Download what an earlier session uploaded (rows, rejects, progress) so this one goes on from there."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    try:
        if not api.repo_exists(repo_id, repo_type="dataset"):
            return False
        snapshot_download(repo_id, repo_type="dataset", local_dir=str(folder), token=token or None,
                          allow_patterns=["data/*", "meta/*"])
        return (Path(folder) / "data" / "train.jsonl").is_file()
    except Exception as exc:  # noqa: BLE001 - start fresh rather than fail
        log(f"could not resume from {repo_id} ({type(exc).__name__}) - starting fresh")
        return False


def hub_upload(api: Any, repo_id: str, folder: Path, private: bool, message: str) -> str:
    hub_make_private(api, repo_id, private)  # every time: it stays private even if someone flipped it meanwhile
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder), commit_message=message,
                      allow_patterns=["README.md", "data/*", "meta/*"])
    return f"https://huggingface.co/datasets/{repo_id}"


def session_start() -> float:
    """When this Kaggle session (the kernel) started - its 12 hours count from there."""
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:  # noqa: BLE001
        return time.time()


def gather(topic: str, cfg: dict, search: Callable[[str], list[dict]],
           fetch: Callable[[str], str]) -> list[tuple[str, tuple[str, str], str, str]]:
    """``[(url, policy, page, passage)]`` for one topic: the web part, run ahead in the background."""
    found = []
    for hit in search(f"{topic} site:minecraft.fandom.com")[: int(cfg.get("pages_per_topic", 1))]:
        url = hit["url"]
        policy = source_policy(url)
        if not policy or policy[0] != "text":
            continue
        page = fetch(url)
        passage = passages(page, topic, size=int(cfg.get("passage_chars", 1400)), top=2)
        if len(passage) >= 200:
            found.append((url, policy, page, passage))
    return found


def collect(cfg: dict, llm: Any, collector: Collector, book: dict, topic_list: list[str],
            search: Callable[[str], list[dict]], fetch: Callable[[str], str], deadline: float,
            on_upload: Callable[[str], None] = lambda _m: None, discord: Discord | None = None) -> None:
    """The main loop: one topic at a time until the time, row or topic limit, uploading every few minutes. The
    next topic's search and pages are fetched in the background while the LLM works on this one, so the CPU never
    waits for the network. Saves what it has even when stopped (the Kaggle stop button)."""
    from concurrent.futures import ThreadPoolExecutor

    discord = discord or Discord("")
    last_upload, last_report = time.time(), time.time()
    todo = [t for t in topic_list if t not in set(collector.done)]
    if cfg.get("max_topics"):
        todo = todo[:int(cfg["max_topics"])]
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    ahead = None
    try:
        for number, topic in enumerate(todo, 1):
            if time.time() > deadline or (cfg.get("max_rows") and len(collector.rows) >= int(cfg["max_rows"])):
                break
            current = ahead or pool.submit(gather, topic, cfg, search, fetch)
            ahead = pool.submit(gather, todo[number], cfg, search, fetch) if number < len(todo) else None
            before = len(collector.rows)
            try:
                for url, policy, page, passage in current.result():
                    for pair in propose(llm, topic, url, passage, int(cfg.get("pairs_per_page", 3))):
                        if collector.seen(pair["question"]):
                            continue
                        passed, checks, why = check_pair(pair, page, llm, book)
                        trust, confirmed = "medium", ""
                        if passed and cfg.get("corroborate", True):
                            verdict, confirmed = corroborate(llm, pair, search, fetch, {url})
                            checks["independent_page"] = verdict
                            if verdict == "CONTRADICTED":
                                passed, why = False, f"an independent page contradicts it ({confirmed})"
                        if not passed:
                            collector.rejected.append({"topic": topic, "url": url, "why": why, **pair,
                                                       "checks": checks})
                            continue
                        if checks.get("game_data") == "ok" or checks.get("independent_page") == "SUPPORTED":
                            trust = "high"
                        if trust == "medium" and cfg.get("keep", "medium") == "high":
                            collector.rejected.append({"topic": topic, "url": url, "why": "only medium trust", **pair,
                                                       "checks": checks})
                            continue
                        collector.add(make_row(topic, pair, url, policy[1], checks, trust, confirmed,
                                               cfg.get("model", "")))
            except Exception as exc:  # noqa: BLE001 - one bad topic never stops the run
                log(f"{topic}: skipped ({type(exc).__name__}: {str(exc)[:160]})")
            collector.done.append(topic)
            stats = collector.stats()
            hours = max((time.time() - collector.started) / 3600, 1e-6)
            rate = (stats["rows"] - collector.start_rows) / hours
            log(f"[{number}/{len(todo)}] {topic}: +{stats['rows'] - before} rows (total {stats['rows']}: "
                f"{stats['high']} high, {stats['medium']} medium; {stats['rejected']} rejected) - {rate:.0f} rows/h, "
                f"{max(deadline - time.time(), 0) / 3600:.1f} h left")
            if time.time() - last_report > 3600:
                last_report = time.time()
                discord.send(f"{stats['rows']} rows ({stats['high']} high trust), {stats['rejected']} rejected, "
                             f"{stats['topics']} topics done, {rate:.0f} rows/h.", "Progress")
            if time.time() - last_upload > float(cfg.get("upload_every_min", 30)) * 60:
                last_upload = time.time()
                collector.save()
                on_upload(f"{stats['rows']} rows after {stats['topics']} topics")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        collector.save()


def main(cfg: dict) -> dict:
    """Everything the notebook's last cell runs. ``cfg`` = the notebook settings (see the settings cell)."""
    started = cfg.get("session_start") or session_start()
    work, tmp = Path(cfg.get("work", "/kaggle/working")), Path(cfg.get("tmp", "/kaggle/tmp"))
    folder = work / DATASET_NAME
    for path in (folder, tmp / "pages", tmp / "models", tmp / "tools"):
        path.mkdir(parents=True, exist_ok=True)
    token = kaggle_secret(cfg.get("hf_secret", "HF_TOKEN"))
    discord = Discord(kaggle_secret(cfg.get("discord_secret", "")) if cfg.get("discord", True) else "",
                      kaggle_secret(cfg.get("ping_role_secret", "ping_role")))
    if cfg.get("model", "qwen3-4b") not in MODELS:
        raise SystemExit(f"MODEL = {cfg.get('model')!r} is not known - use one of: {', '.join(MODELS)}")
    if cfg.get("keep", "medium") not in ("medium", "high"):
        raise SystemExit(f"KEEP = {cfg.get('keep')!r} - use \"medium\" or \"high\"")
    spec = MODELS[cfg.get("model", "qwen3-4b")]
    cfg = {**cfg, "model": spec["name"]}
    deadline = started + float(cfg.get("max_hours", 11.75)) * 3600 - float(cfg.get("reserve_min", 15)) * 60
    api = repo_id = None
    if cfg.get("upload", True):
        if not token:
            raise SystemExit(f"Add your Hugging Face token as the Kaggle secret {cfg.get('hf_secret', 'HF_TOKEN')!r} "
                             "(Add-ons > Secrets), or set UPLOAD = False.")
        from huggingface_hub import HfApi  # type: ignore[import-not-found]

        api = HfApi(token=token)
        repo_id = hub_repo_id(api, cfg.get("dataset_repo", DATASET_NAME))
        if cfg.get("resume", True) and hub_resume(api, repo_id, folder, token):
            log(f"resuming from https://huggingface.co/datasets/{repo_id}")
        private = hub_make_private(api, repo_id, bool(cfg.get("private", True)))
        log(f"uploading to https://huggingface.co/datasets/{repo_id} - "
            + ("PRIVATE: only you can see it" if private else "PUBLIC: anyone can see it (PRIVATE = False)"))
    llm = collector = None

    def upload(message: str) -> None:
        if api is not None and repo_id and collector is not None:
            try:
                link = hub_upload(api, repo_id, folder, bool(cfg.get("private", True)), message)
                discord.send(f"{message}: {link}", "Uploaded", "good", important=True)
            except Exception as exc:  # noqa: BLE001 - try again at the next upload
                log(f"upload failed ({type(exc).__name__}: {str(exc)[:200]})")

    def summary(stats: dict) -> dict:
        link = f"https://huggingface.co/datasets/{repo_id}" if repo_id else str(folder)
        samples = [{k: r[k] for k in ("question", "answer", "trust")} for r in (collector.rows if collector else [])[-3:]]
        return {**stats, "link": link, "samples": samples}

    try:
        discord.send(f"Collecting a trusted Minecraft dataset with {spec['name']} on CPU"
                     + (f" -> {repo_id}" if repo_id else "") + ".", "Notebook started", "info", important=True)
        binary = llama_server_binary(tmp / "tools")
        model = model_file(spec, tmp / "models", token)
        log(f"starting llama-server with {model.name} on {os.cpu_count()} CPU threads")
        llm = LlamaServer(binary, model, ctx=int(cfg.get("ctx", 8192)), hybrid=spec["hybrid"]).start()
        data = load_minecraft_data(cfg.get("minecraft_version", MINECRAFT_DATA_VERSION))
        book = recipe_book(data)
        topic_list = topics(data, random.Random(int(cfg.get("seed", 0))))
        collector = Collector(folder, spec["name"])
        log(f"{len(topic_list)} topics ({len(collector.done)} done before), {len(book)} recipes for the game-data "
            f"check, {len(collector.rows)} rows so far")
        cache = tmp / "pages"

        collect(cfg, llm, collector, book, topic_list,
                lambda q: web_search(q, int(cfg.get("search_results", 8))),
                lambda url: fetch_page(url, cache), deadline, upload, discord)
        stats = collector.stats()
        upload(f"final: {stats['rows']} rows ({stats['high']} high trust) from {stats['topics']} topics")
        result = summary(stats)
        log(f"done: {stats} -> {result['link']} (LLM: {llm.calls} calls, {llm.seconds / 60:.0f} min)")
        discord.send(f"{stats['rows']} rows ({stats['high']} high trust, {stats['medium']} medium), "
                     f"{stats['rejected']} rejected, {stats['topics']} topics.\n{result['link']}", "Notebook finished",
                     "good", important=True)
        return result
    except KeyboardInterrupt:  # the Kaggle stop button: keep and upload what was collected
        if collector is None:
            raise
        stats = collector.stats()
        log(f"stopped - uploading the {stats['rows']} rows collected so far")
        upload(f"stopped early: {stats['rows']} rows ({stats['high']} high trust) from {stats['topics']} topics")
        discord.send(f"Stopped by you after {stats['topics']} topics: {stats['rows']} rows kept.", "Notebook stopped",
                     "warn", important=True)
        return {**summary(stats), "stopped": True}
    except BaseException as exc:
        discord.send(f"`{type(exc).__name__}: {str(exc)[:1500]}`", "Notebook error", "bad", important=True)
        raise
    finally:
        if llm is not None:
            llm.stop()


if __name__ == "__main__" and "ipykernel" not in sys.modules:  # pragma: no cover - the notebook calls main()
    print(json.dumps(main({"upload": False, "max_topics": int(os.environ.get("TOPICS", "3"))}), indent=1))
