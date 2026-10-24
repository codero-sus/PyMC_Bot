"""Minecraft knowledge for the LLM brain: built-in mob facts + web lookups.

Many models know little about Minecraft (which mobs explode, what a piglin wants, how to tame a
wolf). Two things fill the gap:

* The mob database (:mod:`pymc_bot.mobs`, ``data/mobs.json``) - hostility, health, damage, spawn
  conditions and the version every vanilla mob was added in, plus a practical tip
  (:data:`MOB_FACTS`). The mobs near the bot go into its prompt every turn, and questions about a
  mob ("how much damage does a warden do", "is the sulphur cube hostile") are answered from it
  offline before the web is asked. It also covers mobs newer than mineflayer.
* :class:`KnowledgeBase` - the ``search`` action. The model asks a question ("how do i tame a
  wolf", "what does a warden do") and gets a short answer, by default from the Minecraft Wiki's
  public MediaWiki API (no key). DuckDuckGo instant answers, a SearXNG instance or the Brave Search
  API can be used instead; the Brave key is read from an environment variable and never saved.
  Answers are cached on disk (pages rarely change) and the number of real requests per hour is
  capped. Mobs the bot does not know (e.g. from mods) can be looked up automatically.

Everything is shared by all bots of the process: one cache, one rate limit.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from pymc_bot import __version__
from pymc_bot.config import SearchSettings
from pymc_bot.mobs import is_version, load_mobs

USER_AGENT = f"PyMC_Bot/{__version__} (+https://github.com/codero-sus/PyMC_Bot)"

# --------------------------------------------------------------------------- built-in facts
# One line per mob, from the mob database (pymc_bot/data/mobs.json) - see pymc_bot.mobs.
MOB_FACTS: dict[str, str] = {mob_id: mob["fact"] for mob_id, mob in load_mobs().mobs.items()}
UNKNOWN_KINDS = ("unknown", "other", "mob", "entity")


def mob_fact(kind: str) -> str | None:
    kind = str(kind or "").lower().removeprefix("minecraft:")
    return MOB_FACTS.get(kind)


# --------------------------------------------------------------------------- HTML -> text
# <sup> is NOT skipped wholesale: fractions use it (1/3); footnotes carry the "reference" class.
_SKIP_TAGS = {"table", "style", "script", "figure", "math", "noscript", "audio", "video", "gallery"}
_SKIP_CLASSES = ("infobox", "navbox", "hatnote", "mw-editsection", "toc", "notice", "msgbox", "reference",
                 "thumb", "gallery", "noprint", "metadata", "sound", "mw-empty-elt")
_BLOCK_TAGS = {"p", "div", "li", "dd", "dt", "h1", "h2", "h3", "h4", "h5", "h6", "br", "tr", "ul", "ol", "dl"}
_VOID_TAGS = {"br", "img", "hr", "input", "meta", "link", "wbr", "source", "area", "col"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._stack: list[bool] = []  # per open element: does it (or a parent) skip its content?

    @property
    def _skipping(self) -> bool:
        return bool(self._stack and self._stack[-1])

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = " ".join(value or "" for key, value in attrs if key == "class")
        skip = self._skipping or tag in _SKIP_TAGS or any(name in classes for name in _SKIP_CLASSES)
        if tag in _BLOCK_TAGS and not skip:
            self.parts.append("\n- " if tag == "li" else "\n")
        if tag not in _VOID_TAGS:
            self._stack.append(skip)

    def handle_endtag(self, tag: str) -> None:
        if tag in _VOID_TAGS:
            return
        if self._stack:
            self._stack.pop()
        if tag in _BLOCK_TAGS and tag != "li" and not self._skipping:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self.parts.append(data)


def clean_text(text: str) -> str:
    """Plain, prompt-safe text: no braces (the narrative part of the prompt is JSON-free), tidy spaces."""
    text = text.replace("{", "(").replace("}", ")").replace("\u00a0", " ")
    text = re.sub(r"\[\s*(edit|edit source|citation needed|\d+)\s*\]", "", text, flags=re.I)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # one line per paragraph / list item: prompt space is precious
    return "\n".join(line for line in lines if line not in ("", "-")).strip()


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html or "")
    parser.close()
    return clean_text("".join(parser.parts))


def shorten(text: str, limit: int) -> str:
    """Cut at a sentence (or word) boundary near ``limit``."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    stop = max(cut.rfind(". "), cut.rfind(".\n"), cut.rfind("\n"))
    if stop > limit * 0.5:
        return cut[: stop + 1].strip()
    space = cut.rfind(" ")
    return (cut[:space] if space > 0 else cut).rstrip(",;:") + " ..."


# --------------------------------------------------------------------------- results
@dataclass
class LookupResult:
    ok: bool
    query: str
    text: str = ""
    title: str = ""
    url: str = ""
    source: str = ""
    cached: bool = False
    error: str = ""
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self, limit: int = 400) -> str:
        if not self.ok:
            return f"no answer ({self.error})" if self.error else "no answer"
        head = f"{self.title}: " if self.title and not self.text.lower().startswith(self.title.lower()) else ""
        return shorten(head + " ".join(self.text.split()), limit)


class SearchError(RuntimeError):
    pass


STOP_WORDS = {
    "a", "an", "the", "how", "do", "does", "i", "to", "what", "is", "are", "of", "in", "on", "for", "with",
    "can", "you", "my", "me", "minecraft", "mc", "and", "or", "where", "when", "why", "which", "get", "it",
    "best", "way", "should", "from", "find", "make",
}
SECTION_HINTS: dict[str, tuple[str, ...]] = {
    "drop": ("drops", "obtaining"), "loot": ("drops", "obtaining"), "spawn": ("spawning",), "find": ("spawning", "natural generation", "obtaining"),
    "tame": ("taming", "behavior"), "breed": ("breeding", "behavior"), "feed": ("breeding", "taming"),
    "craft": ("crafting", "obtaining", "usage"), "recipe": ("crafting",), "smelt": ("smelting", "obtaining"),
    "kill": ("behavior", "combat", "attacks"), "fight": ("behavior", "combat", "attacks"), "attack": ("behavior", "attacks"),
    "behav": ("behavior",), "avoid": ("behavior",), "do": ("behavior", "usage"), "use": ("usage",),
    "get": ("obtaining",), "obtain": ("obtaining",), "mine": ("obtaining", "breaking"), "break": ("breaking",),
    "trade": ("trading",), "brew": ("brewing",), "enchant": ("enchantments",), "grow": ("growth", "farming"),
    "farm": ("farming", "usage"), "ride": ("riding", "usage"), "sleep": ("usage", "sleeping"),
}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOP_WORDS]


def _is_hint(word: str) -> bool:
    stem = word.rstrip("s")
    return any(stem.startswith(key) for key in SECTION_HINTS)


def topic_of(query: str) -> str:
    """The thing being asked about: "what drops blaze rods" -> "blaze rods".

    Wiki full-text search ranks whole questions badly (question words match random pages), so the
    search uses the topic and the question words only pick the section.
    """
    words = _words(query)
    topic = [w for w in words if not _is_hint(w)]
    return " ".join(topic or words)


# --------------------------------------------------------------------------- providers
class _Provider:
    name = "base"

    def __init__(self, settings: SearchSettings, client: httpx.Client) -> None:
        self.settings = settings
        self.client = client

    def _get(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None) -> Any:
        try:
            response = self.client.get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise SearchError(f"{self.name}: {type(exc).__name__}") from exc
        if response.status_code == 429:
            raise SearchError(f"{self.name}: rate limited")
        if response.status_code >= 400:
            raise SearchError(f"{self.name}: HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as exc:
            raise SearchError(f"{self.name}: not JSON (blocked?)") from exc

    def search(self, query: str) -> LookupResult:  # pragma: no cover - interface
        raise NotImplementedError


class MinecraftWiki(_Provider):
    """The Minecraft Wiki's MediaWiki API: search, then the lead + the section that fits the question."""

    name = "minecraft_wiki"

    def _api(self, **params: Any) -> dict[str, Any]:
        data = self._get(self.settings.wiki_api, {"format": "json", "formatversion": "2", **params})
        if isinstance(data, dict) and data.get("error"):
            raise SearchError(f"wiki: {data['error'].get('code', 'error')}")
        return data if isinstance(data, dict) else {}

    def page_url(self, title: str) -> str:
        base = self.settings.wiki_api.rsplit("/", 1)[0]
        return f"{base}/w/{quote(title.replace(' ', '_'))}"

    def search(self, query: str) -> LookupResult:
        hits: list[dict[str, Any]] = []
        for text in dict.fromkeys(q for q in (topic_of(query), query) if q.strip()):
            hits = (self._api(action="query", list="search", srsearch=text, srlimit="3", srprop="snippet")
                    .get("query", {}).get("search", []))
            if hits:
                break
        if not hits:
            return LookupResult(False, query, source=self.name, error="nothing on the wiki")
        title = str(hits[0].get("title") or "")
        parsed = self._api(action="parse", page=title, prop="sections", redirects="1").get("parse", {})
        title = str(parsed.get("title") or title)
        lead = html_to_text(self._section(title, 0))
        section_text, section_name = "", ""
        wanted = self._pick_section(query, title, parsed.get("sections") or [])
        if wanted:
            section_name, index = wanted
            section_text = html_to_text(self._section(title, index))
            first, _, rest = section_text.partition("\n")
            if first.strip().lower() == section_name.lower():  # the section starts with its own heading
                section_text = rest.strip()
        budget = self.settings.max_chars
        if section_text:
            text = shorten(lead, budget // 2) + f"\n{section_name}: " + shorten(section_text, budget - min(len(lead), budget // 2))
        else:
            text = shorten(lead, budget)
        others = [str(hit.get("title")) for hit in hits[1:] if hit.get("title")]
        if others:
            text += "\nRelated pages: " + ", ".join(others)
        return LookupResult(True, query, text=text, title=title, url=self.page_url(title), source=self.name)

    def _section(self, title: str, index: int | str) -> str:
        parsed = self._api(action="parse", page=title, prop="text", section=str(index), redirects="1",
                           disableeditsection="1", disabletoc="1").get("parse", {})
        text = parsed.get("text", "")
        if isinstance(text, dict):  # formatversion=1 shape
            text = text.get("*", "")
        return str(text or "")

    @staticmethod
    def _pick_section(query: str, title: str, sections: list[dict[str, Any]]) -> tuple[str, str] | None:
        title_words = set(_words(title))
        words = [w for w in _words(query) if w not in title_words]
        wanted: list[str] = []
        for word in words:
            stem = word.rstrip("s")
            for key, names in SECTION_HINTS.items():
                if stem.startswith(key):
                    wanted.extend(names)
            wanted.append(stem)
        if not wanted:
            return None
        for target in wanted:
            for section in sections:
                line = html_to_text(str(section.get("line") or "")).lower()
                if line and (line == target or line.startswith(target) or target in line.split()):
                    return html_to_text(str(section.get("line"))), str(section.get("index") or "")
        return None


class DuckDuckGo(_Provider):
    """DuckDuckGo instant answers (no key; short abstracts, mostly from wikis)."""

    name = "duckduckgo"

    def search(self, query: str) -> LookupResult:
        q = query if "minecraft" in query.lower() else f"minecraft {query}"
        data = self._get("https://api.duckduckgo.com/", {"q": q, "format": "json", "no_html": "1", "skip_disambig": "1"})
        text = str(data.get("AbstractText") or data.get("Answer") or "").strip()
        url = str(data.get("AbstractURL") or "")
        if not text:
            topics: list[str] = []
            for topic in data.get("RelatedTopics") or []:
                for item in topic.get("Topics", [topic]) if isinstance(topic, dict) else []:
                    if item.get("Text"):
                        topics.append(str(item["Text"]))
                        url = url or str(item.get("FirstURL") or "")
            text = "\n".join(f"- {t}" for t in topics[:4])
        if not text:
            return LookupResult(False, query, source=self.name, error="no instant answer")
        return LookupResult(True, query, text=shorten(clean_text(text), self.settings.max_chars),
                            title=str(data.get("Heading") or ""), url=url, source=self.name)


class _ResultList(_Provider):
    def _combine(self, query: str, results: list[tuple[str, str, str]]) -> LookupResult:
        results = [r for r in results if r[0] or r[2]][:3]
        if not results:
            return LookupResult(False, query, source=self.name, error="no results")
        text = "\n".join(f"- {title}: {html_to_text(snippet)}" for title, _, snippet in results)
        return LookupResult(True, query, text=shorten(clean_text(text), self.settings.max_chars),
                            title=results[0][0], url=results[0][1], source=self.name)

    @staticmethod
    def _q(query: str) -> str:
        return query if "minecraft" in query.lower() else f"minecraft {query}"


class SearXNG(_ResultList):
    """A SearXNG instance (yours or a public one) with the JSON format enabled."""

    name = "searxng"

    def search(self, query: str) -> LookupResult:
        base = self.settings.searxng_url.strip().rstrip("/")
        if not base:
            raise SearchError("searxng: set search.searxng_url")
        data = self._get(f"{base}/search", {"q": self._q(query), "format": "json"})
        return self._combine(query, [(str(r.get("title") or ""), str(r.get("url") or ""), str(r.get("content") or ""))
                                     for r in data.get("results") or []])


class Brave(_ResultList):
    """Brave Search API (free tier available); the key comes from an environment variable."""

    name = "brave"

    def search(self, query: str) -> LookupResult:
        key = os.environ.get(self.settings.brave_api_key_env or "", "").strip()
        if not key:
            raise SearchError(f"brave: set the {self.settings.brave_api_key_env} environment variable")
        data = self._get("https://api.search.brave.com/res/v1/web/search", {"q": self._q(query), "count": "3"},
                         headers={"Accept": "application/json", "X-Subscription-Token": key})
        return self._combine(query, [(str(r.get("title") or ""), str(r.get("url") or ""), str(r.get("description") or ""))
                                     for r in (data.get("web") or {}).get("results") or []])


PROVIDERS: dict[str, type[_Provider]] = {
    "minecraft_wiki": MinecraftWiki, "duckduckgo": DuckDuckGo, "searxng": SearXNG, "brave": Brave,
}


# --------------------------------------------------------------------------- knowledge base
def normalize_query(query: str) -> str:
    return " ".join(re.findall(r"[a-z0-9_']+", str(query or "").lower()))[:120]


class KnowledgeBase:
    """Web lookups with a disk cache and an hourly cap, shared by every bot of the process."""

    def __init__(self, settings: SearchSettings, cache_path: Path | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.settings = settings
        self.cache_path = cache_path
        self._transport = transport
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}
        self._requests: deque[float] = deque()
        self._inflight: set[str] = set()
        self.recent: deque[LookupResult] = deque(maxlen=12)
        self.stats = {"lookups": 0, "mob_database": 0, "cache_hits": 0, "web_requests": 0, "failures": 0}
        self._load()

    # ------------------------------------------------------------- cache
    def _load(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict):
            self._cache = {k: v for k, v in raw.items() if isinstance(v, dict) and v.get("ok")}

    def _save(self) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._cache, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.cache_path)
        except OSError:
            pass

    def cached(self, query: str) -> LookupResult | None:
        key = normalize_query(query)
        with self._lock:
            entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - float(entry.get("at", 0)) > self.settings.cache_days * 86400:
            return None
        fields = {k: entry[k] for k in LookupResult.__dataclass_fields__ if k in entry}
        fields.update(cached=True)
        return LookupResult(**fields)

    def clear_cache(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._save()
        return count

    # ------------------------------------------------------------- limits
    def requests_last_hour(self) -> int:
        cutoff = time.time() - 3600
        with self._lock:
            while self._requests and self._requests[0] < cutoff:
                self._requests.popleft()
            return len(self._requests)

    def _take_budget(self) -> bool:
        with self._lock:
            if self.requests_last_hour() >= self.settings.max_per_hour:
                return False
            self._requests.append(time.time())
            return True

    # ------------------------------------------------------------- lookups
    def lookup(self, query: str, record: bool = True) -> LookupResult:
        query = " ".join(str(query or "").split())[:200]
        self.stats["lookups"] += 1
        if not normalize_query(query):
            return LookupResult(False, query, error="empty question")
        local = self.from_mob_database(query)
        if local is not None:
            self.stats["mob_database"] += 1
            if record:
                self.recent.append(local)
            return local
        hit = self.cached(query)
        if hit:
            self.stats["cache_hits"] += 1
            if record:
                self.recent.append(hit)
            return hit
        if not self.settings.enabled:
            return LookupResult(False, query, error="web search is switched off")
        if not self._take_budget():
            return LookupResult(False, query, error=f"search limit reached ({self.settings.max_per_hour}/hour)")
        result = self._search(query)
        if result.ok:
            with self._lock:
                self._cache[normalize_query(query)] = {k: v for k, v in result.to_dict().items() if k != "cached"}
                self._save()
        else:
            self.stats["failures"] += 1
        if record:
            self.recent.append(result)
        return result

    def from_mob_database(self, query: str) -> LookupResult | None:
        """Answer a question about one mob from the built-in database (offline), else None."""
        if not self.settings.use_mob_database:
            return None
        db = load_mobs()
        mob = db.answerable(query)
        if mob is None:
            return None
        version = next((w for w in (t.strip(".?!,") for t in query.split()) if is_version(w)), None)
        return LookupResult(True, query, text=db.describe(mob, version), title=mob["name"],
                            url="https://minecraft.wiki/w/" + quote(mob["name"].replace(" ", "_")),
                            source="mob_database")

    def _search(self, query: str) -> LookupResult:
        chain = [self.settings.provider]
        if self.settings.fallback and self.settings.provider != "duckduckgo":
            chain.append("duckduckgo")
        errors: list[str] = []
        with httpx.Client(timeout=self.settings.timeout, follow_redirects=True, transport=self._transport,
                          headers={"User-Agent": USER_AGENT}) as client:
            for name in chain:
                self.stats["web_requests"] += 1
                try:
                    result = PROVIDERS[name](self.settings, client).search(query)
                except SearchError as exc:
                    errors.append(str(exc))
                    continue
                except Exception as exc:  # a provider changed its format: never crash the brain
                    errors.append(f"{name}: {type(exc).__name__}")
                    continue
                if result.ok:
                    return result
                errors.append(f"{name}: {result.error}")
        return LookupResult(False, query, source=chain[0], error="; ".join(errors) or "no answer")

    # ------------------------------------------------------------- mobs
    def facts_for(self, kinds: list[str], server_version: str | None = None) -> list[str]:
        """One line per mob type: the mob database, else what an earlier lookup found.

        Mineflayer names mobs newer than itself "unknown"; those get a hint listing the newer mobs
        the server version can have (e.g. the sulfur cube on 26.2+).
        """
        db = load_mobs()
        lines: list[str] = []
        for kind in kinds:
            kind = str(kind or "").lower().removeprefix("minecraft:")
            if not kind or kind == "player":
                continue
            mob = db.get(kind)
            if mob is not None:
                lines.append(db.prompt_line(mob))
                continue
            if kind in UNKNOWN_KINDS:
                newer = db.newer_than_mineflayer(server_version)
                if newer:
                    lines.append(f"{kind} creature: probably a mob newer than mineflayer - "
                                 + "; ".join(db.prompt_line(m) for m in newer[:3]))
                continue
            fact = mob_fact(kind)
            if fact is None:
                hit = self.cached(f"mob {kind}")
                fact = hit.summary(220) if hit else None
            if fact:
                lines.append(f"{kind.replace('_', ' ')}: {fact}")
        return lines

    def prefetch_mob(self, kind: str) -> bool:
        """Look an unknown mob (e.g. from a mod) up in the background. True when a lookup started."""
        kind = str(kind or "").lower().split(":", 1)[-1]
        if (not self.settings.enabled or not self.settings.auto_lookup_mobs or not kind or kind in MOB_FACTS
                or kind in ("player", "item", "experience_orb", "arrow", "falling_block", "tnt", "boat", "minecart")
                or kind in UNKNOWN_KINDS or load_mobs().get(kind) is not None
                or self.cached(f"mob {kind}") is not None):
            return False
        with self._lock:
            if kind in self._inflight:
                return False
            self._inflight.add(kind)

        def work() -> None:
            try:
                self.lookup(f"mob {kind}", record=False)
            finally:
                with self._lock:
                    self._inflight.discard(kind)

        threading.Thread(target=work, name=f"lookup-{kind}", daemon=True).start()
        return True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "provider": self.settings.provider,
            "fallback": self.settings.fallback,
            "cached_answers": len(self._cache),
            "requests_last_hour": self.requests_last_hour(),
            "max_per_hour": self.settings.max_per_hour,
            "brave_key_present": bool(os.environ.get(self.settings.brave_api_key_env or "", "").strip()),
            "stats": dict(self.stats),
            "recent": [r.to_dict() for r in list(self.recent)[-6:]],
            "built_in_mobs": len(MOB_FACTS),
            "mob_database": {"mobs": len(load_mobs()), "enabled": self.settings.use_mob_database,
                             "mineflayer_max_version": load_mobs().mineflayer_max},
        }


_SHARED: dict[str, KnowledgeBase] = {}
_SHARED_LOCK = threading.Lock()


def shared_knowledge(settings: SearchSettings, base_dir: Path | None = None) -> KnowledgeBase:
    """The process-wide knowledge base for this cache file (settings are refreshed on every call)."""
    cache = Path(settings.cache_file).expanduser() if settings.cache_file else None
    if cache is not None and not cache.is_absolute():
        cache = (base_dir or Path.cwd()) / cache
    key = str(cache or "")
    with _SHARED_LOCK:
        kb = _SHARED.get(key)
        if kb is None:
            kb = _SHARED[key] = KnowledgeBase(settings, cache)
        kb.settings = settings
        return kb
