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
import threading
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
#: Windows (your own PC, ``scripts/collect_dataset_pc.bat``): llama.cpp's Windows builds, no fork, no /proc.
IS_WINDOWS = os.name == "nt"
#: A claim (``meta/claim-NN.json``) older than this is from a session that died: its number is free again.
CLAIM_STALE_HOURS = 3.0


def server_name() -> str:
    return "llama-server.exe" if IS_WINDOWS else "llama-server"


def _lib_env(binary: Path) -> dict[str, str]:
    """The environment for llama.cpp: its own libraries / DLLs (next to the binary) found first."""
    folder = str(Path(binary).parent)
    if IS_WINDOWS:
        return {**os.environ, "PATH": f"{folder};{os.environ.get('PATH', '')}"}
    return {**os.environ, "LD_LIBRARY_PATH": f"{folder}:{os.environ.get('LD_LIBRARY_PATH', '')}"}

#: The Qwen3 GGUFs it can run on Kaggle's CPU, biggest first (Apache-2.0: the dataset may be used commercially).
#: ``ram_gb`` = what it needs in RAM (the GGUF + 4 parallel 4K contexts + llama.cpp). Qwen3-30B-A3B is a mixture of experts: 30B weights in RAM, but only ~3B work per token, so on a CPU it
#: runs about as fast as the 4B and judges far better - it fits Kaggle's 30 GB.
MODELS = {
    # CPU: Q5_K_M (21.7 GB, ~7x less precision loss than Q4_K_M) when ~26 GB RAM is free, else Q4_K_M (18.6 GB)
    "qwen3-30b-a3b": {"repo": "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "quant": "Q4_K_M",
                      "best_quant": "Q5_K_M", "best_ram_gb": 26,
                      "name": "Qwen3-30B-A3B-Instruct-2507", "hybrid": False, "license": "Apache-2.0", "ram_gb": 23,
                      "gpu_quant": "Q5_K_M", "gpu_vram_gb": 27},
    "qwen3.5-9b": {"repo": "unsloth/Qwen3.5-9B-GGUF", "quant": "Q4_K_M", "name": "Qwen3.5-9B", "hybrid": True,
                   "license": "Apache-2.0", "ram_gb": 7, "gpu_quant": "UD-Q4_K_XL", "gpu_vram_gb": 8},
    # the fast one for the GPU: as strong as the 30B-A3B on benchmarks, ~4x smaller, fits ONE T4 (so 2 x T4 = 2 copies)
    "qwen3.5-4b": {"repo": "unsloth/Qwen3.5-4B-GGUF", "quant": "Q4_K_M", "name": "Qwen3.5-4B", "hybrid": True,
                   "license": "Apache-2.0", "ram_gb": 4.5, "gpu_quant": "UD-Q4_K_XL", "gpu_vram_gb": 4},
    "qwen3-4b": {"repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "quant": "Q4_K_M", "name": "Qwen3-4B-Instruct-2507",
                 "hybrid": False, "license": "Apache-2.0", "ram_gb": 4, "gpu_quant": "Q8_0", "gpu_vram_gb": 7},
    "qwen3-1.7b": {"repo": "unsloth/Qwen3-1.7B-GGUF", "quant": "Q4_K_M", "name": "Qwen3-1.7B", "hybrid": True,
                   "license": "Apache-2.0", "ram_gb": 2.5, "gpu_quant": "Q8_0", "gpu_vram_gb": 4},
}


def available_ram_gb() -> float:
    """RAM free for the model (``MemAvailable``; Kaggle gives 30 GB)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        pass
    if IS_WINDOWS:
        try:
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("free", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_free", ctypes.c_ulonglong), ("virtual_total", ctypes.c_ulonglong),
                            ("virtual_free", ctypes.c_ulonglong), ("extended", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                return status.free / 2**30
        except (OSError, AttributeError, ValueError):
            pass
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):
        return 0.0


#: ``MODEL = "auto"`` on a GPU: small and strong, ~60-90 tok/s per request on a T4, one copy per GPU.
GPU_MODEL = "qwen3.5-4b"


def gpu_info(run: Callable[..., Any] = subprocess.run) -> list[dict[str, Any]]:
    """The NVIDIA GPUs (``[{"name", "vram_gb", "arch"}]``, e.g. 2 x Tesla T4 = arch "75"), ``[]`` without one."""
    try:
        done = run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in (done.stdout or "").splitlines() if done.returncode == 0 else []:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[1].replace(".", "").isdigit():
            gpus.append({"name": parts[0], "vram_gb": float(parts[1]) / 1024, "arch": parts[2].replace(".", "")})
    return gpus


def gpu_quant(key: str, vram_gb: float) -> str:
    """The GGUF quant for ``key`` on GPUs with ``vram_gb`` in total: the better one when it fits (2 x T4: the 30B in
    Q5_K_M), else the CPU one (llama.cpp then keeps what does not fit in RAM)."""
    spec = MODELS[key]
    return spec.get("gpu_quant", spec["quant"]) if vram_gb >= spec.get("gpu_vram_gb", 1e9) else spec["quant"]


def cpu_quant(key: str, ram_gb: float, choice: str = "auto") -> str:
    """The GGUF quant on the CPU: ``QUANT`` when set, else the better one (``best_quant``: the 30B in Q5_K_M) when
    the free RAM holds it, else the normal one (Q4_K_M)."""
    spec = MODELS[key]
    if choice and choice != "auto":
        return choice
    return spec["best_quant"] if spec.get("best_quant") and ram_gb >= spec.get("best_ram_gb", 1e9) else spec["quant"]


def cpu_cores() -> tuple[int, int]:
    """``(physical cores, logical threads)``. Kaggle's 4 "CPUs" can be 2 cores with 2 hyper-threads each."""
    logical = max(1, os.cpu_count() or 4)
    try:
        cores, block = set(), {}
        for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
            if not line.strip():
                if "core id" in block:
                    cores.add((block.get("physical id", "0"), block["core id"]))
                block = {}
            elif ":" in line:
                name, value = line.split(":", 1)
                block[name.strip()] = value.strip()
        physical = len(cores)
    except OSError:
        physical = 0
    if not physical:  # Windows / macOS: no /proc/cpuinfo
        try:
            import psutil  # type: ignore[import-not-found]

            physical = int(psutil.cpu_count(logical=False) or 0)
        except Exception:  # noqa: BLE001
            physical = 0
    return (min(physical, logical) if physical else logical), logical


def cpu_plan(physical: int, logical: int, per_copy: int = 8, max_copies: int = 8) -> tuple[int, int, int]:
    """``(llama-servers, threads, prompt threads)`` for the CPU. A normal session (4 threads) = one server. A big
    machine - a Kaggle **TPU** session has ~96+ CPU threads and 330+ GB RAM - runs several servers with ~8 cores
    each: they share ONE copy of the model in RAM (the file is memory-mapped once), and together work on many more
    requests than one server could feed with that many cores."""
    copies = max(1, min(max_copies, physical // per_copy)) if physical >= 2 * per_copy else 1
    return copies, max(1, physical // copies), max(1, logical // copies)


def tpu_session() -> bool:
    """A Kaggle TPU session (Accelerator: TPU): only its CPUs are used here, and it ends after 9 h, not 12 h."""
    import glob

    if (os.environ.get("TPU_NAME") or os.environ.get("TPU_ACCELERATOR_TYPE") or os.environ.get("COLAB_TPU_ADDR")
            or glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*")):
        return True
    # Kaggle's CPU and GPU machines have 4 cores, its TPU VM has 96: on Kaggle, 64+ threads can only be a TPU session
    on_kaggle = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"))
    return on_kaggle and (os.cpu_count() or 0) >= 64


def pick_model(choice: str = "auto", ram_gb: float | None = None) -> tuple[str, str]:
    """``(model key, why)``: ``"auto"`` = the biggest model that fits in the free RAM."""
    ram = available_ram_gb() if ram_gb is None else ram_gb
    if choice != "auto":
        need = MODELS[choice]["ram_gb"]
        return choice, (f"MODEL = {choice!r}" + (f" - WARNING: it needs ~{need} GB RAM and {ram:.0f} GB is free"
                                                 if ram and need > ram else ""))
    for key, spec in MODELS.items():
        if spec["ram_gb"] <= ram:
            return key, f"the biggest that fits in {ram:.0f} GB of free RAM (needs ~{spec['ram_gb']} GB)"
    return "qwen3-1.7b", f"only {ram:.0f} GB of RAM free"

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
            self.parts.append("\n## " if tag in ("h2", "h3") and not skip and not self._skipping() else "\n")
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
        if line and line.strip("# ") and (not lines or lines[-1] != line):
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


#: Sections that are not worth questions (version changelogs, pictures, links, sound files).
SKIP_SECTIONS = {"history", "gallery", "references", "see also", "external links", "navigation", "issues", "video",
                 "videos", "screenshots", "renders", "concept artwork", "sounds", "in other media", "data history",
                 "development images", "textures", "external link", "notes and references",
                 # IDs / block states / technical tables and trivia: the LLM reads them (CPU time) and its pairs from
                 # them are mostly rejected or worthless ("What is the numeric ID of ...?")
                 "data values", "block states", "block data", "entity data", "item data", "id", "ids", "technical",
                 "trivia", "mentions", "translations", "references and notes", "fuel duration", "data"}


def page_sections(text: str, size: int = 2800, limit: int = 0) -> list[str]:
    """The WHOLE page in parts of about ``size`` characters, cut at line ends; short sections share a part, each
    with its ``[heading]``. Without the changelog / gallery / link sections. ``limit`` > 0: only the first ones."""
    parts: list[str] = []
    current, skipping = "", False
    for line in str(text or "").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
            skipping = norm(heading) in SKIP_SECTIONS
            line = f"[{heading}]"
            if skipping:
                continue
        if skipping:
            continue
        if current and len(current) + len(line) > size:
            parts.append(current.strip())
            current = ""
        current += line[:size] + "\n"
    tail = current.strip()
    if tail and not re.fullmatch(r"\[[^\]]*\]", tail):  # the rest (never a lone heading)
        if len(tail) >= 80:
            parts.append(tail)
        elif parts:  # a short last sentence joins the part before it (it used to be dropped)
            parts[-1] += "\n" + tail
    return parts[:limit] if limit > 0 else parts


#: Fandom wiki titles that are not about the game itself: version / snapshot pages, other games, lists of versions.
SKIP_TITLE = re.compile(
    r"(?i)^(java|bedrock|pocket|legacy console|console|education|xbox 360|xbox one|playstation( \d| vita)?|wii u|"
    r"nintendo switch|new nintendo 3ds|pi|minecraft (dungeons|legends|earth|story mode)|minecraftedu) |"
    r"\b\d{2}w\d{2}[a-z]\b|pre-release|release candidate|\(disambiguation\)|\bbeta \d|\balpha v?\d|\binfdev\b|"
    r"\bindev\b|^classic\b|^\d")


def wiki_url(title: str) -> str:
    title = re.sub(r" \((enchantment|status effect|biome)\)$", "", title)
    return "https://minecraft.fandom.com/wiki/" + urllib.parse.quote(title.replace(" ", "_"))


def wiki_titles(get: Callable[..., bytes] = http_get, cache: Path | None = None) -> list[str]:
    """Every article of the Minecraft Fandom wiki (CC BY-SA) about the game: thousands of pages beyond the
    game-data topics. Without redirects, version pages, other games and sub-pages (except tutorials)."""
    if cache is not None and Path(cache).is_file():
        return json.loads(Path(cache).read_text(encoding="utf-8"))
    titles: list[str] = []
    params = {"action": "query", "list": "allpages", "apnamespace": 0, "apfilterredir": "nonredirects",
              "aplimit": 500, "format": "json"}
    for _page in range(200):
        data = json.loads(get("https://minecraft.fandom.com/api.php?" + urllib.parse.urlencode(params), 60,
                              "application/json").decode("utf-8"))
        titles += [str(p.get("title") or "") for p in (data.get("query") or {}).get("allpages") or []]
        more = (data.get("continue") or {}).get("apcontinue")
        if not more:
            break
        params["apcontinue"] = more
    keep = [t for t in titles if t and ":" not in t and not SKIP_TITLE.search(t)
            and ("/" not in t or t.startswith("Tutorials/"))]
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(keep), encoding="utf-8")
    return keep


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
class WebSearch:
    """``web_search`` as an object that can be sent to a worker process."""

    def __init__(self, results: int = 8) -> None:
        self.results = results

    def __call__(self, query: str) -> list[dict]:
        return web_search(query, self.results)


class PageFetcher:
    """``fetch_page`` (with the page cache) as an object that can be sent to a worker process."""

    def __init__(self, cache: Path | None = None) -> None:
        self.cache = cache

    def __call__(self, url: str) -> str:
        return fetch_page(url, self.cache)


def _worker_start() -> None:
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the Kaggle stop button is for the main process (it saves)
    try:
        os.nice(10)  # web search / parsing waits a moment; the LLM (the slow part) gets the CPU first
    except (OSError, AttributeError):  # Windows: no nice - "below normal" priority instead
        try:
            import psutil  # type: ignore[import-not-found]

            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:  # noqa: BLE001, S110 - normal priority then
            pass


def _ready(_index: int = 0) -> int:
    return os.getpid()


def process_pool(workers: int) -> Any:
    """``workers`` processes for the web part (search, download, HTML parsing, picking passages): real parallel
    work outside the main process. Started right away, before any thread exists. ``None`` where fork is missing."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    try:
        context = multiprocessing.get_context("fork")  # the notebook's functions live in __main__: fork, not spawn
    except ValueError:  # Windows: no fork - spawn works when this file runs as a script (your PC), not in Jupyter
        main_file = getattr(sys.modules.get("__main__"), "__file__", "") or ""
        if not main_file or "ipykernel" in sys.modules:
            return None
        context = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=max(1, workers), mp_context=context, initializer=_worker_start)
    pids = set(pool.map(_ready, range(max(1, workers) * 2)))
    log(f"{len(pids)} worker processes for web search and page parsing")
    return pool


def load_minecraft_data(version: str = MINECRAFT_DATA_VERSION, fetch: Callable[[str], Any] | None = None) -> dict:
    fetch = fetch or (lambda rel: http_json(MINECRAFT_DATA_URL + rel))
    paths = (fetch("dataPaths.json").get("pc") or {}).get(version) or {}
    data: dict[str, Any] = {"version": version}
    for name in ("items", "blocks", "entities", "enchantments", "effects", "biomes", "recipes", "foods"):
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


def all_topics(data: dict, tmp: Path, cfg: dict) -> list[str]:
    """The game-data topics (mechanics, mobs, items, blocks...) first, then every other article of the wiki
    (``WIKI_PAGES``): the more pages, the bigger the dataset. Same order in every session."""
    out = topics(data, random.Random(int(cfg.get("seed", 0))))
    if cfg.get("wiki_pages", True):
        try:
            titles = wiki_titles(cache=Path(tmp) / "wiki_titles.json")
        except Exception as exc:  # noqa: BLE001 - the game-data topics alone still work
            log(f"could not list the wiki pages ({type(exc).__name__}: {str(exc)[:160]}) - game-data topics only")
            titles = []
        known = {norm(re.sub(r" \(.*\)$", "", t)) for t in out}
        extra = [t for t in titles if norm(t) not in known]
        log(f"{len(out)} game-data topics + {len(extra)} more wiki pages")
        out += extra
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
                       # the NUMBERS of the source sentences that prove it (not the sentences: far fewer tokens to
                       # write on a CPU, and the evidence is always exactly the page's own words)
                       "evidence": {"type": "array", "minItems": 1, "maxItems": 3,
                                    "items": {"type": "integer", "minimum": 1}},
                       "edition": {"type": "string", "enum": ["java", "bedrock", "both", "unknown"]}},
        "required": ["question", "answer", "evidence", "edition"]}}},
    "required": ["pairs"],
}
VERDICT_SCHEMA = {
    "type": "object",
    # the verdict only: a reason written AFTER it can't change it, and costs ~40 tokens per check on a CPU
    "properties": {"verdict": {"type": "string", "enum": ["SUPPORTED", "CONTRADICTED", "NOT_FOUND"]}},
    "required": ["verdict"],
}
PROPOSE_SYSTEM = (
    "You write question-answer pairs for a Minecraft knowledge dataset. Use ONLY the source text you are given - "
    "never your own memory. Write questions a player would really ask (varied wording: how, what, why, can, how "
    "many). Each answer: 1-4 sentences in your own words, every fact and number taken from the source, nothing "
    "added. Every sentence of the source has a number like [7]: for each pair give the numbers of the 1-3 sentences "
    "that prove the answer (evidence), nothing else. If "
    "the source says the fact is for Java or Bedrock Edition only, set edition and say so in the answer. Skip "
    "anything the source is vague about, history or trivia nobody asks, and facts that depend on an old version. "
    "Return compact JSON only (one line, no indentation)."
)
JUDGE_SYSTEM = (
    "You check a Minecraft answer against evidence text. SUPPORTED: every fact and number in the answer is stated "
    "in the evidence. CONTRADICTED: the evidence says something different. NOT_FOUND: the evidence does not say "
    "it (or only part of it). Be strict: anything extra in the answer that the evidence does not state means "
    "NOT_FOUND. Return compact JSON only (one line, no indentation)."
)


def _allow_mlock() -> None:
    """In llama-server's process, before it starts: lift the locked-memory limit so ``--mlock`` can keep the whole
    model in RAM (works as root, as on Kaggle; elsewhere llama.cpp just warns and runs normally)."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ImportError, ValueError, OSError):
        pass


class LlamaServer:
    """llama.cpp's ``llama-server`` with one GGUF, on CPU; ``chat()`` returns parsed JSON (schema-constrained)."""

    def __init__(self, binary: Path, model: Path, threads: int = 0, ctx: int = 8192, port: int = 8089,
                 hybrid: bool = False, parallel: int = 1, speculate: bool = False, gpu: int | None = None,
                 threads_batch: int = 0) -> None:
        self.binary, self.model, self.port, self.hybrid = Path(binary), Path(model), port, hybrid
        physical, logical = cpu_cores()
        # writing a token is memory-bound: one thread per physical core (hyper-threads only fight over the cache);
        # reading the prompt is compute-bound: every thread
        self.threads = threads or physical
        self.threads_batch = threads_batch or max(self.threads, logical)
        self.ctx = ctx  # per slot
        self.parallel = max(1, int(parallel))  # slots: requests the server works on together (continuous batching)
        self.gpu = gpu  # only this GPU (one copy of a small model per GPU), None = all / the CPU
        self.speculate = speculate  # n-gram speculation: guesses tokens it has seen (the quotes) and checks many at once
        self.process: subprocess.Popen | None = None
        self.calls = 0
        self.seconds = 0.0
        self.tokens = {"prompt": 0, "prompt_ms": 0.0, "cached": 0, "out": 0, "out_ms": 0.0}
        self._lock = threading.Lock()

    def start(self, wait: float = 900) -> "LlamaServer":
        env = _lib_env(self.binary)
        if self.gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        command = self.command()
        logfile = open(self.log_path(), "w")  # noqa: SIM115 - lives as long as the server
        extra: dict[str, Any] = {} if IS_WINDOWS else {"preexec_fn": _allow_mlock}  # no preexec_fn on Windows
        self.process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, env=env, **extra)
        end = time.time() + wait
        while time.time() < end:
            if self.process.poll() is not None:
                tail = self.log_path().read_text(errors="replace")[-1500:]
                raise RuntimeError(f"llama-server stopped (exit {self.process.returncode}):\n{tail}")
            try:
                if json.loads(http_get(f"http://127.0.0.1:{self.port}/health", 5)).get("status") == "ok":
                    return self
            except (OSError, ValueError):
                pass
            time.sleep(2)
        raise RuntimeError("llama-server did not become ready")

    def log_path(self) -> Path:
        return self.model.parent / ("llama-server.log" if self.port == 8089 else f"llama-server-{self.port}.log")

    def command(self) -> list[str]:
        command = [str(self.binary), "-m", str(self.model), "-c", str(self.ctx * self.parallel), "-t",
                   str(self.threads), "-tb", str(self.threads_batch), "--host", "127.0.0.1", "--port", str(self.port),
                   "--jinja", "-np", str(self.parallel), "--no-webui",
                   # none of these change a single output token:
                   "--mlock",  # the model stays in RAM (never paged out and re-read from disk when Python needs memory)
                   "--prio", "2", "--prio-batch", "2",  # llama.cpp's threads before the web workers
                   "--cache-reuse", "256"]  # reuse the cached part of a prompt even when it moved (same page, new ask)
        if self.speculate:  # no draft model: the guesses come from the prompt and what it wrote (the quotes, JSON keys)
            command += ["--spec-type", "ngram-simple", "--spec-ngram-simple-size-n", "8",
                        "--spec-ngram-simple-size-m", "32", "--spec-draft-n-max", "32"]
        return command

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
        timings = reply.get("timings") or {}
        with self._lock:
            self.calls += 1
            self.seconds += time.time() - started
            for key, field in (("prompt", "prompt_n"), ("prompt_ms", "prompt_ms"), ("cached", "cache_n"),
                               ("out", "predicted_n"), ("out_ms", "predicted_ms")):
                self.tokens[key] += float(timings.get(field) or 0)
        return parse_json(((reply.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


class LLMPool:
    """Several llama-servers used as one (a copy of a small model on each GPU): every request goes to the least busy
    one, so both T4s work at the same time instead of taking turns."""

    def __init__(self, servers: list[Any]) -> None:
        self.servers = servers
        self.parallel = sum(int(getattr(s, "parallel", 1)) for s in servers)
        self._busy = [0] * len(servers)
        self._lock = threading.Lock()

    @property
    def tokens(self) -> dict[str, float]:
        total: dict[str, float] = {}
        for server in self.servers:
            for key, value in server.tokens.items():
                total[key] = total.get(key, 0) + value
        return total

    @property
    def calls(self) -> int:
        return sum(s.calls for s in self.servers)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.servers)

    def chat(self, system: str, user: str, schema: dict, max_tokens: int = 700) -> Any:
        with self._lock:
            index = min(range(len(self.servers)), key=lambda i: self._busy[i])
            self._busy[index] += 1
        try:
            return self.servers[index].chat(system, user, schema, max_tokens)
        finally:
            with self._lock:
                self._busy[index] -= 1

    def stop(self) -> None:
        for server in self.servers:
            server.stop()


def llm_speed(llm: Any) -> str:
    """``"reads 45 tok/s, writes 8.1 tok/s"`` from llama-server's timings (per request), or ``""``."""
    tokens = getattr(llm, "tokens", None)
    if not tokens or not tokens.get("out_ms"):
        return ""
    read = tokens["prompt"] / max(tokens["prompt_ms"] / 1000, 1e-6)
    cached = tokens["cached"] / max(tokens["cached"] + tokens["prompt"], 1)
    reads = f"LLM reads {read:.0f} tok/s ({cached:.0%} from cache), " if tokens.get("prompt_ms") else "LLM "
    return (f"{reads}writes {tokens['out'] / (tokens['out_ms'] / 1000):.1f} tok/s per request x "
            f"{getattr(llm, 'parallel', 1)} at once")


#: The speed test: the real job (write question-answer pairs with quotes) on a short text, all slots at once.
BENCH_TOPIC = "Lead"
BENCH_URL = "https://minecraft.fandom.com/wiki/Lead"
BENCH_PASSAGE = (
    "A lead is an item that ties a passive mob to the player or to a fence. Leads are crafted from 5 string, and "
    "one craft gives 2 leads. A mob on a lead follows the player; if the player moves more than 10 blocks away, the "
    "lead breaks and drops as an item. A lead can be tied to a fence post by using it on the fence, which leaves a "
    "lead knot. Wandering traders arrive with their llamas on leads. Hostile mobs such as zombies and creepers "
    "cannot be put on a lead, but zoglins and hoglins can. Boats can be pulled with a lead in recent versions."
)


def benchmark(llm: Any) -> float:
    """Tokens/s the LLM writes with all its slots busy (the real job, so the n-gram speculation counts too).
    0 when the replies are broken."""
    from concurrent.futures import ThreadPoolExecutor

    parallel = int(getattr(llm, "parallel", 1))
    llm.chat(JUDGE_SYSTEM, "QUESTION: q?\nANSWER: a\n\nEVIDENCE:\n<<<\na\n>>>", VERDICT_SCHEMA, max_tokens=8)  # warm-up
    before = dict(llm.tokens)
    with ThreadPoolExecutor(parallel) as pool:
        results = list(pool.map(lambda _i: propose(llm, BENCH_TOPIC, BENCH_URL, BENCH_PASSAGE, 2), range(parallel)))
    if not any(results):
        return 0.0
    written = llm.tokens["out"] - before["out"]
    seconds = (llm.tokens["out_ms"] - before["out_ms"]) / 1000 / parallel  # the requests ran side by side
    return written / max(seconds, 1e-6)


def fast_llm(keys: list[str], start: Callable[[str, bool], Any], target: float,
             speculate: str = "auto", step_down: bool = True) -> tuple[Any, str, float, bool]:
    """``(server, model key, tokens/s, speculating)``: the biggest of ``keys`` (biggest first) that writes at least
    ``target`` tokens/s here, measured; with ``speculate="auto"`` with and without the n-gram speculation (the
    faster wins). When none is fast enough: the last (smallest) one. ``start(key, speculate)`` = a started server.
    ``step_down=False``: when the first model does not work at all, raise (the build is broken, not the model)."""
    modes = {"auto": [True, False], "on": [True], "off": [False]}.get(speculate, [True, False])
    for number, key in enumerate(keys, 1):
        best: tuple[float, Any, bool] | None = None
        for spec in modes:
            try:
                server = start(key, spec)
            except Exception as exc:  # noqa: BLE001 - e.g. an older llama.cpp without the flag: try the next way
                log(f"{MODELS[key]['name']}{' + speculation' if spec else ''} did not start "
                    f"({type(exc).__name__}: {str(exc)[:200]})")
                continue
            try:
                speed = benchmark(server)
            except Exception as exc:  # noqa: BLE001
                log(f"speed test failed ({type(exc).__name__}: {str(exc)[:200]})")
                speed = 0.0
            log(f"speed test: {MODELS[key]['name']}{' + n-gram speculation' if spec else ''} writes {speed:.1f} "
                f"tok/s ({getattr(server, 'parallel', 1)} requests at once; target {target:g})")
            if speed <= 0:  # no usable replies: broken, not slow
                server.stop()
                continue
            if best is not None and best[0] >= speed:
                server.stop()
                continue
            if best is not None:
                best[1].stop()
            best = (speed, server, spec)
            if target > 0 and speed >= target:
                break
        if best is None:
            if not step_down:
                break
            continue
        if target <= 0 or best[0] >= target or number == len(keys):
            return best[1], key, best[0], best[2]
        log(f"{MODELS[key]['name']} is below {target:g} tok/s - trying a smaller model")
        best[1].stop()
    raise RuntimeError("no LLM could start (see llama-server.log in the models folder)")


def _runs(binary: Path) -> bool:
    env = _lib_env(binary)
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


def _cuda_devices(binary: Path) -> int:
    """How many CUDA GPUs this llama-server sees (0 = the CUDA part cannot load here)."""
    env = _lib_env(binary)
    try:
        done = subprocess.run([str(binary), "--list-devices"], capture_output=True, text=True, env=env, timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    return len(re.findall(r"(?m)^\s*CUDA\d+\s*:", done.stdout + done.stderr))


def llama_server_binary(tools: Path, fetch_json: Callable[[str], Any] = http_json,
                        download: Callable[[str, Path], Path] = _download, cuda_arch: str = "",
                        prebuilt: bool = True) -> Path:
    """``llama-server``: one already in ``tools``, else the newest llama.cpp Linux x64 build, else compiled from
    that release's source when the prebuilt one cannot start here (~10 min on 4 cores for the CPU).
    ``cuda_arch`` (``"75"`` = Tesla T4): the CUDA build, which must see the GPU; compiled for exactly that GPU with
    Kaggle's nvcc (~20-30 min) when the prebuilt one cannot. ``prebuilt=False``: compile (the prebuilt one started
    but failed on the GPU)."""
    tools = Path(tools) / "cuda" if cuda_arch else Path(tools)
    kind = "CUDA" if cuda_arch else "CPU"

    def usable(path: Path) -> bool:
        return path.is_file() and _runs(path) and (not cuda_arch or _cuda_devices(path) > 0)

    exe = server_name()
    for found in sorted(tools.rglob(exe)) if tools.is_dir() else []:
        if (prebuilt or "build" in found.parts) and usable(found):
            return found
    if IS_WINDOWS and not prebuilt:
        raise RuntimeError("compiling llama.cpp is not done on Windows - update the NVIDIA driver (CUDA 12.4+)")
    if IS_WINDOWS:  # CUDA 12.4: runs with any NVIDIA driver from 2024 on (RTX 20/30/40/50, GTX 16)
        pattern = r"win-cuda-12\.\d+-x64\.zip" if cuda_arch else r"win-cpu-x64\.zip"
    else:
        pattern = r"ubuntu-cuda-12\.\d+-x64\.tar\.gz" if cuda_arch else r"ubuntu-x64\.(tar\.gz|zip)"
    releases = fetch_json(f"https://api.github.com/repos/{LLAMA_CPP_REPO}/releases?per_page=15")
    for release in releases:
        tag = str(release.get("tag_name") or "")
        assets = release.get("assets") or []
        asset = next((a for a in assets if re.fullmatch(rf"llama-{re.escape(tag)}-bin-{pattern}", str(a.get("name")))),
                     None)
        if not asset:
            continue
        folder = tools / f"llama.cpp-{tag}"
        if prebuilt:
            log(f"llama.cpp {tag}: downloading the {kind} build")
            _extract(download(asset["browser_download_url"], folder / asset["name"]), folder / "bin")
            binary = next((p for p in sorted((folder / "bin").rglob(exe)) if p.is_file()), None)
            # the CUDA runtime: "cudart-<asset>" (Linux) or the same without the tag (Windows)
            runtimes = {"cudart-" + str(asset["name"]), "cudart-" + str(asset["name"]).replace(f"-{tag}-", "-")}
            runtime = next((a for a in assets if a.get("name") in runtimes), None)
            if binary and cuda_arch and runtime:  # the CUDA runtime libraries, next to the binary
                _extract(download(runtime["browser_download_url"], folder / runtime["name"]), folder / "cudart")
                for lib in [*(folder / "cudart").rglob("*.so*"), *(folder / "cudart").rglob("*.dll")]:
                    shutil.copy2(lib, binary.parent / lib.name)
            if binary:
                binary.chmod(0o755)
                if usable(binary):
                    return binary
            log(f"the prebuilt {kind} llama-server does not work here - compiling it from source "
                f"({'~20-30 min, for this GPU' if cuda_arch else '~10 min'})")
        src = folder / "src"
        if not src.is_dir():
            _extract(download(f"https://codeload.github.com/{LLAMA_CPP_REPO}/tar.gz/refs/tags/{tag}",
                              folder / "src.tar.gz"), src)
        top = next(p.parent for p in src.rglob("CMakeLists.txt") if (p.parent / "tools").is_dir() or
                   (p.parent / "examples").is_dir())
        flags = ["-DGGML_NATIVE=ON", "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"]
        if cuda_arch:
            flags += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
            if Path("/usr/local/cuda/bin/nvcc").is_file():
                flags.append("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
        subprocess.run(["cmake", "-S", str(top), "-B", str(top / "build"), *flags], check=True)
        subprocess.run(["cmake", "--build", str(top / "build"), "--target", "llama-server", "-j",
                        str(os.cpu_count() or 4)], check=True)
        binary = next(p for p in sorted((top / "build").rglob("llama-server")) if p.is_file())
        if cuda_arch and not usable(binary):
            raise RuntimeError("the llama-server compiled with CUDA does not see the GPU")
        return binary
    raise RuntimeError(f"no llama.cpp {'Windows' if IS_WINDOWS else 'Linux'} x64 {kind} release found")


def model_file(spec: dict, folder: Path, token: str = "", quant: str = "") -> Path:
    """The GGUF of ``spec`` (downloaded once into ``folder``); ``quant`` = another quant than the CPU one, used
    when the repo has it."""
    from huggingface_hub import HfApi, hf_hub_download  # type: ignore[import-not-found]

    files = HfApi(token=token or None).list_repo_files(spec["repo"])
    wanted = [f for f in files if f.endswith(f"{quant}.gguf") and "/" not in f] if quant else []
    wanted = wanted or [f for f in files if f.endswith(f"{spec['quant']}.gguf") and "/" not in f]
    if not wanted:
        raise RuntimeError(f"{spec['repo']} has no {spec['quant']} GGUF")
    return Path(hf_hub_download(spec["repo"], wanted[0], local_dir=str(folder), token=token or None))


# ---------------------------------------------------------------------------- the pipeline
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def numbered(passage: str) -> tuple[str, list[str]]:
    """``(text, sentences)``: the passage with every sentence numbered ``[1] ... [2] ...`` (headings become
    ``## heading``). The LLM names its evidence by number instead of copying it out."""
    sentences: list[str] = []
    lines: list[str] = []
    for line in str(passage or "").splitlines():
        line = line.strip()
        if not line:
            continue
        heading = re.fullmatch(r"\[([^\]]+)\]", line)
        if heading:
            lines.append(f"## {heading.group(1)}")
            continue
        out = []
        for piece in _SENTENCE_END.split(line):
            if piece.strip():
                sentences.append(piece.strip())
                out.append(f"[{len(sentences)}] {piece.strip()}")
        lines.append(" ".join(out))
    return "\n".join(lines), sentences


def _evidence(items: Any, sentences: list[str]) -> list[str]:
    """Sentence numbers -> the sentences (a quote, from an older reply, is kept as it is: it is checked later)."""
    out: list[str] = []
    for item in items if isinstance(items, list) else []:
        text = str(item).strip().strip("[]").strip()
        if isinstance(item, (int, float)) or text.isdigit():
            number = int(float(text))
            if 1 <= number <= len(sentences):
                text = sentences[number - 1]
            else:
                continue  # a number that is not in the text: no evidence
        text = " ".join(text.split())
        if text and text not in out:
            out.append(text)
    return out[:3]


def propose(llm: Any, topic: str, url: str, passage: str, k: int = 3) -> list[dict[str, Any]]:
    text, sentences = numbered(passage)
    user = (f"Topic: {topic}\nSource: {url}\n\nSOURCE TEXT (sentences numbered):\n<<<\n{text}\n>>>\n\n"
            f"Write up to {k} question-answer pairs about {topic} that this text fully answers.")
    reply = llm.chat(PROPOSE_SYSTEM, user, PAIR_SCHEMA, max_tokens=200 * k + 100) or {}
    pairs = reply.get("pairs") if isinstance(reply, dict) else None
    out = []
    for pair in pairs or []:
        if not isinstance(pair, dict):
            continue
        question = " ".join(str(pair.get("question") or "").split())
        answer = " ".join(str(pair.get("answer") or "").split())
        evidence = _evidence(pair.get("evidence"), sentences)
        if question and answer and evidence:
            out.append({"question": question, "answer": answer, "evidence": evidence[:3],
                        "edition": str(pair.get("edition") or "unknown")})
    return out[:k]


def judge(llm: Any, question: str, answer: str, evidence: str) -> tuple[str, str]:
    user = f"QUESTION: {question}\nANSWER: {answer}\n\nEVIDENCE:\n<<<\n{evidence}\n>>>"
    reply = llm.chat(JUDGE_SYSTEM, user, VERDICT_SCHEMA, max_tokens=16) or {}
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
    status, note = data_check(question, answer, book)  # free: before the LLM call, not after it
    checks["game_data"] = status
    if status == "contradicts":
        return False, checks, note
    verdict, reason = judge(llm, question, answer, "\n".join(pair["evidence"]))
    checks["second_look"] = verdict
    if verdict != "SUPPORTED":
        return False, checks, f"second look: {verdict}" + (f" ({reason[:120]})" if reason else "")
    return True, checks, ""


def about_it(pair: dict, text: str, share: float = 0.3) -> bool:
    """Does this passage of another page even talk about the pair (30% of its words, by their first 4 letters)?
    When not, the LLM is not asked - it could only say NOT_FOUND. A page that CONTRADICTS the answer talks about
    the same things, so it is still checked."""
    words = {w[:4] for w in norm(pair["question"] + " " + pair["answer"]).split()
             if len(w) > 3 and w not in _STOP and not w.isdigit()}
    if not words:
        return True
    have = {w[:4] for w in norm(text).split() if len(w) > 3}
    return len(words & have) / len(words) >= share


def corroborate(llm: Any, pair: dict, search: Callable[[str], list[dict]], fetch: Callable[[str], str],
                skip: set[str], pages: int = 3) -> tuple[str, str]:
    """``(verdict, url)`` from an independent page: another site, or an official one, or another wiki page."""
    query = pair["question"] if "minecraft" in pair["question"].lower() else pair["question"] + " Minecraft"
    for hit in search(query)[:pages]:
        url = hit["url"]
        if url in skip:
            continue
        try:
            text = passages(fetch(url), pair["question"] + " " + pair["answer"], size=1200, top=1)
        except Exception:  # noqa: BLE001 - one page failing is fine
            continue
        if not text or not about_it(pair, text):
            continue
        verdict, _reason = judge(llm, pair["question"], pair["answer"], text)
        if verdict in ("SUPPORTED", "CONTRADICTED"):
            return verdict, url
    return "NOT_FOUND", ""


def question_key(question: str) -> str:
    return " ".join(sorted(set(query_terms(question))))


# ---------------------------------------------------------------------------- game-data rows (no LLM)
GAME_DATA_SOURCE = "https://github.com/PrismarineJS/minecraft-data/blob/master/data/pc/{version}/{file}.json"
ROMAN = ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X")
TOOLS = ("pickaxe", "axe", "shovel", "hoe", "sword", "shears")


def _pick(rng: random.Random, *options: str) -> str:
    return options[rng.randrange(len(options))]


def _a(word: str) -> str:
    return ("an " if word[:1].upper() in "AEIOU" else "a ") + word


def _list(names: list[str], word: str = "or") -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" {word} " + names[-1]


def game_data_rows(data: dict, book: dict[str, dict[str, Any]] | None = None, seed: int = 0) -> list[dict]:
    """Exact question/answer rows written straight from the game's data (minecraft-data, MIT) by templates - NO
    LLM, so thousands of rows in a second on any CPU, and nothing left to verify: crafting recipes, stack sizes,
    durability and repairs, block hardness / blast resistance / the tool that mines it, hunger restored by food,
    enchantment levels and conflicts, mob hitbox sizes, biome dimensions. ``trust`` = high, ``source`` =
    game-data. Only facts minecraft-data gets right are used (not its food saturation, not rain vs snow)."""
    rng = random.Random(seed)
    version = str(data.get("version") or MINECRAFT_DATA_VERSION)
    items = {i.get("id"): i for i in data.get("items") or [] if isinstance(i, dict)}
    by_name = {i.get("name"): i for i in items.values()}
    rows: list[dict] = []
    seen: set[str] = set()
    created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def add(file: str, topic: str, question: str, answer: str, fact: dict) -> None:
        key = question_key(question)
        if key in seen:
            return
        seen.add(key)
        answer = "In Minecraft Java Edition, " + answer
        rows.append({
            "question": question, "answer": answer,
            "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            "topic": topic, "edition": "java", "trust": "high",
            "sources": [{"url": GAME_DATA_SOURCE.format(version=version, file=file), "license": "MIT (minecraft-data)"}],
            "evidence": [json.dumps(fact, ensure_ascii=False, sort_keys=True)],
            "checks": {"game_data": "source", "minecraft_data": version}, "source": "game-data", "kind": "mcqa",
            "model": "", "created": created,
        })

    # crafting recipes (the same "recipe most variants share" as the recipe check)
    variants: dict[str, int] = {}
    for result, recipes in (data.get("recipes") or {}).items():
        item = items.get(int(result)) if str(result).isdigit() else None
        if item is not None and isinstance(recipes, list):
            variants[_label(item).lower()] = len({_recipe_signature(r) for r in recipes if isinstance(r, dict)})
    for key, recipe in sorted((recipe_book(data) if book is None else book).items()):
        name, count = recipe["name"], int(recipe.get("count") or 1)
        parts = [f"{n} {ing[4:]} (any kind)" if ing.startswith("any ") else f"{n} {ing}"
                 for ing, n in sorted(recipe["ingredients"].items())]
        add("recipes", name, _pick(rng, f"How do you craft {name} in Minecraft?", f"What is the crafting recipe for {name}?",
                                   f"What do I need to craft {name}?"),
            f"{'one recipe for' if variants.get(key, 1) > 1 else 'the crafting recipe for'} {name} uses "
            f"{_list(parts, 'and')}; one craft makes {count}.",
            {"result": name, "count": count, "ingredients": recipe["ingredients"]})

    # items: stack size, durability, repair material
    for item in items.values():
        name, label = str(item.get("name") or ""), _label(item)
        if not name or name == "air" or name.endswith("spawn_egg"):
            continue
        size = int(item.get("stackSize") or 0)
        if size:
            add("items", label, _pick(rng, f"What is the maximum stack size of {label}?",
                                      f"How many {label} fit in one stack?", f"Does {label} stack in Minecraft?"),
                f"{label} does not stack: each one takes a whole inventory slot (stack size 1)." if size == 1 else
                f"{label} stacks up to {size} in one inventory slot.", {"item": name, "stackSize": size})
        if item.get("maxDurability"):
            uses = int(item["maxDurability"])
            add("items", label, _pick(rng, f"How much durability does {_a(label)} have?", f"How many uses does {_a(label)} have?"),
                f"{_a(label)} has {uses} durability.", {"item": name, "maxDurability": uses})
        repair = [_label(by_name[r]) for r in item.get("repairWith") or [] if r in by_name]
        if repair:
            add("items", label, f"What can you repair {_a(label)} with in an anvil?",
                f"{_a(label)} is repaired in an anvil with {_list(repair)} (or by combining it with another {label}).",
                {"item": name, "repairWith": item.get("repairWith")})

    # blocks: hardness and blast resistance, and the tool that mines them
    for block in data.get("blocks") or []:
        if not isinstance(block, dict) or block.get("name") not in by_name or block.get("name") == "air":
            continue  # technical blocks (Moving Piston, Wall Torch, Potted ...) are not items: no questions
        name, label = str(block.get("name") or ""), _label(block)
        hardness, resistance = block.get("hardness"), block.get("resistance")
        if isinstance(hardness, (int, float)) and isinstance(resistance, (int, float)) and block.get("diggable", True):
            add("blocks", label, _pick(rng, f"What are the hardness and blast resistance of {label}?",
                                       f"How strong is {label} against mining and explosions?"),
                f"{label} has a hardness of {hardness:g} and a blast resistance of {resistance:g}"
                + (" - it breaks instantly." if hardness == 0 else "."),
                {"block": name, "hardness": hardness, "resistance": resistance})
        elif hardness is not None and isinstance(hardness, (int, float)) and hardness < 0:
            add("blocks", label, f"Can you break {label} in Survival mode?",
                f"no - {label} is unbreakable in Survival mode.", {"block": name, "hardness": hardness})
        harvest = [items[int(i)] for i in (block.get("harvestTools") or {}) if str(i).isdigit() and int(i) in items]
        mineable = re.findall(r"mineable/(\w+)", str(block.get("material") or ""))
        if harvest:
            tools = [_label(t) for t in sorted(harvest, key=lambda t: t.get("id", 0))]
            pickaxes = [t for t in harvest if str(t.get("name", "")).endswith("_pickaxe")]
            needs = "any pickaxe" if len(pickaxes) == len(harvest) == 6 else f"a {_list(tools)}"
            add("blocks", label, _pick(rng, f"What tool do you need to mine {label}?",
                                       f"What do I need to mine {label} and get its drop?"),
                f"{label} only drops something when it is broken with {needs}; broken by hand or with another tool it "
                f"drops nothing.", {"block": name, "harvestTools": [t.get("name") for t in harvest]})
        elif len(mineable) == 1 and mineable[0] in TOOLS and isinstance(hardness, (int, float)) and hardness > 0:
            add("blocks", label, f"What is the best tool to break {label}?",
                f"{label} is mined fastest with {'an' if mineable[0] == 'axe' else 'a'} {mineable[0]}.",
                {"block": name, "material": block.get("material")})

    for food in data.get("foods") or []:
        if isinstance(food, dict) and food.get("foodPoints"):
            label, points = _label(food), float(food["foodPoints"])
            add("foods", label, _pick(rng, f"How much hunger does {label} restore?",
                                      f"How many hunger points does eating {label} give?"),
                f"eating {label} restores {points:g} hunger points ({points / 2:g} of the 10 drumsticks on the hunger "
                f"bar).", {"food": food.get("name"), "foodPoints": points})

    names = {e.get("name"): _label(e) for e in data.get("enchantments") or [] if isinstance(e, dict)}
    for ench in data.get("enchantments") or []:
        if not isinstance(ench, dict) or not ench.get("maxLevel"):
            continue
        label, level = _label(ench), int(ench["maxLevel"])
        add("enchantments", label + " (enchantment)", f"What is the maximum level of {label}?",
            f"{label} has only one level (no II or higher)." if level == 1 else
            f"{label} goes up to level {ROMAN[level] if level < len(ROMAN) else level}.", {"enchantment": ench.get("name"), "maxLevel": level})
        conflicts = [names[x] for x in ench.get("exclude") or [] if x in names]
        if conflicts:
            add("enchantments", label + " (enchantment)", f"Which enchantments can't be combined with {label}?",
                f"{label} is incompatible with {_list(conflicts)}: they cannot be on the same item.",
                {"enchantment": ench.get("name"), "exclude": ench.get("exclude")})
        if ench.get("treasureOnly"):
            add("enchantments", label + " (enchantment)", f"Can you get {label} from an enchanting table?",
                f"no - {label} is a treasure enchantment{' (a curse)' if ench.get('curse') else ''}: the enchanting "
                f"table never gives it.", {"enchantment": ench.get("name"), "treasureOnly": True})

    for mob in data.get("entities") or []:
        if isinstance(mob, dict) and mob.get("type") in ("hostile", "animal", "passive", "water_creature", "ambient",
                                                         "mob") and mob.get("width") and mob.get("height"):
            label = _label(mob)
            add("entities", label, _pick(rng, f"How big is {_a(label)} in Minecraft?", f"How tall is {_a(label)}?"),
                f"{_a(label)}'s hitbox is {float(mob['width']):g} blocks wide and {float(mob['height']):g} blocks tall.",
                {"entity": mob.get("name"), "width": mob["width"], "height": mob["height"]})

    places = {"overworld": "the Overworld", "nether": "the Nether", "end": "the End"}
    for biome in data.get("biomes") or []:
        if not isinstance(biome, dict) or biome.get("name") == "the_void" or biome.get("dimension") not in places:
            continue
        label = _label(biome)
        add("biomes", label + " (biome)", f"Which dimension is the {label} biome in?",
            f"the {label} biome is in {places[biome['dimension']]}.",
            {"biome": biome.get("name"), "dimension": biome["dimension"]})
        if biome.get("has_precipitation") is False:
            add("biomes", label + " (biome)", f"Does it rain in the {label} biome?",
                f"no - it never rains or snows in the {label} biome.",
                {"biome": biome.get("name"), "has_precipitation": False})
    return rows


class Collector:
    """The dataset on disk (``/kaggle/working/Minecraftism-dataset``): rows, rejects, progress, card. Each Kaggle
    session writes only its own files (``data/train-01.jsonl``, ...; ``session`` = the ``SESSION`` setting), so
    several sessions can run at the same time without overwriting each other; the rows and topics of the other
    sessions are read too, so nothing is asked or done twice."""

    LEGACY = {"data/train.jsonl": "data/train-{}.jsonl", "data/high/train.jsonl": "data/high/train-{}.jsonl",
              "meta/rejected.jsonl": "meta/rejected-{}.jsonl", "meta/progress.json": "meta/progress-{}.json"}

    def __init__(self, folder: Path, model: str, session: int = 1) -> None:
        self.folder, self.model = Path(folder), model
        self.tag = f"{max(1, int(session)):02d}"
        self.rows: list[dict] = []
        self.rejected: list[dict] = []
        self.done: list[str] = []
        self.done_elsewhere: set[str] = set()  # topics other sessions finished
        self.pages: set[str] = set()  # pages already read (two topics can lead to the same page)
        self.keys: set[str] = set()
        self.other_rows = self.other_high = 0
        self.started = time.time()
        self.run = f"{os.getpid()}-{time.time_ns()}"  # this run's id in its claim (meta/claim-NN.json)
        self.ended = False  # True for the last save: the claim says the number is free again
        self.load()

    def path(self, name: str) -> Path:
        return self.folder / self.LEGACY[name].format(self.tag)

    def load(self) -> None:
        if self.tag == "01":  # a dataset from before the sessions: it becomes session 01's
            for old, new in self.LEGACY.items():
                if (self.folder / old).is_file() and not (self.folder / new.format(self.tag)).is_file():
                    (self.folder / old).replace(self.folder / new.format(self.tag))
        for name, target in (("data/train.jsonl", self.rows), ("meta/rejected.jsonl", self.rejected)):
            path = self.path(name)
            if path.is_file():
                target[:] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.keys = {question_key(r["question"]) for r in self.rows}
        for path in sorted((self.folder / "data").glob("train*.jsonl")):
            if path != self.path("data/train.jsonl"):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        self.keys.add(question_key(row["question"]))
                        self.other_rows += 1
                        self.other_high += row.get("trust") == "high"
        for path in sorted((self.folder / "meta").glob("progress*.json")):
            progress = json.loads(path.read_text(encoding="utf-8"))
            self.pages.update(progress.get("done_pages") or [])
            if path == self.path("meta/progress.json"):
                self.done = list(progress.get("done_topics") or [])
            else:
                self.done_elsewhere.update(progress.get("done_topics") or [])
        self.start_rows = len(self.rows) + self.other_rows

    def finished(self) -> set[str]:
        return set(self.done) | self.done_elsewhere

    def seen(self, question: str) -> bool:
        return question_key(question) in self.keys

    def add(self, row: dict) -> None:
        self.rows.append(row)
        self.keys.add(question_key(row["question"]))

    def save(self) -> None:
        high = [r for r in self.rows if r.get("trust") == "high"]
        for name, items in (("data/train.jsonl", self.rows), ("data/high/train.jsonl", high),
                            ("meta/rejected.jsonl", self.rejected)):
            path = self.path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8")
        (self.folder / self.claim_name).write_text(json.dumps(
            {"run": self.run, "time": time.time(), "ended": self.ended}), encoding="utf-8")
        self.path("meta/progress.json").write_text(json.dumps(
            {"done_topics": self.done, "done_pages": sorted(self.pages), "rows": len(self.rows),
             "rejected": len(self.rejected), "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=1), encoding="utf-8")
        stats = self.stats()
        (self.folder / "README.md").write_text(dataset_card(self.rows, self.rejected, self.model, stats["rows"],
                                                            stats["high"]), encoding="utf-8")

    def own_files(self) -> list[str]:
        return [self.LEGACY[name].format(self.tag) for name in self.LEGACY]

    @property
    def claim_name(self) -> str:
        return f"meta/claim-{self.tag}.json"

    def note_others(self, folder: Path) -> int:
        """Topics and pages the OTHER sessions finished since this one started (their ``meta/progress-NN.json`` in
        ``folder``, fresh from the Hub): skipped from now on. Returns how many topics are new."""
        before = len(self.done_elsewhere)
        for path in sorted((Path(folder) / "meta").glob("progress-*.json")):
            if path.name == Path(self.path("meta/progress.json")).name:
                continue
            try:
                progress = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            self.done_elsewhere.update(progress.get("done_topics") or [])
            self.pages.update(progress.get("done_pages") or [])
        return len(self.done_elsewhere) - before

    def stats(self) -> dict[str, int]:
        """All sessions together (the other sessions as they were when this one started)."""
        high = sum(1 for r in self.rows if r.get("trust") == "high") + self.other_high
        rows = len(self.rows) + self.other_rows
        return {"rows": rows, "high": high, "medium": rows - high, "rejected": len(self.rejected),
                "topics": len(self.finished())}


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


def dataset_card(rows: list[dict], rejected: list[dict], model: str, total: int = -1, total_high: int = -1) -> str:
    """The README. ``total`` / ``total_high``: the rows of all sessions (default: ``rows``)."""
    high = sum(1 for r in rows if r.get("trust") == "high") if total_high < 0 else total_high
    count = len(rows) if total < 0 else total
    sites: dict[str, int] = {}
    licences: dict[str, str] = {}
    for row in rows:
        for source in row.get("sources") or []:
            sites[domain(source["url"])] = sites.get(domain(source["url"]), 0) + 1
            licences.setdefault(domain(source["url"]), SOURCES.get(domain(source["url"]), ("", source.get("license", "?")))[1])
    table = "\n".join(f"| {site} | {licences[site]} | {count} |" for site, count in
                      sorted(sites.items(), key=lambda kv: -kv[1])) or "| - | - | 0 |"
    exact = sum(1 for r in rows if r.get("source") == "game-data")
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
- {size_category(count)}
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.jsonl
- config_name: high
  data_files:
  - split: train
    path: data/high/train-*.jsonl
---

# Minecraftism dataset

Checked Minecraft questions and answers for the **Minecraftism** models of
[PyMC Bot](https://github.com/codero-sus/PyMC_Bot), collected by the notebook
`pynb/Minecraftism_Dataset_Kaggle.ipynb` (Kaggle): web search + an LLM ({model}, on llama.cpp) that may only
write what a page says, and then checks on every answer.

**{count} rows** - {high} `high` trust (confirmed by the game's own data or an independent page), {count - high}
`medium` (one source, passed every check). Rejected pairs (with the reason) are in `meta/rejected-*.jsonl` for
review - they are not training data. Each Kaggle session writes its own `data/train-NN.jsonl`; `load_dataset` reads
them all.

## How an answer gets in

**Game-data rows** (`source: game-data`{f", {exact} in this session's file" if exact else ""}) are written straight from the
game's own data (minecraft-data {MINECRAFT_DATA_VERSION}, MIT) by fixed templates - no LLM, so they are exact:
crafting recipes, stack sizes, durability and repairs, block hardness / blast resistance / the tool that mines it,
hunger restored by food, enchantment levels and conflicts, mob hitbox sizes, biome dimensions. All are `high`.

**Web rows** (`source: web-verified`):

1. Only pages whose licence allows commercial use are read (Minecraft Fandom wiki CC BY-SA 3.0, Wikipedia
   CC BY-SA 4.0). minecraft.wiki (CC BY-NC-SA) is never used. Official Mojang pages only confirm answers;
   only their links are kept.
2. The LLM writes the pair from the page text alone and copies the sentences that prove it (`evidence`).
3. Every evidence quote must really be on the page; every number in the answer must be in the evidence.
4. A second LLM pass sees only question, answer and evidence and must answer SUPPORTED.
5. Crafting answers must match the game's recipes (minecraft-data {MINECRAFT_DATA_VERSION}, MIT).
6. With `CORROBORATE = True`, answers the game data did not already confirm are searched again: an independent page that contradicts the
   answer rejects it, one that confirms it makes it `high`.

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
        return any((Path(folder) / "data").glob("train*.jsonl"))
    except Exception as exc:  # noqa: BLE001 - never overwrite an existing slice after a failed resume
        raise RuntimeError(f"Could not resume from {repo_id} ({type(exc).__name__}); stopping to protect existing "
                           "dataset slices. Check access/network and retry.") from exc


def hub_upload(api: Any, repo_id: str, folder: Path, private: bool, message: str,
               own: list[str] | None = None) -> str:
    """Upload this session's files (``own``) and the card - never the other sessions' copies (they may be newer on
    the Hub). Session 01 also removes the files from before the sessions (they became its own)."""
    hub_make_private(api, repo_id, private)  # every time: it stays private even if someone flipped it meanwhile
    own = own or ["data/train-01.jsonl", "data/high/train-01.jsonl", "meta/rejected-01.jsonl", "meta/progress-01.json"]
    extra = {"delete_patterns": list(Collector.LEGACY)} if "data/train-01.jsonl" in own else {}
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(folder), commit_message=message,
                      allow_patterns=["README.md", *own], **extra)
    return f"https://huggingface.co/datasets/{repo_id}"


def hub_progress(repo_id: str, token: str, target: Path) -> Path:
    """The other sessions' progress, fresh from the Hub, into ``target`` (never into the dataset folder: that would
    overwrite this session's newer files)."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(target), token=token or None,
                      allow_patterns=["meta/progress-*.json"])
    return Path(target)


def sync_scraped_pages(repo_id: str, token: str, target: Path, cache: Path) -> int:
    """Import PC scrape batches without touching dataset rows or trusting archive paths."""
    import zipfile

    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    snapshot_download(repo_id, repo_type="dataset", local_dir=str(target), token=token or None,
                      allow_patterns=["scrape-cache/*.zip"])
    cache.mkdir(parents=True, exist_ok=True)
    added = 0
    for archive in sorted((target / "scrape-cache").glob("*.zip")):
        try:
            with zipfile.ZipFile(archive) as batch:
                for entry in batch.infolist():
                    if not re.fullmatch(r"[0-9a-f]{40}\.txt", entry.filename) or entry.file_size > 10_000_000:
                        continue
                    dest = cache / entry.filename
                    if not dest.exists():
                        data = batch.read(entry)
                        data.decode("utf-8")
                        temp = dest.with_suffix(".scrape-tmp")
                        temp.write_bytes(data)
                        temp.replace(dest)
                        added += 1
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            log(f"skipping invalid scrape batch {archive.name}: {type(exc).__name__}")
    return added


def claim_live(claim: dict | None, now: float) -> bool:
    return bool(claim) and not claim.get("ended") and now - float(claim.get("time", 0)) < CLAIM_STALE_HOURS * 3600


def claim_session(sessions: int, read_claims: Callable[[], dict[int, dict]], write_claim: Callable[[int, dict], None],
                  run: str, wait: float = 20.0, sleep: Callable[[float], None] = time.sleep,
                  rng: random.Random | None = None) -> int:
    """``SESSION = 0`` (auto): the lowest session number no running session holds. It writes its claim, waits and
    reads it back - when two sessions took the same number at the same moment, the later write wins and the other
    one takes the next free number. So 10 notebooks can be started with the SAME settings, no editing in between."""
    rng = rng or random.Random()
    sleep(rng.uniform(0, wait / 2))  # sessions started together spread out a little
    lost: set[int] = set()
    for _ in range(sessions + 3):
        now = time.time()
        claims = read_claims()
        free = [n for n in range(1, sessions + 1) if n not in lost and not claim_live(claims.get(n), now)]
        if not free:
            raise SystemExit(f"all {sessions} sessions are running (SESSIONS = {sessions}): set SESSIONS higher - "
                             "the same number in every notebook and on the PC")
        number = free[0]
        write_claim(number, {"run": run, "time": time.time(), "ended": False})
        sleep(wait + rng.uniform(0, 5))
        if (read_claims().get(number) or {}).get("run") == run:
            return number
        lost.add(number)
        log(f"session {number} was taken by another notebook at the same moment - taking the next free one")
    raise SystemExit("could not get a session number - set SESSION by hand (1 ... SESSIONS)")


def hub_claims(api: Any, repo_id: str, token: str, target: Path) -> tuple[Callable[[], dict[int, dict]],
                                                                              Callable[[int, dict], None]]:
    """``(read_claims, write_claim)`` on the Hub: ``meta/claim-NN.json`` = which run holds session NN, and since when
    (refreshed with every upload; marked ended at the end)."""
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    def read() -> dict[int, dict]:
        claims: dict[int, dict] = {}
        names = api.list_repo_files(repo_id, repo_type="dataset")  # fail closed if claims cannot be checked
        for name in names:
            found = re.fullmatch(r"meta/claim-(\d+)\.json", name)
            if not found:
                continue
            try:
                path = hf_hub_download(repo_id, name, repo_type="dataset", token=token or None,
                                       local_dir=str(target), force_download=True)
                claims[int(found.group(1))] = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - unreadable claims must not be treated as free
                raise RuntimeError(f"Cannot read session claim {name}; stopping to protect dataset slices") from exc
        return claims

    def write(number: int, claim: dict) -> None:
        api.upload_file(path_or_fileobj=json.dumps(claim).encode("utf-8"), path_in_repo=f"meta/claim-{number:02d}.json",
                        repo_id=repo_id, repo_type="dataset", commit_message=f"session {number} started")

    return read, write


def hf_token(secret: str = "HF_TOKEN") -> str:
    """The Hugging Face token: the Kaggle secret, else the ``HF_TOKEN`` environment variable, else the one saved by
    ``hf auth login`` on this computer."""
    token = kaggle_secret(secret)
    if token:
        return token
    try:
        from huggingface_hub import get_token  # type: ignore[import-not-found]

        return str(get_token() or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def session_start() -> float:
    """When this Kaggle session (the kernel) started - its 12 hours count from there."""
    try:
        import psutil  # type: ignore[import-not-found]

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:  # noqa: BLE001
        return time.time()


def gather(topic: str, cfg: dict, search: Callable[[str], list[dict]],
           fetch: Callable[[str], str]) -> list[tuple[str, tuple[str, str], str, str]]:
    """``[(url, policy, page, part)]`` for one topic: its wiki page opened directly (no search needed; a search
    when there is no page of that name, and for ``pages_per_topic`` > 1), read WHOLE in parts of ~2,800 characters.
    The web part, run ahead in the background."""
    found: list[tuple[str, tuple[str, str], str, str]] = []
    urls: list[str] = []
    try:
        direct = wiki_url(topic)
        found_page = fetch(direct)
        urls.append(direct)
    except Exception:  # noqa: BLE001 - no page of that name: search for it
        found_page = ""
    wanted = int(cfg.get("pages_per_topic", 1))
    if len(urls) < wanted:
        for hit in search(f"{topic} site:minecraft.fandom.com"):
            if len(urls) >= wanted:
                break
            policy = source_policy(hit["url"])
            if policy and policy[0] == "text" and hit["url"] not in urls:
                urls.append(hit["url"])
    for url in urls:
        policy = source_policy(url)
        if not policy or policy[0] != "text":
            continue
        try:
            page = found_page if url == urls[0] and found_page else fetch(url)
        except Exception:  # noqa: BLE001 - one page failing is fine
            continue
        for part in page_sections(page, int(cfg.get("part_chars", 2800)), int(cfg.get("parts_per_page", 0))):
            found.append((url, policy, page, part))
    return found


def work_topic(topic: str, found: list, cfg: dict, llm: Any, collector: Collector, book: dict,
               search: Callable[[str], list[dict]], fetch: Callable[[str], str],
               deadline: float) -> tuple[list[dict], list[dict]]:
    """The LLM part of one topic: ``(rows, rejected)``. Only reads ``collector``, so several topics can run at
    once (the main loop adds the rows)."""
    rows: list[dict] = []
    rejected: list[dict] = []
    asked: set[str] = set()
    read: set[str] = set()
    try:
        for url, policy, page, passage in found:
            if time.time() > deadline:
                break
            page_id = hashlib.sha1(page[:4000].encode("utf-8")).hexdigest()[:16]
            if page_id in collector.pages and page_id not in read:
                continue  # another topic already read this page (a redirect, or another session)
            read.add(page_id)
            for pair in propose(llm, topic, url, passage, int(cfg.get("pairs_per_page", 3))):
                key = question_key(pair["question"])
                if collector.seen(pair["question"]) or key in asked or time.time() > deadline:
                    continue
                asked.add(key)
                passed, checks, why = check_pair(pair, page, llm, book)
                trust, confirmed = "medium", ""
                # the game data already confirms it: no need to search again (that is most of the CPU time)
                if passed and cfg.get("corroborate", True) and checks.get("game_data") != "ok":
                    verdict, confirmed = corroborate(llm, pair, search, fetch, {url},
                                                     int(cfg.get("corroborate_pages", 3)))
                    checks["independent_page"] = verdict
                    if verdict == "CONTRADICTED":
                        passed, why = False, f"an independent page contradicts it ({confirmed})"
                if not passed:
                    rejected.append({"topic": topic, "url": url, "why": why, **pair, "checks": checks})
                    continue
                if checks.get("game_data") == "ok" or checks.get("independent_page") == "SUPPORTED":
                    trust = "high"
                if trust == "medium" and cfg.get("keep", "medium") == "high":
                    rejected.append({"topic": topic, "url": url, "why": "only medium trust", **pair, "checks": checks})
                    continue
                rows.append(make_row(topic, pair, url, policy[1], checks, trust, confirmed, cfg.get("model", "")))
    except Exception as exc:  # noqa: BLE001 - one bad topic never stops the run
        log(f"{topic}: skipped ({type(exc).__name__}: {str(exc)[:160]})")
    else:
        collector.pages.update(read)
    return rows, rejected


def session_topics(topic_list: list[str], sessions: int, session: int, help_others: bool = True) -> list[str]:
    """Assign topics by a stable hash, independent of topic list order or changes between notebooks.

    Optional helping processes other parts after the session's own. Helping can overlap with active sessions.
    """
    if sessions < 1 or not 1 <= session <= sessions:
        raise ValueError("session must be between 1 and sessions")
    shares: list[list[str]] = [[] for _ in range(sessions)]
    for topic in topic_list:
        digest = hashlib.sha256(norm(topic).encode("utf-8")).digest()
        shares[int.from_bytes(digest[:8], "big") % sessions].append(topic)
    own = shares[session - 1]
    if not help_others or sessions == 1:
        return own
    others = [share[::-1] for i, share in enumerate(shares) if i != session - 1]
    longest = max((len(share) for share in others), default=0)
    return own + [share[i] for i in range(longest) for share in others if i < len(share)]


def collect(cfg: dict, llm: Any, collector: Collector, book: dict, topic_list: list[str],
            search: Callable[[str], list[dict]], fetch: Callable[[str], str], deadline: float,
            on_upload: Callable[[str], None] = lambda _m: None, discord: Discord | None = None,
            processes: Any = None) -> None:
    """The main loop until the time, row or topic limit, uploading every few minutes. ``cfg["parallel"]`` topics
    are worked on at once (llama-server batches their requests, and while one waits for a web search the others
    keep the CPU busy); the next topic's search and pages are always fetched in the background - in the worker
    ``processes`` when given (see ``process_pool``), else in threads. Rows are added in topic order. Saves what it has even when stopped (the Kaggle stop button)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    discord = discord or Discord("")
    last_upload, last_report = time.time(), time.time()
    finished = collector.finished()
    todo = [t for t in topic_list if t not in finished]
    if cfg.get("max_topics"):
        todo = todo[:int(cfg["max_topics"])]
    workers = max(1, int(cfg.get("parallel", 1)))
    own_fetcher = processes is None
    fetcher = processes or ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prefetch")
    llm_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="topic") if workers > 1 else None
    fetched: dict[int, Any] = {}
    running: deque = deque()  # (number, topic, future) in topic order

    def fetch_ahead(index: int) -> None:
        if index < len(todo) and index not in fetched:
            fetched[index] = fetcher.submit(gather, todo[index], cfg, search, fetch)

    def run(index: int) -> tuple[list[dict], list[dict]]:
        found = fetched.pop(index).result()
        return work_topic(todo[index], found, cfg, llm, collector, book, search, fetch, deadline)

    def finish(number: int, topic: str, result: tuple[list[dict], list[dict]]) -> None:
        nonlocal last_upload, last_report
        before = len(collector.rows)
        rows, rejected = result
        for row in rows:
            if not collector.seen(row["question"]):  # two topics at once can write the same question
                collector.add(row)
        collector.rejected.extend(rejected)
        collector.done.append(topic)
        stats = collector.stats()
        hours = max((time.time() - collector.started) / 3600, 1e-6)
        rate = (stats["rows"] - collector.start_rows) / hours
        left = "no time limit" if deadline == float("inf") else f"{max(deadline - time.time(), 0) / 3600:.1f} h left"
        log(f"[{number}/{len(todo)}] {topic}: +{stats['rows'] - before} rows (total {stats['rows']}: "
            f"{stats['high']} high, {stats['medium']} medium; {stats['rejected']} rejected) - {rate:.0f} rows/h, "
            f"{left}")
        if time.time() - last_report > 3600:
            last_report = time.time()
            speed = llm_speed(llm)
            log(speed) if speed else None
            discord.send(f"{stats['rows']} rows ({stats['high']} high trust), {stats['rejected']} rejected, "
                         f"{stats['topics']} topics done, {rate:.0f} rows/h.", "Progress")
        if time.time() - last_upload > float(cfg.get("upload_every_min", 30)) * 60:
            last_upload = time.time()
            collector.save()
            on_upload(f"{stats['rows']} rows after {stats['topics']} topics")

    try:
        for index, topic in enumerate(todo):
            if time.time() > deadline or (cfg.get("max_rows") and len(collector.rows) >= int(cfg["max_rows"])):
                break
            if topic in collector.done_elsewhere:  # another session did it meanwhile (synced at every upload)
                stale = fetched.pop(index, None)
                if stale is not None:
                    stale.cancel()
                continue
            for ahead in range(1 + max(1, int(cfg.get("prefetch", 1)))):  # the next topics' pages, while the LLM works
                fetch_ahead(index + ahead)
            if llm_pool is None:
                finish(index + 1, topic, run(index))
                continue
            running.append((index + 1, topic, llm_pool.submit(run, index)))
            if len(running) >= workers:
                number, name, future = running.popleft()
                finish(number, name, future.result())
        while running:  # the topics still being worked on (they stop early after the deadline)
            number, name, future = running.popleft()
            finish(number, name, future.result())
    finally:
        for pool in (fetcher if own_fetcher else None, llm_pool):
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
        for future in fetched.values():  # a shared process pool: drop the pages fetched ahead
            future.cancel()
        collector.save()


def main(cfg: dict) -> dict:
    """Everything the notebook's last cell runs. ``cfg`` = the notebook settings (see the settings cell)."""
    started = cfg.get("session_start") or session_start()
    work, tmp = Path(cfg.get("work", "/kaggle/working")), Path(cfg.get("tmp", "/kaggle/tmp"))
    folder = work / DATASET_NAME
    for path in (folder, tmp / "pages", tmp / "models", tmp / "tools"):
        path.mkdir(parents=True, exist_ok=True)
    token = hf_token(cfg.get("hf_secret", "HF_TOKEN"))
    discord = Discord(kaggle_secret(cfg.get("discord_secret", "")) if cfg.get("discord", True) else "",
                      kaggle_secret(cfg.get("ping_role_secret", "ping_role")))
    if cfg.get("model", "auto") != "auto" and cfg.get("model") not in MODELS:
        raise SystemExit(f"MODEL = {cfg.get('model')!r} is not known - use \"auto\" or one of: {', '.join(MODELS)}")
    if cfg.get("keep", "medium") not in ("medium", "high"):
        raise SystemExit(f"KEEP = {cfg.get('keep')!r} - use \"medium\" or \"high\"")
    ram = available_ram_gb()
    key, why = pick_model(cfg.get("model", "auto"), ram)
    keys = list(MODELS)[list(MODELS).index(key):] if cfg.get("model", "auto") == "auto" else [key]
    spec = MODELS[key]
    auto = cfg.get("model", "auto") == "auto"
    max_hours = float(cfg.get("max_hours", 11.75))
    if tpu_session() and max_hours > 8.75:  # Kaggle stops TPU sessions after 9 h: the last upload comes before that
        log(f"TPU session: only its CPUs are used, and it ends after 9 h - MAX_HOURS {max_hours:g} -> 8.75")
        max_hours = 8.75
    # MAX_HOURS 0 = no limit (your own PC: until every topic is done or Ctrl+C, which saves and uploads)
    deadline = started + max_hours * 3600 - float(cfg.get("reserve_min", 15)) * 60 if max_hours > 0 else float("inf")
    gpus = gpu_info() if cfg.get("device", "auto") in ("auto", "gpu") else []
    if cfg.get("device") == "gpu" and not gpus:
        log("DEVICE = \"gpu\" but no NVIDIA GPU is attached (Settings > Accelerator > GPU T4 x2) - using the CPU")
    vram = sum(g["vram_gb"] for g in gpus)
    if gpus:
        log(f"GPU: {len(gpus)} x {gpus[0]['name']} ({vram:.0f} GB VRAM) - llama.cpp runs on it")
    gpu_keys = [GPU_MODEL] if auto else [key]  # the GPU: the small fast one, one copy per GPU
    log(f"LLM: {MODELS[gpu_keys[0]]['name']} on the GPU (else {spec['name']} on the CPU)" if gpus else
        f"LLM: {spec['name']} {cpu_quant(key, ram, cfg.get('quant', 'auto'))} ({why})" + (f", or smaller if it writes under {cfg.get('min_tokens_per_s', 0)} tok/s"
                                          if len(keys) > 1 and float(cfg.get("min_tokens_per_s", 0)) > 0 else ""))

    def tuned(on_gpu: bool) -> dict[str, int]:
        """The settings left at 0 (auto) for the device that really runs the LLM."""
        return {"parallel": max(1, int(cfg.get("parallel", 0) or (8 if on_gpu else 4))),
                "pages_per_topic": int(cfg.get("pages_per_topic", 0) or 1),  # the whole page is read
                "pairs_per_page": min(6, int(cfg.get("pairs_per_page", 0) or 5))}  # per part of ~2,800 characters

    parallel = tuned(bool(gpus))["parallel"] * (1 if gpus else cpu_plan(*cpu_cores())[0])  # all servers together
    api = repo_id = None
    if cfg.get("upload", True):
        if not token:
            raise SystemExit(f"Add your Hugging Face token as the Kaggle secret {cfg.get('hf_secret', 'HF_TOKEN')!r} "
                             "(Add-ons > Secrets; on a PC: run 'hf auth login' or set HF_TOKEN), or set UPLOAD = False.")
        from huggingface_hub import HfApi  # type: ignore[import-not-found]

        api = HfApi(token=token)
        repo_id = hub_repo_id(api, cfg.get("dataset_repo", DATASET_NAME))
        if cfg.get("resume", True) and hub_resume(api, repo_id, folder, token):
            log(f"resuming from https://huggingface.co/datasets/{repo_id}")
        private = hub_make_private(api, repo_id, bool(cfg.get("private", True)))
        log(f"uploading to https://huggingface.co/datasets/{repo_id} - "
            + ("PRIVATE: only you can see it" if private else "PUBLIC: anyone can see it (PRIVATE = False)"))
    llm = collector = None
    # the web part in worker processes, started before llama-server and the topic threads
    # a GPU writes far faster than one topic's pages arrive: more web workers (every free CPU thread), fetching
    # many topics ahead. On a 4-core Kaggle CPU the LLM needs the cores: one worker per topic at once, as before.
    web = min(32, max(2 * parallel, cpu_cores()[1] - 2)) if gpus else min(parallel, 32)
    workers = process_pool(web) if cfg.get("processes", True) else None
    prefetch = None
    if not gpus:  # the CPU: llama.cpp and the model download while the game data and the instant rows get ready

        def download() -> None:
            try:
                llama_server_binary(tmp / "tools")
                model_file(spec, tmp / "models", token, cpu_quant(key, ram, cfg.get("quant", "auto")))
            except Exception as exc:  # noqa: BLE001 - the normal start below tries again and reports it
                log(f"background download failed ({type(exc).__name__}: {str(exc)[:160]}) - trying again")

        prefetch = threading.Thread(target=download, name="download-llm", daemon=True)
        prefetch.start()

    def upload(message: str) -> None:
        if api is not None and repo_id and collector is not None:
            try:
                link = hub_upload(api, repo_id, folder, bool(cfg.get("private", True)), message,
                                  [*collector.own_files(), collector.claim_name])
                discord.send(f"{message}: {link}", "Uploaded", "good", important=True)
            except Exception as exc:  # noqa: BLE001 - try again at the next upload
                log(f"upload failed ({type(exc).__name__}: {str(exc)[:200]})")
            try:
                added = sync_scraped_pages(repo_id, token, tmp / "scrape-sync", tmp / "pages")
                if added:
                    log(f"{added} scraped pages imported from the PC")
            except Exception as exc:  # noqa: BLE001 - normal web fetch remains the fallback
                log(f"could not sync PC scrape cache ({type(exc).__name__})")
            if int(cfg.get("sessions", 1)) > 1 and not collector.ended:  # what the others did meanwhile: skipped
                try:
                    new = collector.note_others(hub_progress(repo_id, token, tmp / "sync"))
                    log(f"{new} more topics done by the other sessions - skipped here") if new else None
                except Exception as exc:  # noqa: BLE001 - next time
                    log(f"could not read the other sessions' progress ({type(exc).__name__})")

    def summary(stats: dict) -> dict:
        link = f"https://huggingface.co/datasets/{repo_id}" if repo_id else str(folder)
        samples = [{k: r[k] for k in ("question", "answer", "trust")} for r in (collector.rows if collector else [])[-3:]]
        return {**stats, "link": link, "samples": samples}

    try:
        discord.send(f"Collecting a trusted Minecraft dataset on the {'GPU' if gpus else 'CPU'} "
                     f"(testing {spec['name']} first)"
                     + (f" -> {repo_id}" if repo_id else "") + ".", "Notebook started", "info", important=True)
        data = load_minecraft_data(cfg.get("minecraft_version", MINECRAFT_DATA_VERSION))
        book = recipe_book(data)
        topic_list = all_topics(data, tmp, cfg)
        sessions, session = max(1, int(cfg.get("sessions", 1))), int(cfg.get("session", 1))
        run = f"{os.getpid()}-{time.time_ns()}"
        shared = sessions > 1 and api is not None and repo_id  # several sessions on one Hub dataset: claims
        if session == 0 and not shared:
            session = 1
        elif session == 0:  # auto: the lowest number no running notebook / PC holds
            session = claim_session(sessions, *hub_claims(api, repo_id, token, tmp / "claims"), run)
            log(f"SESSION = 0 (auto): this is session {session} of {sessions}")
        elif shared and 1 <= session <= sessions:
            read_claims, write_claim = hub_claims(api, repo_id, token, tmp / "claims")
            if claim_live(read_claims().get(session), time.time()):
                raise SystemExit(f"SESSION = {session} is already running somewhere else (two sessions with one number "
                                 "overwrite each other's rows) - use SESSION = 0 (auto) or another number")
            write_claim(session, {"run": run, "time": time.time(), "ended": False})
        if not 1 <= session <= sessions:
            raise SystemExit(f"SESSION = {session} - use 1 to SESSIONS ({sessions}), or 0 = auto")
        share = len(session_topics(topic_list, sessions, session, help_others=False))
        topic_list = session_topics(topic_list, sessions, session, bool(cfg.get("help_others", True)))
        collector = Collector(folder, spec["name"], session)
        collector.run = run  # the claim on the Hub is refreshed with every upload
        stats = collector.stats()
        if session == 1 and cfg.get("game_data", True):  # exact rows from the game data, in a second, no LLM
            new = [r for r in game_data_rows(data, book, int(cfg.get("seed", 0))) if not collector.seen(r["question"])]
            if int(cfg.get("max_rows", 0)) > 0:
                new = new[:max(0, int(cfg["max_rows"]) - len(collector.rows))]
            for row in new:
                collector.add(row)
            if new:
                collector.save()
                log(f"{len(new)} exact rows from the game data (recipes, stack sizes, durability, hardness, tools, "
                    f"food, enchantments, mobs, biomes) - no LLM needed")
                upload(f"game data: {len(new)} exact rows")
        if prefetch is not None:
            prefetch.join()

        def starter(binary: Path, on_gpu: bool) -> Callable[[str, bool], Any]:
            def start(name: str, speculate: bool) -> Any:
                one_gpu = min((g["vram_gb"] for g in gpus), default=0.0)
                # a copy on EACH GPU when the model fits one (2 x T4: twice the work), else one model across them
                copies = len(gpus) if on_gpu and len(gpus) > 1 and one_gpu >= MODELS[name].get("gpu_vram_gb", 1e9) + 2 \
                    else 1
                threads = threads_batch = 0  # 0 = LlamaServer's own choice (all cores)
                if not on_gpu:  # a big CPU machine (TPU session): several servers sharing the model in RAM
                    copies, threads, threads_batch = cpu_plan(*cpu_cores())
                chosen_quant = str(cfg.get("quant", "auto") or "auto")
                quant = (chosen_quant if chosen_quant != "auto" else gpu_quant(name, one_gpu if copies > 1 else vram)) \
                    if on_gpu else cpu_quant(name, ram, chosen_quant)
                model = model_file(MODELS[name], tmp / "models", token, quant)
                slots = tuned(on_gpu)["parallel"]
                where = (f"{copies} GPUs, one copy on each" if copies > 1 else f"{len(gpus)} GPU(s)") if on_gpu \
                    else ("{} CPU cores / {} threads".format(*cpu_cores()) if copies == 1 else
                          f"{copies} servers x {threads} CPU cores (one shared copy of the model in RAM)")
                log(f"starting llama-server with {model.name} on {where}, {slots} requests at once"
                    + (" each" if copies > 1 else "") + (", n-gram speculation" if speculate else ""))
                servers: list[LlamaServer] = []
                try:
                    for index in range(copies):
                        servers.append(LlamaServer(binary, model, ctx=int(cfg.get("ctx", 4096)),
                                                   hybrid=MODELS[name]["hybrid"], parallel=slots, speculate=speculate,
                                                   port=8089 + index, gpu=index if copies > 1 and on_gpu else None,
                                                   threads=threads, threads_batch=threads_batch).start())
                except BaseException:
                    for server in servers:
                        server.stop()
                    raise
                return LLMPool(servers) if copies > 1 else servers[0]
            return start

        target, speculate = float(cfg.get("min_tokens_per_s", 0)), str(cfg.get("speculate", "auto"))
        plans: list[tuple[str, Callable[[], Path]]] = []
        if gpus:
            arch = gpus[0]["arch"]
            plans += [("GPU", lambda: llama_server_binary(tmp / "tools", cuda_arch=arch)),
                      ("GPU (compiled here)", lambda: llama_server_binary(tmp / "tools", cuda_arch=arch, prebuilt=False))]
        plans.append(("CPU", lambda: llama_server_binary(tmp / "tools")))
        chosen = None
        used: list[Path] = []
        no_cuda_build = False
        for label, get_binary in plans:
            if no_cuda_build and label != "CPU":
                continue
            try:
                try:
                    binary = get_binary()
                except Exception:
                    no_cuda_build = label != "CPU"  # downloading / compiling failed: compiling again will not help
                    raise
                if binary in used:  # the prebuilt one failed and was already the compiled one
                    continue
                used.append(binary)
                chosen = (label, *fast_llm(keys if label == "CPU" else gpu_keys, starter(binary, label != "CPU"),
                                           target, speculate, step_down=label == "CPU"))
                break
            except Exception as exc:  # noqa: BLE001 - the next way (compiled for the GPU, then the CPU)
                log(f"{label}: llama.cpp did not work ({type(exc).__name__}: {str(exc)[:300]})")
        if chosen is None:
            raise RuntimeError("llama.cpp did not start on the GPU or the CPU (see llama-server.log)")
        device, llm, key, speed, speculating = chosen
        spec = MODELS[key]
        settings = {**tuned(device != "CPU"), "parallel": int(getattr(llm, "parallel", 1)),  # all copies together
                    "prefetch": web if device != "CPU" else 1}  # topics whose pages are fetched ahead
        parallel = settings["parallel"]
        cfg = {**cfg, **settings, "model": spec["name"]}
        collector.model = spec["name"]
        log(f"{parallel} topics at once, {cfg['pages_per_topic']} page(s) per topic read whole, "
            f"{cfg['pairs_per_page']} pairs per part of the page")
        log(f"LLM: {spec['name']} on the {device}, writing {speed:.1f} tok/s"
            + (" with n-gram speculation" if speculating else ""))
        discord.send(f"LLM: {spec['name']} on the {device}, writing {speed:.1f} tok/s ({parallel} requests at once).",
                     "LLM ready")
        log(f"session {session} of {sessions}: {share} topics of its own"
            + (f" (then it helps the other sessions with theirs, from their end: {len(topic_list) - share} more)"
               if len(topic_list) > share else "")
            + f", {len(collector.finished() & set(topic_list))} done before; {len(book)} recipes for the game-data check; {stats['rows']} rows so far (all sessions)")
        if api is not None and repo_id:
            try:
                added = sync_scraped_pages(repo_id, token, tmp / "scrape-sync", tmp / "pages")
                log(f"{added} scraped pages imported from the PC; missing pages are fetched normally")
            except Exception as exc:  # noqa: BLE001
                log(f"PC scrape cache unavailable ({type(exc).__name__}); fetching pages normally")
        collect(cfg, llm, collector, book, topic_list, WebSearch(int(cfg.get("search_results", 8))),
                PageFetcher(tmp / "pages"), deadline, upload, discord, processes=workers)
        stats = collector.stats()
        collector.ended = True
        collector.save()
        upload(f"final: {stats['rows']} rows ({stats['high']} high trust) from {stats['topics']} topics")
        result = summary(stats)
        log(f"done: {stats} -> {result['link']} (LLM: {llm.calls} calls; {llm_speed(llm)})")
        discord.send(f"{stats['rows']} rows ({stats['high']} high trust, {stats['medium']} medium), "
                     f"{stats['rejected']} rejected, {stats['topics']} topics.\n{result['link']}", "Notebook finished",
                     "good", important=True)
        return result
    except KeyboardInterrupt:  # the Kaggle stop button: keep and upload what was collected
        if collector is None:
            raise
        stats = collector.stats()
        collector.ended = True
        collector.save()
        log(f"stopped - uploading the {stats['rows']} rows collected so far")
        upload(f"stopped early: {stats['rows']} rows ({stats['high']} high trust) from {stats['topics']} topics")
        discord.send(f"Stopped by you after {stats['topics']} topics: {stats['rows']} rows kept.", "Notebook stopped",
                     "warn", important=True)
        return {**summary(stats), "stopped": True}
    except BaseException as exc:
        if collector is not None and not collector.ended:  # keep what it has and free the session number
            try:
                collector.ended = True
                collector.save()
                upload(f"error ({type(exc).__name__}): {collector.stats()['rows']} rows kept")
            except Exception:  # noqa: BLE001, S110 - the error below matters more
                pass
        discord.send(f"`{type(exc).__name__}: {str(exc)[:1500]}`", "Notebook error", "bad", important=True)
        raise
    finally:
        if llm is not None:
            llm.stop()
        if workers is not None:
            workers.shutdown(wait=False, cancel_futures=True)


def pc_config(argv: list[str] | None = None) -> dict:
    """The settings for your own computer (``python pymc_bot/llm_webdata.py``; on Windows
    ``scripts/collect_dataset_pc.bat``): it joins the Kaggle sessions as one more session of the same dataset."""
    import argparse

    parser = argparse.ArgumentParser(prog="llm_webdata", description=(
        "Collect the Minecraftism dataset on this computer (NVIDIA GPU, else the CPU), as one more session next to "
        "the Kaggle notebooks. Ctrl+C stops it and uploads what it has."))
    parser.add_argument("--sessions", type=int, default=11, help="SESSIONS of the notebooks (the same everywhere)")
    parser.add_argument("--session", type=int, default=0, help="0 = auto: the lowest number nobody runs")
    parser.add_argument("--hours", type=float, default=0, help="stop after this many hours; 0 = when all is done")
    parser.add_argument("--device", default="auto", choices=("auto", "gpu", "cpu"))
    parser.add_argument("--model", default="auto", help="auto, or one of: " + ", ".join(MODELS))
    parser.add_argument("--parallel", type=int, default=0, help="requests at once; 0 = auto (8 on a GPU)")
    parser.add_argument("--folder", default=str(Path.home() / "Minecraftism"), help="where the data is kept")
    parser.add_argument("--dataset-repo", default=DATASET_NAME)
    parser.add_argument("--topics", type=int, default=0, help="only this many topics (a test run); 0 = all")
    parser.add_argument("--no-upload", action="store_true", help="keep everything on this computer")
    args = parser.parse_args(argv)
    work = Path(args.folder).expanduser()
    return {"sessions": args.sessions, "session": args.session, "max_hours": args.hours, "reserve_min": 5,
            "device": args.device, "model": args.model, "parallel": args.parallel, "work": str(work),
            "tmp": str(work / "tmp"), "dataset_repo": args.dataset_repo, "max_topics": args.topics,
            "upload": not args.no_upload, "discord_secret": "discord_webhook_inform_me1", "session_start": time.time()}


if __name__ == "__main__" and "ipykernel" not in sys.modules:  # pragma: no cover - the notebook calls main()
    print(json.dumps(main(pc_config()), indent=1, ensure_ascii=False))
