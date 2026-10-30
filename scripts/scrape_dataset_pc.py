"""Scrape source pages on a PC for the Kaggle dataset workers. No model is used."""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pymc_bot import llm_webdata as web  # noqa: E402


def fetch_with_retry(url: str, cache: Path, attempts: int = 3) -> str:
    """Retry transient connection errors, but do not repeatedly request missing or forbidden pages."""
    for attempt in range(attempts):
        try:
            return web.fetch_page(url, cache)
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt + 1 == attempts:
                raise
            # Respect an explicit Retry-After, within a bounded waiting period.
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            delay = min(60.0, float(retry_after)) if retry_after.isdigit() else 0.0
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            if attempt + 1 == attempts:
                raise
            delay = 0.0
        time.sleep(max(delay, min(30.0, 2 ** attempt + random.uniform(0.0, 2.0))))
    raise RuntimeError("no fetch attempts")


def scrape(topic: str, cache: str) -> str:
    url = web.wiki_url(topic)
    folder = Path(cache)
    try:
        fetch_with_retry(url, folder)
        return url
    except ValueError:  # no article with this exact title; use the same source search as Kaggle
        for hit in web.web_search(f"{topic} site:minecraft.fandom.com"):
            candidate = hit.get("url", "")
            policy = web.source_policy(candidate)
            if policy and policy[0] == "text" and candidate != url:
                try:
                    fetch_with_retry(candidate, folder)
                    return candidate
                except (ValueError, urllib.error.URLError, TimeoutError, OSError):
                    continue
        raise


def remote_inventory(repo, token, folder):
    """Read already uploaded page names and exact contents; fail closed if inventory is unavailable."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo, repo_type="dataset", local_dir=str(folder), token=token or None,
                      allow_patterns=["scrape-cache/*.zip"])
    names, hashes = set(), set()
    for archive in (folder / "scrape-cache").glob("*.zip"):
        with zipfile.ZipFile(archive) as batch:
            for entry in batch.infolist():
                if not web.re.fullmatch(r"[0-9a-f]{40}\.txt", entry.filename):
                    continue
                if entry.file_size > 10_000_000:
                    raise ValueError(f"oversized page in {archive.name}")
                content = batch.read(entry)
                names.add(entry.filename)
                hashes.add(web.hashlib.sha256(content).hexdigest())
    return names, hashes


def upload_pending(api, repo, cache, state, batch_dir, known_urls=None, known_hashes=None):
    """Upload each URL and exact page text at most once; update state only after successful upload."""
    saved = json.loads(state.read_text(encoding="utf-8")) if state.exists() else []
    if isinstance(saved, list):  # older PC versions stored only URL-hash names
        uploaded, hashes = set(saved), set()
        for name in uploaded:
            path = cache / name
            if path.is_file():
                hashes.add(web.hashlib.sha256(path.read_bytes()).hexdigest())
    else:
        uploaded, hashes = set(saved.get("urls", [])), set(saved.get("hashes", []))
    uploaded.update(known_urls or ())
    hashes.update(known_hashes or ())
    pending = sorted(p for p in cache.glob("*.txt") if p.name not in uploaded)
    batch_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(pending), 100):
        group = pending[start:start + 100]
        unique, skipped = [], []
        group_hashes = set(hashes)
        for path in group:
            content = path.read_bytes()
            digest = web.hashlib.sha256(content).hexdigest()
            if digest in group_hashes:
                skipped.append(path.name)
            else:
                unique.append((path, content, digest))
                group_hashes.add(digest)
        if unique:
            identity = "\n".join(path.name + ":" + digest for path, _, digest in unique)
            batch_id = web.hashlib.sha256(identity.encode()).hexdigest()[:24]
            archive = batch_dir / f"{batch_id}.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for path, content, _ in unique:
                    z.writestr(path.name, content)
            api.upload_file(path_or_fileobj=str(archive), path_in_repo=f"scrape-cache/{archive.name}",
                            repo_id=repo, repo_type="dataset", commit_message="Upload scraped source pages")
            archive.unlink()
        uploaded.update(skipped)
        uploaded.update(path.name for path, _, _ in unique)
        hashes.update(digest for _, _, digest in unique)
        temp = state.with_suffix(".tmp")
        temp.write_text(json.dumps({"urls": sorted(uploaded), "hashes": sorted(hashes)}), encoding="utf-8")
        temp.replace(state)
        print(f"Uploaded {len(unique)} pages; skipped {len(skipped)} exact duplicates", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Scrape pages for Kaggle; no LLM or GPU is used.")
    parser.add_argument("--folder", default=str(Path.home() / "Minecraftism-scrape"))
    parser.add_argument("--dataset-repo", default=web.DATASET_NAME)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4))
    parser.add_argument("--topics", type=int, default=0, help="limit topics for a test")
    args = parser.parse_args()
    token = web.hf_token()
    if not token:
        raise SystemExit("Log in with hf auth login or set HF_TOKEN (write access required).")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo = web.hub_repo_id(api, args.dataset_repo)
    web.hub_make_private(api, repo, True)
    root = Path(args.folder).expanduser()
    cache = root / "pages"
    cache.mkdir(parents=True, exist_ok=True)
    state = root / ("uploaded-" + web.hashlib.sha256(repo.encode()).hexdigest()[:16] + ".json")
    # Reconcile with the Hub even if local state was lost or the scraper moved to another PC.
    # If the Hub cannot be checked, keep scraping locally and retry inventory before uploading.
    inventory = None
    data = web.load_minecraft_data()
    topics = web.all_topics(data, root, {})
    if args.topics:
        topics = topics[:args.topics]
    workers = max(1, min(args.workers, 32))
    print(f"Scraping {len(topics)} topics with {workers} processes; no LLM or GPU.", flush=True)
    def upload():
        nonlocal inventory
        try:
            if inventory is None:
                inventory = remote_inventory(repo, token, root / "remote-inventory")
            upload_pending(api, repo, cache, state, root / "batches", *inventory)
            return True
        except Exception as exc:  # preserve local cache and retry on next run
            print(f"Upload failed; pages retained for retry: {type(exc).__name__}: {exc}", flush=True)
            return False
    upload()
    pool = ProcessPoolExecutor(max_workers=workers)
    last = time.monotonic()
    try:
        # Bounded batches keep memory usage and the interruption delay manageable.
        for start in range(0, len(topics), workers * 4):
            futures = [pool.submit(scrape, topic, str(cache)) for topic in topics[start:start + workers * 4]]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Page skipped: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            if time.monotonic() - last >= 60:
                upload()
                last = time.monotonic()
    except KeyboardInterrupt:
        print("Stopping: saving completed downloads", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        if not upload():
            raise SystemExit("Upload failed; rerun to retry the saved pages.")


if __name__ == "__main__":
    main()
