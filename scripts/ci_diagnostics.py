#!/usr/bin/env python3
"""Publish the tail of a CI log as a GitHub check summary.

GitHub Actions job logs live on Azure Blob Storage; some environments (locked-down
sandboxes, restricted proxies) can download the *annotations* of a check run but not the
log archive. This helper makes a failed step debuggable everywhere: run it in an
``if: failure()`` step and the last few thousand characters of the build/test log become
readable through the API and the Checks UI.

Usage::

    python scripts/ci_diagnostics.py build.log --title "Gradle build failed"
    gh api "repos/$GITHUB_REPOSITORY/commits/$GITHUB_SHA/check-runs" \\
      --jq '.check_runs[] | select(.name=="ci diagnostics") | .output.summary'

Only the standard library is used, so it works on any runner without extra installs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

CHECK_NAME = "ci diagnostics"
MAX_SUMMARY = 60000


def read_tail(path: str, limit: int) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        return f"(could not read {path}: {exc})"
    tail = text[-limit:] if limit > 0 else text
    header = "" if len(text) <= limit else f"... ({len(text) - limit} earlier characters omitted)\n"
    return header + tail


def post_check_run(repo: str, sha: str, token: str, title: str, summary: str, conclusion: str) -> str:
    payload = {
        "name": CHECK_NAME,
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {"title": title[:255], "summary": summary[:MAX_SUMMARY]},
    }
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/check-runs",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - fixed GitHub URL
        body = json.loads(response.read().decode("utf-8"))
    return str(body.get("html_url", ""))


def write_step_summary(title: str, summary: str) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"### {title}\n\n{summary}\n\n")
    except OSError as exc:  # pragma: no cover - CI only
        print(f"ci_diagnostics: could not write the step summary ({exc})", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="ci.log", help="log file to summarise")
    parser.add_argument("--title", default="CI step failed", help="check title")
    parser.add_argument("--limit", type=int, default=12000, help="how many trailing characters to publish")
    parser.add_argument("--conclusion", default="failure", choices=["failure", "neutral", "success"])
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY", ""))
    parser.add_argument("--sha", default=os.environ.get("GITHUB_SHA", ""))
    args = parser.parse_args(argv)

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token or not args.repo or not args.sha:
        print("ci_diagnostics: no token/repo/sha in the environment; printing the tail instead:")
        print(read_tail(args.log, args.limit))
        return 0

    summary = "```text\n" + read_tail(args.log, args.limit) + "\n```"
    # The step summary needs no extra permissions and is visible in the Actions UI,
    # so write it even when the check-run call below succeeds.
    write_step_summary(args.title, summary)
    try:
        url = post_check_run(args.repo, args.sha, token, args.title, summary, args.conclusion)
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:  # pragma: no cover - CI only
        print(f"ci_diagnostics: could not post the check run ({exc}); tail follows:", file=sys.stderr)
        print(read_tail(args.log, args.limit))
        return 0
    print(f"ci_diagnostics: published '{args.title}' as a check summary ({url})")
    return 0


if __name__ == "__main__":  # pragma: no cover - CI helper
    raise SystemExit(main())
