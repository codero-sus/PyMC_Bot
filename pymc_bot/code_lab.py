"""Reviewable Mineflayer code proposals; generated code is never executed."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

from pymc_bot.experience import SENSITIVE
from pymc_bot.skill_drafts import NAME

MAX_SOURCE_BYTES = 16000
MAX_PROPOSALS = 16
_LOCK = threading.RLock()


def validate_request(name: Any, request: Any) -> tuple[str, str]:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise ValueError("name must be lowercase letters, numbers or underscores, starting with a letter")
    if not isinstance(request, str) or not request.strip() or len(request) > 500 or SENSITIVE.search(request):
        raise ValueError("request must be 1–500 characters without obvious credentials")
    return name, request.strip()


def extract_source(response: str) -> str:
    if not isinstance(response, str) or not response.strip():
        raise ValueError("model returned no JavaScript")
    source = response.strip()
    fenced = re.search(r"```(?:javascript|js|commonjs)?\s*\n([\s\S]*?)```", source, re.I)
    if fenced:
        source = fenced.group(1).strip()
    if not source or len(source.encode('utf-8')) > MAX_SOURCE_BYTES or '\x00' in source:
        raise ValueError("generated JavaScript is empty, too large or contains a NUL")
    return source + "\n"


class CodeStore:
    def __init__(self, directory: Path, identity: str) -> None:
        key = hashlib.sha256(identity.lower().encode()).hexdigest()[:24]
        self.path = directory / 'code-skill-proposals' / f'{key}.json'

    def _load(self) -> dict[str, Any]:
        try:
            if self.path.stat().st_size > 300000:
                raise ValueError('proposal store too large')
            data = json.loads(self.path.read_text(encoding='utf-8'))
        except FileNotFoundError:
            return {}
        if not isinstance(data, dict) or len(data) > MAX_PROPOSALS:
            raise ValueError('invalid proposal store')
        for name, row in data.items():
            if not isinstance(row, dict):
                raise ValueError('invalid proposal')
            validate_request(name, row.get('request'))
            source = row.get('source')
            if not isinstance(source, str) or not source or len(source.encode('utf-8')) > MAX_SOURCE_BYTES + 1:
                raise ValueError('invalid stored source')
            if row.get('digest') != hashlib.sha256(source.encode()).hexdigest():
                raise ValueError('proposal source digest mismatch')
        return data

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp = tempfile.mkstemp(prefix='.code-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(data, stream, ensure_ascii=False, separators=(',', ':'))
            os.replace(temp, self.path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)

    def list(self) -> list[dict[str, Any]]:
        with _LOCK:
            return [{k: v for k, v in row.items() if k != 'source'} for row in self._load().values()]

    def get(self, name: str) -> dict[str, Any]:
        with _LOCK:
            data = self._load()
            if name not in data:
                raise KeyError(name)
            return data[name].copy()

    def save(self, name: str, request: str, source: str) -> dict[str, Any]:
        name, request = validate_request(name, request)
        source = extract_source(source)
        row = {'name': name, 'request': request, 'source': source,
               'digest': hashlib.sha256(source.encode()).hexdigest(),
               'status': 'unreviewed', 'test': {'status': 'not_run'}}
        with _LOCK:
            data = self._load()
            if name not in data and len(data) >= MAX_PROPOSALS:
                raise ValueError('proposal limit reached; delete one first')
            data[name] = row
            self._write(data)
        return row.copy()

    def set_test(self, name: str, digest: str, result: dict[str, Any]) -> bool:
        with _LOCK:
            data = self._load()
            row = data.get(name)
            if not row or row['digest'] != digest:
                return False
            row['test'] = {'status': str(result.get('status', 'error'))[:40],
                           'detail': str(result.get('detail', ''))[:240]}
            self._write(data)
            return True

    def delete(self, name: str) -> bool:
        with _LOCK:
            data = self._load()
            if name not in data:
                return False
            del data[name]
            if data:
                self._write(data)
            else:
                self.path.unlink(missing_ok=True)
            return True



def run_isolated(source: str) -> dict[str, str]:
    """Fail closed until a separately audited isolation runner is available."""
    return {"status": "blocked", "detail": "No verified isolated test-world runner is configured; code was not executed."}
