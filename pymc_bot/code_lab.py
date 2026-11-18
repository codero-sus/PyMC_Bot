"""Reviewable Mineflayer proposals and an opt-in isolated trial runner.

Generated code is never executed by the live bridge; lab results are untrusted.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from pymc_bot.experience import SENSITIVE
from pymc_bot.skill_drafts import NAME

MAX_SOURCE_BYTES = 16000
MAX_PROPOSALS = 16
_LOCK = threading.RLock()
_LAB_LOCK = threading.Lock()
_IMAGE = re.compile(r"^sha256:[a-f0-9]{64}$")
_AI_LOCK = threading.Lock()
_LAST_AI_PROPOSAL = 0.0


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
        self.scope = key
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
            if 'revision' not in row:  # proposals created before revision tracking
                legacy = json.dumps([name, row.get('request'), row.get('digest')], ensure_ascii=False)
                row['revision'] = hashlib.sha256(legacy.encode()).hexdigest()[:32]
            if (row.get('name') != name or row.get('status') not in ('unreviewed', 'reviewed')
                    or not isinstance(row.get('test'), dict) or not re.fullmatch(r'[0-9a-f]{32}', str(row.get('revision', '')))):
                raise ValueError('invalid proposal metadata')
            source = row.get('source')
            if not isinstance(source, str) or not source or len(source.encode('utf-8')) > MAX_SOURCE_BYTES + 1:
                raise ValueError('invalid stored source')
            if row.get('digest') != hashlib.sha256(source.encode()).hexdigest():
                raise ValueError('proposal source digest mismatch')
        return data

    def _write(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        if len(payload.encode('utf-8')) > 300000:
            raise ValueError('proposal store is too large')
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temp = tempfile.mkstemp(prefix='.code-', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(payload)
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

    def save(self, name: str, request: str, source: str, *, replace: bool = True) -> dict[str, Any]:
        name, request = validate_request(name, request)
        source = extract_source(source)
        row = {'name': name, 'request': request, 'source': source, 'revision': uuid.uuid4().hex,
               'digest': hashlib.sha256(source.encode()).hexdigest(),
               'status': 'unreviewed', 'test': {'status': 'not_run'}}
        with _LOCK:
            data = self._load()
            if name in data and not replace:
                raise ValueError('proposal name already exists; only an operator can replace it')
            if name not in data and len(data) >= MAX_PROPOSALS:
                raise ValueError('proposal limit reached; delete one first')
            data[name] = row
            self._write(data)
        return row.copy()

    def set_test(self, name: str, digest: str, revision: str, result: dict[str, Any]) -> bool:
        with _LOCK:
            data = self._load()
            row = data.get(name)
            if not row or row['digest'] != digest or row['revision'] != revision:
                return False
            row['test'] = {'status': str(result.get('status', 'error'))[:40],
                           'detail': str(result.get('detail', ''))[:240]}
            row['status'] = 'unreviewed'
            row.pop('reviewed_digest', None)
            self._write(data)
            return True

    def approve(self, name: str, digest: str, revision: str) -> dict[str, Any]:
        """Record a human review of exact source; this does not enable execution."""
        with _LOCK:
            data = self._load()
            if name not in data:
                raise KeyError(name)
            row = data[name]
            if row['digest'] != digest or row['revision'] != revision:
                raise ValueError('proposal changed; review it again')
            if row.get('test', {}).get('status') != 'completed_unverified':
                raise ValueError('a completed isolated trial is required before approval')
            row['status'] = 'reviewed'
            row['reviewed_digest'] = digest
            self._write(data)
            return row.copy()

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


def submit_ai_proposal(store: CodeStore, name: Any, request: Any, source: Any) -> dict[str, Any]:
    """Bound autonomous proposal creation; testing occurs asynchronously."""
    global _LAST_AI_PROPOSAL
    with _AI_LOCK:
        now = time.monotonic()
        if _LAST_AI_PROPOSAL and now - _LAST_AI_PROPOSAL < 300:
            raise ValueError('AI proposals are limited to one every five minutes')
        row = store.save(name, request, source, replace=False)
        _LAST_AI_PROPOSAL = now

    def trial() -> None:
        result = run_isolated(row['source'])
        try:
            store.set_test(row['name'], row['digest'], row['revision'], result)
        except (OSError, ValueError):
            pass  # The proposal can be deleted or edited while the sandbox runs.

    threading.Thread(target=trial, name='pymc-skill-trial', daemon=True).start()
    return row


def _inspect(docker: str, args: list[str], env: dict[str, str]) -> str:
    try:
        result = subprocess.run([docker, *args], env=env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=8, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError('sandbox runtime inspection failed') from exc
    if result.returncode or len(result.stdout) > 16384:
        raise RuntimeError('sandbox runtime or image unavailable')
    return result.stdout.strip()


def run_isolated(source: str) -> dict[str, str]:
    """Run only in an explicitly configured disposable, networkless gVisor container.

    Image, Docker daemon and gVisor are trusted operator infrastructure. Candidate
    code is not trusted, and the in-sandbox completion signal is not proof of success.
    """
    image = os.environ.get('PYMC_BOT_LAB_IMAGE', '')
    if os.environ.get('PYMC_BOT_LAB_EULA') != 'accept':
        return {'status': 'blocked', 'detail': 'Operator must explicitly accept the Minecraft EULA for the test world.'}
    if not _IMAGE.fullmatch(image):
        return {'status': 'blocked', 'detail': 'Set PYMC_BOT_LAB_IMAGE to the trusted local lab image ID (sha256:...).'}
    if sys.platform != 'linux':
        return {'status': 'blocked', 'detail': 'The skill lab requires Linux, rootless Docker and gVisor.'}
    docker = shutil.which('docker')
    runtime_dir = os.environ.get('XDG_RUNTIME_DIR', '')
    if not docker or not runtime_dir or not os.path.isabs(runtime_dir):
        return {'status': 'blocked', 'detail': 'Rootless Docker socket and gVisor are required.'}
    if not _LAB_LOCK.acquire(blocking=False):
        return {'status': 'blocked', 'detail': 'Another isolated trial is already running.'}
    name = 'pymc-skill-lab-' + uuid.uuid4().hex
    proc: subprocess.Popen[bytes] | None = None
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': os.environ.get('HOME', '/tmp'),
           'XDG_RUNTIME_DIR': runtime_dir, 'DOCKER_HOST': 'unix://' + runtime_dir + '/docker.sock'}
    try:
        if not isinstance(source, str) or not source or len(source.encode('utf-8')) > MAX_SOURCE_BYTES + 1:
            raise RuntimeError('invalid candidate source')
        options = json.loads(_inspect(docker, ['info', '--format', '{{json .SecurityOptions}}'], env))
        runtimes = json.loads(_inspect(docker, ['info', '--format', '{{json .Runtimes}}'], env))
        if not isinstance(options, list) or not any('name=rootless' in str(x) for x in options):
            raise RuntimeError('Docker daemon must be rootless')
        if not isinstance(runtimes, dict) or 'runsc' not in runtimes:
            raise RuntimeError('gVisor runsc runtime is required')
        limits = _inspect(docker, ['info', '--format',
                          '{{json .MemoryLimit}} {{json .PidsLimit}} {{json .CPUCfsQuota}} {{.CgroupVersion}}'], env)
        if limits != 'true true true 2':
            raise RuntimeError('Docker must report cgroup v2 memory, PID and CPU quota support')
        identity = _inspect(docker, ['image', 'inspect', image, '--format',
                            '{{.Id}} {{index .Config.Labels "org.pymcbot.skill-lab"}}'], env)
        if identity != f'{image} 1':
            raise RuntimeError('lab image identity or label did not match')
        command = [docker, 'run', '--rm', '--name', name, '--pull=never', '--runtime=runsc',
                   '--network=none', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
                   '--user=10001:10001', '--memory=2g', '--cpus=2', '--pids-limit=128',
                   '--tmpfs=/tmp:rw,nosuid,nodev,mode=1777,size=1g', '--log-driver=none', '-i', image]
        proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, start_new_session=True)
        assert proc.stdin is not None and proc.stdout is not None
        payload = json.dumps({'source': source}, ensure_ascii=False).encode()
        output = bytearray()
        deadline = time.monotonic() + 115
        with selectors.DefaultSelector() as selector:
            os.set_blocking(proc.stdin.fileno(), False)
            selector.register(proc.stdin, selectors.EVENT_WRITE)
            selector.register(proc.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                for key, _ in selector.select(timeout=0.2):
                    if key.fileobj is proc.stdin:
                        count = os.write(key.fd, payload[:4096])
                        payload = payload[count:]
                        if not payload:
                            selector.unregister(proc.stdin)
                            proc.stdin.close()
                    else:
                        chunk = os.read(key.fd, 4096)
                        if not chunk:
                            selector.unregister(proc.stdout)
                        else:
                            output.extend(chunk)
                            if len(output) > 16384:
                                raise RuntimeError('sandbox output limit exceeded')
                if proc.poll() is not None and not selector.get_map():
                    break
            else:
                raise RuntimeError('isolated trial timed out')
        if proc.returncode != 0:
            raise RuntimeError('test server or candidate exited unsuccessfully')
        try:
            report = json.loads(output.decode('utf-8').strip().splitlines()[-1])
        except (ValueError, IndexError, UnicodeDecodeError) as exc:
            raise RuntimeError('invalid sandbox report') from exc
        if not isinstance(report, dict) or report.get('status') != 'completed_unverified':
            raise RuntimeError('candidate did not complete the disposable run')
        return {'status': 'completed_unverified', 'detail': 'Untrusted completion signal only; behavior and safety are not verified.'}
    except (RuntimeError, OSError, ValueError) as exc:
        return {'status': 'blocked_or_failed', 'detail': str(exc)[:240]}
    finally:
        if proc and proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if proc:
            try:
                subprocess.run([docker, 'rm', '-f', name], env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass
        _LAB_LOCK.release()
