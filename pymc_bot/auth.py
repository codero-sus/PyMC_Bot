"""A lightweight login for the control panel: one account, signed session cookies, an API token.

The panel can start bots, change the config and run training, and it listens on ``0.0.0.0`` by
default, so it is locked behind a login. Everything here is standard library only:

* **One account** (username + password) in ``pymc_bot_auth.json`` next to the config - a separate
  file, so ``GET /api/config`` never shows it and "Reset to defaults" never removes it. The password
  is stored as a salted PBKDF2-SHA256 hash; the file is written atomically with mode ``0600``.
* **Sessions** are stateless signed cookies (HMAC-SHA256 over ``username|expiry``) with a random key
  from the auth file, so they survive a restart. Changing the password rotates the key: every other
  session is logged out.
* **API token** for scripts, ``python -m pymc_bot action`` and the Fabric mod's ``/pymcbot bot``:
  ``Authorization: Bearer <token>``. Only its SHA-256 is stored; it is shown once when created.
* **First run**: with no account yet, the panel shows a "create account" form that needs the one-time
  setup code printed in the server console (so nobody else on the network can claim the panel first).
  ``python -m pymc_bot auth set-password`` does the same from the terminal.
* **Brute force**: repeated failed logins from one address are slowed down (HTTP 429).

``python -m pymc_bot serve --no-auth`` (or ``PYMC_BOT_NO_AUTH=1``) turns it all off, for a panel that
only you can reach.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AUTH_FILENAME = "pymc_bot_auth.json"
COOKIE_NAME = "pymc_session"
TOKEN_PREFIX = "pymc_"
#: PBKDF2-SHA256 work factor (OWASP 2023 recommendation). Stored per hash, so it can be raised later.
PBKDF2_ITERATIONS = 600_000
SESSION_SECONDS = 12 * 3600
REMEMBER_SECONDS = 30 * 24 * 3600
MIN_PASSWORD = 8
MAX_PASSWORD = 256
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,32}$")
#: failed logins allowed per address inside the window before it has to wait
MAX_FAILURES = 5
FAILURE_WINDOW = 300.0
LOCKOUT_SECONDS = 60.0


class AuthError(ValueError):
    """A login, setup or password change was refused (the message is safe to show)."""


def auth_disabled() -> bool:
    """``PYMC_BOT_NO_AUTH=1`` (what ``serve --no-auth`` sets) turns the login off."""
    return os.environ.get("PYMC_BOT_NO_AUTH", "").strip().lower() in ("1", "true", "yes", "on")


def default_auth_path(config_path: Path | str | None = None) -> Path:
    """``PYMC_BOT_AUTH_FILE``, else ``pymc_bot_auth.json`` in the config's folder."""
    env = os.environ.get("PYMC_BOT_AUTH_FILE")
    if env:
        return Path(env).expanduser()
    if config_path is not None:
        return Path(config_path).expanduser().resolve().parent / AUTH_FILENAME
    from pymc_bot.config import default_config_path

    return default_config_path().resolve().parent / AUTH_FILENAME


# ---------------------------------------------------------------------------
# passwords and tokens
# ---------------------------------------------------------------------------
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def hash_password(password: str, iterations: int | None = None) -> dict[str, Any]:
    iterations = iterations or PBKDF2_ITERATIONS
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return {"algorithm": "pbkdf2_sha256", "iterations": iterations, "salt": _b64(salt), "hash": _b64(digest)}


def verify_password(password: str, record: dict[str, Any] | None) -> bool:
    if not record or record.get("algorithm") != "pbkdf2_sha256":
        return False
    try:
        salt, expected = _unb64(record["salt"]), _unb64(record["hash"])
        iterations = int(record["iterations"])
    except (KeyError, ValueError, TypeError):
        return False
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(digest, expected)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def check_password_rules(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD:
        raise AuthError(f"the password needs at least {MIN_PASSWORD} characters")
    if len(password) > MAX_PASSWORD:
        raise AuthError(f"the password can be at most {MAX_PASSWORD} characters")


def check_username(username: str) -> str:
    username = (username or "").strip()
    if not USERNAME_RE.match(username):
        raise AuthError("the username needs 1-32 letters, digits or _ . @ -")
    return username


# ---------------------------------------------------------------------------
# the account file
# ---------------------------------------------------------------------------
@dataclass
class Session:
    username: str
    expires: float
    via: str = "cookie"  # "cookie" | "token" | "disabled"


class AuthStore:
    """The account in ``pymc_bot_auth.json`` (thread-safe; reloaded when the file changes on disk)."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._data: dict[str, Any] = {}
        self._failures: dict[str, deque[float]] = {}
        self._locked_until: dict[str, float] = {}
        self.setup_code = ""
        self.load()

    # ------------------------------------------------------------------ file
    def load(self) -> None:
        """Read the file again. It is tiny, and re-reading it every time (instead of trusting its
        modification time, which is too coarse for two quick writes) means a password or token changed
        with ``python -m pymc_bot auth ...`` applies to a running panel at once."""
        with self._lock:
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError):  # missing (no account yet) or unreadable
                data = {}
            self._data = data if isinstance(data, dict) else {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        payload = json.dumps(self._data, indent=2) + "\n"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, self.path)
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # pragma: no cover - e.g. some Windows file systems
            pass

    def _fresh(self) -> dict[str, Any]:
        self.load()  # picks up `python -m pymc_bot auth ...` run while the panel is up
        return self._data

    # --------------------------------------------------------------- account
    @property
    def configured(self) -> bool:
        with self._lock:
            data = self._fresh()
            return bool(data.get("username") and data.get("password"))

    @property
    def username(self) -> str:
        with self._lock:
            return str(self._fresh().get("username") or "")

    @property
    def has_token(self) -> bool:
        with self._lock:
            return bool(self._fresh().get("token_sha256"))

    def set_account(self, username: str, password: str) -> None:
        """Create the account or replace it; logs out every session (new signing key)."""
        username = check_username(username)
        check_password_rules(password)
        with self._lock:
            data = dict(self._fresh())
            data.update({"version": 1, "username": username, "password": hash_password(password),
                         "secret": secrets.token_urlsafe(32), "updated": int(time.time())})
            self._data = data
            self._save()
            self.setup_code = ""

    def change_password(self, current: str, new: str) -> None:
        with self._lock:
            if not verify_password(current, self._fresh().get("password")):
                raise AuthError("the current password is wrong")
            self.set_account(self.username, new)

    def reset(self) -> None:
        """Forget the account (the panel asks for a new one with a fresh setup code)."""
        with self._lock:
            self._data = {}
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------ setup
    def new_setup_code(self) -> str:
        """The one-time code the first-run form asks for (printed in the server console)."""
        with self._lock:
            if not self.setup_code:
                self.setup_code = "-".join(secrets.token_hex(2).upper() for _ in range(3))
            return self.setup_code

    def setup(self, username: str, password: str, code: str) -> None:
        with self._lock:
            if self.configured:
                raise AuthError("an account already exists - log in instead")
            expected = self.setup_code
            given = re.sub(r"[^A-Z0-9]", "", (code or "").upper())
            if not expected or not hmac.compare_digest(given, expected.replace("-", "")):
                raise AuthError("wrong setup code - it is printed in the console that runs the panel")
            self.set_account(username, password)

    # ----------------------------------------------------------------- login
    def login(self, username: str, password: str, address: str = "") -> None:
        """Raises :class:`AuthError` on a wrong login; a :class:`LockedOut` after too many."""
        self.check_lockout(address)
        with self._lock:
            data = self._fresh()
            user_ok = hmac.compare_digest((username or "").strip().encode(), str(data.get("username") or "").encode())
            # always hash, so a wrong username takes as long as a wrong password
            password_ok = verify_password(password or "", data.get("password") or _dummy_record())
        if not (user_ok and password_ok and data.get("password")):
            self._record_failure(address)
            raise AuthError("wrong username or password")
        with self._lock:
            self._failures.pop(address, None)

    def check_lockout(self, address: str) -> None:
        with self._lock:
            until = self._locked_until.get(address, 0.0)
            if until > time.time():
                raise LockedOut(int(until - time.time()) + 1)

    def _record_failure(self, address: str) -> None:
        with self._lock:
            now = time.time()
            failures = self._failures.setdefault(address, deque())
            failures.append(now)
            while failures and now - failures[0] > FAILURE_WINDOW:
                failures.popleft()
            if len(failures) >= MAX_FAILURES:
                self._locked_until[address] = now + LOCKOUT_SECONDS
                failures.clear()

    # --------------------------------------------------------------- sessions
    def _secret(self) -> bytes:
        return str(self._fresh().get("secret") or "").encode("utf-8")

    def issue_session(self, remember: bool = False) -> tuple[str, int]:
        """A signed cookie value and its lifetime in seconds."""
        with self._lock:
            lifetime = REMEMBER_SECONDS if remember else SESSION_SECONDS
            body = _b64(json.dumps({"u": self.username, "e": int(time.time()) + lifetime}).encode())
            signature = _b64(hmac.new(self._secret(), body.encode(), hashlib.sha256).digest())
            return f"{body}.{signature}", lifetime

    def verify_session(self, cookie: str | None) -> Session | None:
        if not cookie or "." not in cookie:
            return None
        with self._lock:
            secret = self._secret()
            if not secret or not self.configured:
                return None
            body, _, signature = cookie.partition(".")
            expected = _b64(hmac.new(secret, body.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                return None
            try:
                payload = json.loads(_unb64(body))
            except (ValueError, TypeError):
                return None
            if payload.get("u") != self.username or float(payload.get("e", 0)) < time.time():
                return None
            return Session(self.username, float(payload["e"]))

    # ----------------------------------------------------------------- tokens
    def new_token(self) -> str:
        """A new API token (replaces the old one); only its hash is kept."""
        token = TOKEN_PREFIX + secrets.token_urlsafe(30)
        with self._lock:
            data = dict(self._fresh())
            data["token_sha256"] = _token_hash(token)
            data["token_created"] = int(time.time())
            self._data = data
            self._save()
        return token

    def revoke_token(self) -> None:
        with self._lock:
            data = dict(self._fresh())
            data.pop("token_sha256", None)
            data.pop("token_created", None)
            self._data = data
            if self.path.exists() or data:
                self._save()

    def verify_token(self, token: str | None) -> Session | None:
        if not token:
            return None
        with self._lock:
            stored = str(self._fresh().get("token_sha256") or "")
            if not stored or not self.configured:
                return None
            if not hmac.compare_digest(_token_hash(token.strip()), stored):
                return None
            return Session(self.username, float("inf"), via="token")

    def status(self) -> dict[str, Any]:
        with self._lock:
            data = self._fresh()
            return {"configured": self.configured, "username": self.username, "token": bool(data.get("token_sha256")),
                    "token_created": data.get("token_created"), "file": str(self.path)}


class LockedOut(AuthError):
    def __init__(self, seconds: int) -> None:
        super().__init__(f"too many failed logins - try again in {seconds} s")
        self.seconds = seconds


_DUMMY: dict[str, Any] = {}


def _dummy_record() -> dict[str, Any]:
    """A hash to check against when there is no account, so every failed login costs the same."""
    if _DUMMY.get("iterations") != PBKDF2_ITERATIONS:
        _DUMMY.clear()
        _DUMMY.update(hash_password(secrets.token_urlsafe(16)))
    return _DUMMY


def bearer_token(header: str | None) -> str | None:
    """``Authorization: Bearer <token>`` -> the token."""
    scheme, _, value = (header or "").partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()
