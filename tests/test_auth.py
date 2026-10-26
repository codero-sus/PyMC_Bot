"""The panel's login (pymc_bot/auth.py + the middleware in server.py)."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from pymc_bot import auth
from pymc_bot.__main__ import main
from pymc_bot.server import PUBLIC_PATHS, create_app

PASSWORD = "correct horse battery"


@pytest.fixture()
def login_on(monkeypatch):
    monkeypatch.delenv("PYMC_BOT_NO_AUTH", raising=False)


@pytest.fixture()
def app(tmp_path: Path, login_on):
    return create_app(config_path=tmp_path / "cfg.json")


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def account(app):
    app.state.auth.set_account("admin", PASSWORD)
    return app.state.auth


def login(client: TestClient, password: str = PASSWORD, **extra):
    return client.post("/api/auth/login", json={"username": "admin", "password": password, **extra})


# --------------------------------------------------------------------------- the store
def test_passwords_are_salted_hashes_in_a_private_file(tmp_path: Path):
    store = auth.AuthStore(tmp_path / "auth.json")
    assert not store.configured
    store.set_account("admin", PASSWORD)
    raw = (tmp_path / "auth.json").read_text()
    assert PASSWORD not in raw and json.loads(raw)["password"]["algorithm"] == "pbkdf2_sha256"
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "auth.json").stat().st_mode) == 0o600
    first = json.loads(raw)["password"]
    store.set_account("admin", PASSWORD)
    assert json.loads((tmp_path / "auth.json").read_text())["password"]["salt"] != first["salt"]
    assert auth.verify_password(PASSWORD, store._data["password"]) and not auth.verify_password("nope", first | {"hash": "x"})
    with pytest.raises(auth.AuthError, match="8 characters"):
        store.set_account("admin", "short")
    with pytest.raises(auth.AuthError, match="username"):
        store.set_account("bad name!", PASSWORD)


def test_sessions_are_signed_and_expire(tmp_path: Path, monkeypatch):
    store = auth.AuthStore(tmp_path / "auth.json")
    store.set_account("admin", PASSWORD)
    cookie, lifetime = store.issue_session()
    assert lifetime == auth.SESSION_SECONDS and store.verify_session(cookie).username == "admin"
    body, _, signature = cookie.partition(".")
    assert store.verify_session(body + "." + signature[::-1]) is None  # tampered
    assert store.verify_session("garbage") is None and store.verify_session(None) is None
    assert store.issue_session(remember=True)[1] == auth.REMEMBER_SECONDS
    real_time = auth.time.time
    monkeypatch.setattr(auth.time, "time", lambda: real_time() + auth.SESSION_SECONDS + 5)
    assert store.verify_session(cookie) is None  # expired


def test_tokens_are_stored_as_hashes_and_can_be_revoked(tmp_path: Path):
    store = auth.AuthStore(tmp_path / "auth.json")
    store.set_account("admin", PASSWORD)
    token = store.new_token()
    assert token.startswith("pymc_") and token not in (tmp_path / "auth.json").read_text()
    assert store.verify_token(token).via == "token" and store.verify_token("pymc_wrong") is None
    newer = store.new_token()
    assert store.verify_token(token) is None and store.verify_token(newer)  # replaced
    store.revoke_token()
    assert store.verify_token(newer) is None
    assert auth.bearer_token("Bearer abc") == "abc" and auth.bearer_token("Basic abc") is None
    assert auth.bearer_token(None) is None and auth.bearer_token("Bearer ") is None


def test_changes_from_the_cli_apply_to_a_running_panel_at_once(tmp_path: Path):
    panel, cli = auth.AuthStore(tmp_path / "auth.json"), auth.AuthStore(tmp_path / "auth.json")
    cli.set_account("admin", PASSWORD)
    assert panel.configured
    for _ in range(5):  # quicker than the file system's timestamps
        token = cli.new_token()
        assert panel.verify_token(token)
    cookie, _ = panel.issue_session()
    cli.set_account("admin", "another password")
    assert panel.verify_session(cookie) is None  # the new password logged it out


def test_too_many_failures_lock_the_address_out(tmp_path: Path):
    store = auth.AuthStore(tmp_path / "auth.json")
    store.set_account("admin", PASSWORD)
    for _ in range(auth.MAX_FAILURES):
        with pytest.raises(auth.AuthError):
            store.login("admin", "wrong", "1.2.3.4")
    with pytest.raises(auth.LockedOut):
        store.login("admin", PASSWORD, "1.2.3.4")  # even the right password waits
    store.login("admin", PASSWORD, "5.6.7.8")  # other addresses are not affected


def test_the_auth_file_sits_next_to_the_config(tmp_path: Path, monkeypatch):
    assert auth.default_auth_path(tmp_path / "x" / "cfg.json") == (tmp_path / "x" / auth.AUTH_FILENAME).resolve()
    monkeypatch.setenv("PYMC_BOT_AUTH_FILE", str(tmp_path / "elsewhere.json"))
    assert auth.default_auth_path(tmp_path / "cfg.json") == tmp_path / "elsewhere.json"


# --------------------------------------------------------------------------- the panel
def test_everything_but_the_login_page_needs_a_login(app, client, account):
    assert client.get("/healthz").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
    reply = client.get("/api/status")
    assert reply.status_code == 401 and reply.json()["login"] == "/login"
    assert reply.headers["www-authenticate"].startswith("Bearer")
    page = client.get("/", follow_redirects=False)
    assert page.status_code == 303 and page.headers["location"] == "/login?next=%2F"
    assert client.get("/docs", follow_redirects=False).status_code == 303
    assert client.post("/api/bot/start").status_code == 401
    assert client.put("/api/config", json={"minecraft": {"username": "Evil"}}).status_code == 401
    # every route that is not on the public list is closed
    for route in app.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or set()
        if path in PUBLIC_PATHS or path.startswith("/static") or "{" in path or not methods:
            continue
        for method in methods - {"HEAD"}:
            answer = client.request(method, path, follow_redirects=False)
            assert answer.status_code in (401, 303), (method, path, answer.status_code)


def test_login_page_logout_and_the_cookie(client, account):
    page = client.get("/login")
    assert page.status_code == 200 and "Log in" in page.text and "login.js" in page.text
    assert login(client, "wrong").status_code == 401
    reply = login(client)
    assert reply.status_code == 200 and reply.json()["username"] == "admin"
    cookie = reply.headers["set-cookie"]
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Max-Age" not in cookie  # browser session
    assert client.get("/api/status").status_code == 200
    assert client.get("/", follow_redirects=False).status_code == 200
    assert client.get("/login?next=/docs", follow_redirects=False).headers["location"] == "/docs"
    assert client.get("/login?next=//evil.example", follow_redirects=False).headers["location"] == "/"
    status = client.get("/api/auth/status").json()
    assert status == {"enabled": True, "configured": True, "authenticated": True, "username": "admin",
                      "via": "cookie", "token": False}
    client.post("/api/auth/logout")
    assert client.get("/api/status").status_code == 401
    assert "Max-Age" in login(client, remember=True).headers["set-cookie"]


def test_login_is_throttled(client, account):
    for _ in range(auth.MAX_FAILURES):
        assert login(client, "wrong").status_code == 401
    reply = login(client)
    assert reply.status_code == 429 and int(reply.headers["retry-after"]) > 0


def test_first_run_needs_the_setup_code_from_the_console(app, capsys):
    with TestClient(app) as client:
        _first_run(app, client, capsys)


def _first_run(app, client, capsys):
    assert client.get("/api/auth/status").json()["configured"] is False
    code = app.state.auth.setup_code
    assert code and code in capsys.readouterr().out  # printed for the person who started the panel
    page = client.get("/login")
    assert "Create your login" in page.text and code not in page.text
    assert login(client).status_code == 409
    body = {"username": "steve", "password": PASSWORD}
    assert client.post("/api/auth/setup", json={**body, "code": "WRONG-CODE"}).status_code == 400
    assert client.post("/api/auth/setup", json={**body, "code": code.lower(), "password": "short"}).status_code == 400
    reply = client.post("/api/auth/setup", json={**body, "code": code.lower()})
    assert reply.status_code == 200 and client.get("/api/status").status_code == 200  # logged in right away
    assert client.post("/api/auth/setup", json={**body, "code": code}).status_code == 409  # only once
    assert app.state.auth.setup_code == ""


def test_password_change_logs_out_other_browsers(app, account):
    with TestClient(app) as first, TestClient(app) as second:
        assert login(first).status_code == 200 and login(second).status_code == 200
        wrong = first.post("/api/auth/password", json={"current": "nope", "new": "another password"})
        assert wrong.status_code == 400
        assert first.post("/api/auth/password", json={"current": PASSWORD, "new": "another password"}).status_code == 200
        assert first.get("/api/status").status_code == 200  # this one got a fresh cookie
        assert second.get("/api/status").status_code == 401  # the other one is out
        assert login(second, "another password").status_code == 200


def test_api_token_for_scripts_and_the_mod(app, client, account):
    assert client.post("/api/auth/token").status_code == 401
    login(client)
    token = client.post("/api/auth/token").json()["token"]
    with TestClient(app) as script:
        headers = {"Authorization": f"Bearer {token}"}
        assert script.get("/api/status", headers=headers).status_code == 200
        assert script.post("/api/bot/action", headers=headers, json={"action": "stop"}).status_code != 401
        assert script.get("/api/status", headers={"Authorization": "Bearer pymc_wrong"}).status_code == 401
        # a token cannot mint tokens or change the password
        assert script.post("/api/auth/token", headers=headers).status_code == 403
        assert script.post("/api/auth/password", headers=headers,
                           json={"current": PASSWORD, "new": "another password"}).status_code == 403
    assert client.get("/api/auth/status").json()["token"] is True
    assert client.delete("/api/auth/token").status_code == 200
    with TestClient(app) as script:
        assert script.get("/api/status", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_websocket_needs_a_login(client, account):
    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws") as socket:
        socket.receive_json()
    login(client)
    with client.websocket_connect("/ws") as socket:
        assert socket.receive_json()["type"] == "hello"


def test_config_api_never_shows_the_account(client, account):
    login(client)
    text = json.dumps(client.get("/api/config").json())
    assert "password" not in text and "pbkdf2" not in text
    client.post("/api/config/reset")
    assert client.get("/api/status").status_code == 200  # "Reset to defaults" keeps the login


def test_no_auth_turns_it_off(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("PYMC_BOT_NO_AUTH", "1")
    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        assert client.get("/api/status").status_code == 200
        assert client.get("/api/auth/status").json()["enabled"] is False
        assert client.get("/login", follow_redirects=False).status_code == 303


# --------------------------------------------------------------------------- the CLI
def test_cli_auth_commands(tmp_path: Path, monkeypatch, capsys, login_on):
    config = str(tmp_path / "cfg.json")
    answers = iter([PASSWORD, PASSWORD])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert main(["auth", "--config", config, "token"]) == 1  # no account yet
    assert main(["auth", "--config", config, "set-password"]) == 0
    store = auth.AuthStore(tmp_path / auth.AUTH_FILENAME)
    assert store.username == "admin" and store.login("admin", PASSWORD) is None
    capsys.readouterr()
    assert main(["auth", "--config", config, "token"]) == 0
    token = capsys.readouterr().out.strip()
    assert store.verify_token(token)
    assert main(["auth", "--config", config]) == 0
    assert "api token   : set" in capsys.readouterr().out
    assert main(["auth", "--config", config, "revoke-token"]) == 0 and not store.verify_token(token)
    answers = iter(["new password!", "different!!"])
    monkeypatch.setattr("getpass.getpass", lambda prompt="": next(answers))
    assert main(["auth", "--config", config, "set-password"]) == 1  # the two did not match
    assert main(["auth", "--config", config, "reset"]) == 0 and not store.configured


def test_cli_action_sends_the_token(monkeypatch, capsys):
    import httpx

    seen = {}

    def post(url, json=None, headers=None, timeout=0):
        seen.update(url=url, headers=headers or {})
        return httpx.Response(401, json={"detail": "login required"})

    monkeypatch.setattr(httpx, "post", post)
    assert main(["action", '{"action":"stop"}', "--token", "pymc_abc"]) == 1
    assert seen["headers"] == {"Authorization": "Bearer pymc_abc"}
    assert "auth token" in capsys.readouterr().err
    monkeypatch.setenv("PYMC_BOT_TOKEN", "pymc_env")
    main(["action", '{"action":"stop"}'])
    assert seen["headers"] == {"Authorization": "Bearer pymc_env"}


def test_the_mod_sends_the_token():
    root = Path(__file__).resolve().parents[1] / "mod/src"
    botlink = (root / "client/java/dev/pymc/pymcb/client/BotLink.java").read_text(encoding="utf-8")
    commands = (root / "client/java/dev/pymc/pymcb/client/PymcCommands.java").read_text(encoding="utf-8")
    config = (root / "main/java/dev/pymc/pymcb/PlaytimeConfig.java").read_text(encoding="utf-8")
    assert '"Authorization", "Bearer " + token' in botlink and "setInstanceFollowRedirects(false)" in botlink
    assert "code == 401" in botlink and "/pymcbot bot token <token>" in botlink
    assert 'literal("token")' in commands and 'guarded("clear")' in commands and ".then(token)" in commands
    assert 'props.setProperty("panelToken", panelToken)' in config and 'getProperty("panelToken")' in config

