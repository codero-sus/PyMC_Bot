"""Cortex LLMHoster integration: client, config, agent brain, stub routes and API.

Nothing here needs a running Cortex: the wire format is exercised with
``httpx.MockTransport`` (shaped like Cortex's real responses) and with the
OpenAI-compatible routes of the bundled demo stub.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from pymc_bot.agent import AgentLoop
from pymc_bot.bot import MinecraftBot
from pymc_bot.config import AppSettings, ConfigStore, CortexSettings
from pymc_bot.cortex import CortexClient, CortexError, client_from_settings
from pymc_bot.events import EventLog
from pymc_bot.fleet import AI_MODES, primary_brain
from pymc_bot.ollama_stub import STUB_MODEL, serve
from pymc_bot.server import create_app

MODELS = {
    "object": "list",
    "data": [
        {"id": "qwen-local", "object": "model", "runtime": "llama.cpp", "capabilities": ["text_generation"]},
        {"id": "whisper", "object": "model", "runtime": "command", "capabilities": ["audio_transcription"]},
    ],
}


def _chat_reply(content, model: str = "qwen-local") -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


class FakeCortex:
    """Answers like Cortex LLMHoster; records every request it sees."""

    def __init__(self, *, key: str | None = None, ready: bool = True, chat_status: int = 200,
                 reject_json_mode: bool = False, reply: object = '{"action": "wander"}') -> None:
        self.key = key
        self.ready = ready
        self.chat_status = chat_status
        self.reject_json_mode = reject_json_mode
        self.reply = reply
        self.requests: list[httpx.Request] = []

    def _error(self, status: int, message: str) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": message, "type": "invalid_request_error"}})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/ready":
            if self.ready:
                return httpx.Response(200, json={"status": "ready", "models": 2})
            return httpx.Response(
                503,
                json={"status": "not_ready", "reason": "default_local_model_not_running", "model": "qwen-local"},
            )
        if self.key is not None and request.headers.get("authorization") != f"Bearer {self.key}":
            return httpx.Response(401, json={"error": {"message": "A valid bearer token is required."}})
        if path == "/v1/models":
            return httpx.Response(200, json=MODELS)
        if path == "/v1/chat/completions":
            body = json.loads(request.content)
            if self.reject_json_mode and "response_format" in body:
                return self._error(400, "response_format is not supported by this runtime")
            if self.chat_status != 200:
                return self._error(self.chat_status, "Requested capability is not supported")
            return httpx.Response(200, json=_chat_reply(self.reply, body.get("model", "qwen-local")))
        if path == "/v1/completions":
            return httpx.Response(
                200, json={"object": "text_completion", "model": "qwen-local",
                           "choices": [{"index": 0, "text": '{"action": "jump"}'}]},
            )
        return self._error(404, "not found")


def _client(fake: FakeCortex, **kwargs) -> CortexClient:
    return CortexClient(client=httpx.Client(transport=httpx.MockTransport(fake)), **kwargs)


# ------------------------------------------------------------------- client
def test_chat_sends_openai_payload_and_reads_the_reply():
    fake = FakeCortex()
    client = _client(fake, model="qwen-local", temperature=0.2, max_tokens=99)
    reply = client.chat([{"role": "user", "content": "state"}], system="be a player")
    assert reply == '{"action": "wander"}'
    assert client.last_model == "qwen-local"
    body = json.loads(fake.requests[-1].content)
    assert fake.requests[-1].url.path == "/v1/chat/completions"
    assert body["model"] == "qwen-local"
    assert body["messages"][0] == {"role": "system", "content": "be a player"}
    assert body["temperature"] == 0.2 and body["max_tokens"] == 99 and body["stream"] is False
    assert body["response_format"] == {"type": "json_object"}


def test_blank_model_lets_cortex_pick_its_default():
    fake = FakeCortex()
    client = _client(fake, model="")
    client.chat([{"role": "user", "content": "hi"}])
    assert "model" not in json.loads(fake.requests[-1].content)


def test_key_comes_from_the_named_env_var(monkeypatch):
    fake = FakeCortex(key="s3cret")
    monkeypatch.delenv("PYMC_TEST_CORTEX_KEY", raising=False)
    client = _client(fake, api_key_env="PYMC_TEST_CORTEX_KEY")
    with pytest.raises(CortexError) as missing:
        client.list_models()
    assert missing.value.status == 401 and "PYMC_TEST_CORTEX_KEY" in str(missing.value)

    monkeypatch.setenv("PYMC_TEST_CORTEX_KEY", "s3cret")
    assert [m["id"] for m in client.list_models()] == ["qwen-local", "whisper"]
    assert fake.requests[-1].headers["authorization"] == "Bearer s3cret"


def test_no_key_means_no_authorization_header(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    fake = FakeCortex()
    _client(fake).list_models()
    assert "authorization" not in fake.requests[-1].headers


def test_json_mode_is_dropped_when_the_runtime_rejects_it():
    fake = FakeCortex(reject_json_mode=True)
    reply = _client(fake).chat([{"role": "user", "content": "x"}])
    assert reply == '{"action": "wander"}'
    first, second = (json.loads(r.content) for r in fake.requests[-2:])
    assert "response_format" in first and "response_format" not in second


def test_decide_falls_back_to_plain_completions():
    fake = FakeCortex(chat_status=422)
    reply = _client(fake, json_mode=False).decide("state", system="sys")
    assert reply == '{"action": "jump"}'
    assert fake.requests[-1].url.path == "/v1/completions"
    assert json.loads(fake.requests[-1].content)["prompt"].startswith("sys")


def test_auth_errors_do_not_fall_back(monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    fake = FakeCortex(key="k")
    with pytest.raises(CortexError) as err:
        _client(fake).decide("state")
    assert err.value.status == 401
    assert all(r.url.path != "/v1/completions" for r in fake.requests)


def test_content_parts_are_joined():
    fake = FakeCortex(reply=[{"type": "text", "text": '{"action":'}, {"type": "text", "text": ' "stop"}'}])
    assert _client(fake).chat([{"role": "user", "content": "x"}]) == '{"action": "stop"}'


def test_empty_reply_is_an_error():
    with pytest.raises(CortexError, match="empty"):
        _client(FakeCortex(reply="  ")).chat([{"role": "user", "content": "x"}])


def test_ping_reports_ready_models():
    ok, detail = _client(FakeCortex(), model="qwen-local").ping()
    assert ok is True and "qwen-local" in detail and "2 model" in detail


def test_ping_explains_what_is_wrong():
    ok, detail = _client(FakeCortex(ready=False)).ping()
    assert ok is False and "default_local_model_not_running" in detail

    ok, detail = _client(FakeCortex(), model="nope").ping()
    assert ok is False and "'nope' is not configured" in detail

    ok, detail = _client(FakeCortex(), model="whisper").ping()
    assert ok is False and "text_generation" in detail

    client = CortexClient(base_url="http://127.0.0.1:1", timeout=1.0)
    ok, detail = client.ping(timeout=1.0)
    client.close()
    assert ok is False and "Could not reach Cortex" in detail


def test_custom_api_path():
    fake = FakeCortex()

    def moved(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/openai/"):
            request.url = request.url.copy_with(path=request.url.path.replace("/openai/", "/v1/", 1))
        return fake(request)

    client = CortexClient(api_path="openai", client=httpx.Client(transport=httpx.MockTransport(moved)))
    assert client.api_base.endswith("/openai")
    assert client.list_models()


# ------------------------------------------------------------------- config
def test_cortex_settings_defaults_and_validation():
    settings = AppSettings()
    assert settings.cortex.enabled is False  # opt-in, like Ollama
    assert settings.cortex.base_url == "http://127.0.0.1:8624"
    assert settings.cortex.api_path == "/v1" and settings.cortex.model == ""
    assert settings.cortex.api_key_env == "CORTEX_API_KEY"
    assert CortexSettings(base_url=" http://box:9000/ ", api_path="v1/").model_dump()["base_url"] == "http://box:9000"
    assert CortexSettings(api_path="v1/").api_path == "/v1"
    with pytest.raises(ValidationError):
        CortexSettings(base_url="ftp://nope")
    with pytest.raises(ValidationError):
        CortexSettings(api_key_env="not a variable")


def test_key_is_never_persisted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CORTEX_API_KEY", "do-not-save-me")
    store = ConfigStore(tmp_path / "cfg.json")
    store.load()
    store.update({"cortex": {"enabled": True, "model": "qwen-local"}})
    text = (tmp_path / "cfg.json").read_text()
    assert "do-not-save-me" not in text
    assert store.settings.cortex.key_present() is True
    assert client_from_settings(store.settings.cortex).api_key == "do-not-save-me"


def test_auto_brain_resolution(store: ConfigStore):
    assert "cortex" in AI_MODES
    assert primary_brain(store.settings) == "heuristic"
    store.update({"cortex": {"enabled": True}})
    assert primary_brain(store.settings) == "cortex"
    store.update({"ollama": {"enabled": True}})
    assert primary_brain(store.settings) == "ollama"  # Ollama keeps priority in auto
    store.update({"agent": {"mode": "cortex"}})
    assert primary_brain(store.settings) == "cortex"


# -------------------------------------------------------------------- agent
def test_agent_thinks_with_cortex(live_bot: MinecraftBot, store: ConfigStore, log: EventLog):
    fake = FakeCortex(reply='{"action": "say", "message": "hello from cortex"}')
    store.update({"agent": {"mode": "cortex"}, "cortex": {"decision_interval": 2.5}})
    agent = AgentLoop(live_bot, store, log, cortex=_client(fake))
    ok, detail = agent.run_once()
    status = agent.status
    assert status["brain"] == "cortex" and status["resolved_brain"] == "cortex"
    assert status["last_decision"]["source"] == "cortex:qwen-local"
    assert status["last_decision"]["action"] == "say"
    assert agent.decision_interval() == 2.5
    # the prompt the model saw carries the world state and the action contract
    body = json.loads(fake.requests[-1].content)
    assert '"position"' in body["messages"][-1]["content"]
    assert "Reply with a single JSON object" in body["messages"][0]["content"]
    assert ok, detail


def test_auto_mode_uses_cortex_when_it_is_the_enabled_llm(live_bot, store: ConfigStore, log: EventLog):
    store.update({"agent": {"mode": "auto"}, "cortex": {"enabled": True}})
    agent = AgentLoop(live_bot, store, log, cortex=_client(FakeCortex()))
    agent.run_once()
    assert agent.status["resolved_brain"] == "cortex"
    assert agent.status["last_decision"]["source"].startswith("cortex:")


def test_agent_falls_back_when_cortex_is_down(live_bot, store: ConfigStore, log: EventLog):
    store.update({"agent": {"mode": "cortex"}, "cortex": {"base_url": "http://127.0.0.1:1", "request_timeout": 1}})
    agent = AgentLoop(live_bot, store, log)
    ok, detail = agent.run_once()
    assert isinstance(ok, bool) and detail
    assert agent.status["last_decision"]["source"] == "heuristic"
    assert agent.status["errors"] == 1
    assert "Could not reach Cortex" in agent.status["last_error"]


def test_agent_follows_live_cortex_config(live_bot, store: ConfigStore, log: EventLog):
    fake = FakeCortex()
    store.update({"agent": {"mode": "cortex"}})
    agent = AgentLoop(live_bot, store, log, cortex=_client(fake))
    agent.run_once()
    store.update({"cortex": {"model": "qwen-local", "temperature": 1.1}})
    agent.run_once()
    body = json.loads(fake.requests[-1].content)
    assert body["model"] == "qwen-local" and body["temperature"] == 1.1


# --------------------------------------------------- stub (OpenAI-compatible)
@pytest.fixture()
def stub_url():
    httpd = serve("127.0.0.1", 0)
    host, port = httpd.server_address[:2]
    yield f"http://{host}:{port}"
    httpd.shutdown()
    httpd.server_close()


def test_stub_speaks_the_openai_dialect(stub_url: str):
    client = CortexClient(base_url=stub_url, timeout=5.0)
    try:
        assert client.health() is True
        ok, detail = client.ping()  # the stub has no /ready -> treated as a plain OpenAI server
        assert ok is True and STUB_MODEL in detail
        state = '{"health": 20, "food": 20, "players": [], "position": {"x": 0, "y": 64, "z": 0}}'
        assert '"action"' in client.chat([{"role": "user", "content": state}])
        assert '"action"' in client.complete(state)
    finally:
        client.close()


def test_agent_loop_runs_on_an_openai_endpoint(store: ConfigStore, log: EventLog, stub_url: str, wait_for):
    store.update(
        {
            "minecraft": {"backend": "simulated"},
            "agent": {"mode": "cortex"},
            "cortex": {"base_url": stub_url, "decision_interval": 0.5},
        }
    )
    bot = MinecraftBot(store, log, auto_reconnect=False)
    bot.start()
    time.sleep(0.5)
    agent = AgentLoop(bot, store, log)
    try:
        assert agent.start() is True
        # some stub actions (mine, goto) take a while in the simulated world
        assert wait_for(lambda: agent.status["decisions"] >= 2, timeout=20.0)
        status = agent.status
        assert status["decisions"] >= 2 and status["errors"] == 0
        assert status["last_decision"]["source"] == f"cortex:{STUB_MODEL}"
    finally:
        agent.stop()
        bot.stop()


# ---------------------------------------------------------------------- API
@pytest.fixture()
def api(tmp_path: Path):
    with TestClient(create_app(config_path=tmp_path / "cfg.json")) as client:
        yield client


def test_api_reports_cortex(api: TestClient, monkeypatch):
    monkeypatch.delenv("CORTEX_API_KEY", raising=False)
    api.put("/api/config", json={"cortex": {"base_url": "http://127.0.0.1:1"}})
    health = api.get("/api/cortex/health?force=true").json()
    assert health["ok"] is False and health["enabled"] is False
    assert health["key_env"] == "CORTEX_API_KEY" and health["key_set"] is False
    status = api.get("/api/status").json()
    assert status["cortex"]["settings"]["base_url"] == "http://127.0.0.1:1"
    assert "cortex" in api.get("/api/contract").json()["brains"]
    assert api.get("/api/cortex/models").status_code == 502


def test_api_lists_models_from_an_openai_endpoint(api: TestClient, stub_url: str):
    response = api.put("/api/config", json={"cortex": {"enabled": True, "base_url": stub_url}, "agent": {"mode": "cortex"}})
    assert response.status_code == 200
    assert response.json()["config"]["agent"]["mode"] == "cortex"
    models = api.get("/api/cortex/models").json()
    assert models["models"] == [STUB_MODEL] and models["details"][0]["chat"] is True
    assert api.get("/api/cortex/health?force=true").json()["ok"] is True


def test_api_rejects_a_bad_cortex_config(api: TestClient):
    # custom validators used to crash the endpoint (500: ValueError in the error context)
    response = api.put("/api/config", json={"cortex": {"api_key_env": "has spaces"}})
    assert response.status_code == 422
    assert "environment variable" in response.text
    assert api.put("/api/config", json={"cortex": {"base_url": "ftp://x"}}).status_code == 422


def test_fleet_accepts_cortex_and_trained_brains(api: TestClient):
    api.put("/api/config", json={"minecraft": {"backend": "simulated"}})
    for name, ai in (("CortexBuddy", "cortex"), ("TrainedBuddy", "trained")):
        response = api.post("/api/fleet/spawn", json={"username": name, "ai": ai, "start": False})
        assert response.status_code == 200, response.text
    roster = {bot["configured_username"]: bot["ai"] for bot in api.get("/api/fleet/status").json()["bots"]}
    assert roster["CortexBuddy"] == "cortex" and roster["TrainedBuddy"] == "trained"
