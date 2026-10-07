"""Running a trained model: the policy that turns predictions into live controls."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pymc_bot.bot import MinecraftBot
from pymc_bot.config import ConfigStore
from pymc_bot.events import EventLog
from pymc_bot.local_model import (
    PolicyError,
    TrainedPolicy,
    build_ollama_modelfile,
    find_run,
    policy_priors,
)
from pymc_bot.playtime import synthesize_playtime
from pymc_bot.train import TrainConfig, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> Path:
    """A tiny model trained on synthetic playtime (shared by this module)."""
    root = tmp_path_factory.mktemp("policy")
    dataset = root / "dataset.jsonl"
    synthesize_playtime(dataset, minutes=1.0, seed=21)
    train(
        TrainConfig(
            dataset=str(dataset),
            run_name="policy",
            models_dir=str(root / "models"),
            steps=30,
            checkpoint_every=30,
            batch_size=32,
            log_every=30,
        )
    )
    return root / "models"


def snapshot(**overrides) -> dict:
    data = {
        "position": {"x": 10.0, "y": 64.0, "z": -5.0},
        "yaw": 0.0,  # radians, like mineflayer
        "pitch": 0.0,
        "health": 20.0,
        "food": 18.0,
        "players": [{"name": "Steve", "x": 12.0, "y": 64.0, "z": -4.0, "distance": 2.3}],
        "selected_slot": 2,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------- loading
def test_policy_loads_the_newest_run_by_name(trained: Path):
    policy = TrainedPolicy.load("policy", trained)
    assert policy.run.run == "policy"
    assert policy.run.engine == "mlp"
    assert policy.run.checkpoint.is_file()
    described = policy.describe()
    assert described["feature_dim"] == 215
    assert described["actions"][0] == "forward"
    assert described["trained_on"]


def test_policy_loads_from_a_run_directory_or_checkpoint(trained: Path):
    by_dir = TrainedPolicy.load(trained / "policy", trained)
    by_file = TrainedPolicy.load(trained / "policy" / "latest.npz", trained)
    assert by_dir.run.run == by_file.run.run == "policy"
    assert find_run("policy", trained).checkpoint.name == "latest.npz"


def test_missing_model_explains_how_to_make_one(trained: Path, tmp_path: Path):
    with pytest.raises(PolicyError, match="no trained model"):
        TrainedPolicy.load("nope", trained)
    with pytest.raises(PolicyError, match="train one"):
        TrainedPolicy.load("nope", tmp_path / "empty")


def test_feature_version_mismatch_is_refused(trained: Path):
    policy = TrainedPolicy.load("policy", trained)
    policy.run.card["feature_version"] = 99
    with pytest.raises(PolicyError, match="feature layout"):
        TrainedPolicy(policy.run, policy.model)


# -------------------------------------------------------------------- observing
def test_observe_converts_radians_to_degrees_if_needed(trained: Path):
    policy = TrainedPolicy.load("policy", trained)
    observed = policy.observe(snapshot(yaw=math.pi / 2, pitch=math.radians(30)))
    assert observed["yaw"] == pytest.approx(90.0, abs=0.01)
    assert observed["pitch"] == pytest.approx(30.0, abs=0.01)
    assert observed["blocks"], "the encoder must always receive a block neighbourhood"
    assert len(observed["blocks"]) == 27
    assert observed["nearby"][0]["type"] == "minecraft:player"
    assert observed["nearby"][0]["player"] is True


def test_observe_tracks_distance_moved_between_snapshots(trained: Path):
    policy = TrainedPolicy.load("policy", trained)
    policy.observe(snapshot())
    observed = policy.observe(snapshot(position={"x": 13.0, "y": 64.0, "z": -5.0}))
    assert observed["moved"] == pytest.approx(3.0, abs=0.01)


# -------------------------------------------------------------------- inference
def test_prediction_is_a_valid_action_with_probabilities(trained: Path):
    policy = TrainedPolicy.load("policy", trained)
    prediction = policy.predict(snapshot())
    assert prediction.action in policy.actions
    assert sum(prediction.probs.values()) == pytest.approx(1.0, abs=1e-4)
    assert prediction.source == "trained:policy"
    payload = prediction.to_dict()
    assert set(payload["motion"]) == {"dyaw", "dpitch", "forward", "strafe"}


def test_temperature_sampling_mixes_actions(trained: Path):
    greedy = TrainedPolicy.load("policy", trained, temperature=0.0)
    hot = TrainedPolicy.load("policy", trained, temperature=1.5)
    observations = [greedy.observe(snapshot(position={"x": float(index), "y": 64.0, "z": 0.0})) for index in range(30)]
    greedy_actions = {greedy.predict_observation(observation).action for observation in observations}
    hot_actions = {hot.predict_observation(observation).action for observation in observations}
    assert len(greedy_actions) == 1  # deterministic
    assert len(hot_actions) >= 2  # the same model, sampled


def test_decide_returns_an_agent_decision(trained: Path):
    from pymc_bot.agent import Decision

    policy = TrainedPolicy.load("policy", trained)
    decision = policy.decide(snapshot())
    assert isinstance(decision, Decision)
    assert decision.action in policy.actions
    assert decision.source == "trained:policy"
    assert "seconds" in decision.params


# -------------------------------------------------------------------- execution
@pytest.fixture()
def live_bot(store: ConfigStore, log: EventLog, wait_for):
    from pymc_bot.backends import SimulatedBackend

    store.update({"minecraft": {"backend": "simulated", "username": "TrainedTest"}})
    bot = MinecraftBot(store, log, auto_reconnect=False, backend=SimulatedBackend(log, ambient_chat=False))
    bot.start()
    assert wait_for(lambda: bot.connected, timeout=5.0)
    yield bot
    bot.stop()


def _dist(bot: MinecraftBot) -> float:
    position = bot.snapshot()["position"]
    return math.hypot(position["x"], position["z"])


def test_act_walks_the_bot_forward(trained: Path, live_bot: MinecraftBot):
    policy = TrainedPolicy.load("policy", trained, step_seconds=0.3)
    decision = _decision("forward")
    ok, detail = policy.act(live_bot, decision)
    assert ok and "forward" in detail
    assert _dist(live_bot) > 0.5
    # the burst must release the key again
    assert live_bot.backend._controls["forward"] is False


def test_act_releases_controls_even_when_cancelled(trained: Path, live_bot: MinecraftBot):
    policy = TrainedPolicy.load("policy", trained, step_seconds=3.0)
    live_bot.cancel_actions()
    # durations are clamped to a sane maximum, so 30s becomes a 3s burst that stops early
    ok, detail = policy.act(live_bot, _decision("forward", seconds=30.0))
    assert ok and "3.00s" in detail
    assert live_bot.backend._controls["forward"] is False
    live_bot.clear_cancel()


def test_act_turns_the_view_in_degrees(trained: Path, live_bot: MinecraftBot):
    policy = TrainedPolicy.load("policy", trained)
    live_bot.look(0.0, 0.0)
    decision = _decision("look", dyaw=20.0, dpitch=10.0)
    ok, detail = policy.act(live_bot, decision)
    assert ok
    assert "20.0deg" in detail
    angle = live_bot.snapshot()["yaw"]
    # the backend speaks radians: 20 degrees must not be handed over as 20 radians
    assert angle == pytest.approx(math.radians(20.0), abs=0.02)


def test_act_clamps_runaway_turns(trained: Path, live_bot: MinecraftBot):
    policy = TrainedPolicy.load("policy", trained)
    live_bot.look(0.0, 0.0)
    # A broken regression head must not spin the bot on the spot.
    policy.act(live_bot, _decision("look", dyaw=170.0, dpitch=90.0))
    angle = live_bot.snapshot()["yaw"]
    assert abs(math.degrees(angle)) <= policy.MAX_YAW_PER_BURST + 0.5


def test_act_handles_every_action_in_the_vocabulary(trained: Path, live_bot: MinecraftBot):
    policy = TrainedPolicy.load("policy", trained, step_seconds=0.15)
    for action in policy.actions:
        ok, detail = policy.act(live_bot, _decision(action))
        assert ok is True, action
        assert detail, action
    assert live_bot.backend.arm_swings >= 2  # attack + use swing the arm
    assert live_bot.backend._controls["forward"] is False
    assert live_bot.backend._controls["sneak"] is False


def test_act_reports_unknown_actions_and_dead_bots(trained: Path, live_bot: MinecraftBot):
    policy = TrainedPolicy.load("policy", trained)
    ok, detail = policy.act(live_bot, _decision("teleport"))
    assert ok is False and "unknown action" in detail
    live_bot.stop()
    ok, detail = policy.act(live_bot, _decision("forward"))
    assert ok is False and "not connected" in detail


def _decision(action: str, **params):
    from pymc_bot.agent import Decision

    return Decision(action=action, params=params or {"seconds": 0.3}, source="trained:policy")


# --------------------------------------------------------------- ollama export
def test_policy_priors_and_modelfile(trained: Path):
    policy = TrainedPolicy.load("policy", trained)
    observations = [policy.observe(snapshot(position={"x": float(index), "y": 64.0, "z": 0.0})) for index in range(5)]
    priors = policy_priors(policy, observations)
    assert sum(priors["priorities"].values()) == 5
    assert priors["examples"] and priors["examples"][0]["action"]

    modelfile = build_ollama_modelfile(policy, priors, "llama3.2")
    assert modelfile.startswith("FROM llama3.2")
    assert "forward" in modelfile
    assert "Minecraft" in modelfile


# ------------------------------------------------------------------ permission gates
def test_mask_actions_renormalises_the_rest(trained: Path):
    from pymc_bot.local_model import mask_actions

    masked = mask_actions({"forward": 0.5, "attack": 0.3, "none": 0.2}, {"forward", "none"})
    assert masked["attack"] == 0.0
    assert masked["forward"] + masked["none"] == pytest.approx(1.0)
    assert masked["forward"] > masked["none"]


def test_mask_actions_forces_idle_when_everything_is_forbidden(trained: Path):
    from pymc_bot.local_model import mask_actions

    masked = mask_actions({"forward": 0.9, "attack": 0.1, "none": 0.0}, {"look"})
    assert masked["none"] == 1.0
    assert max(masked, key=masked.get) == "none"


def test_decide_never_picks_a_forbidden_action(trained: Path):
    policy = TrainedPolicy.load("policy", trained, temperature=1.0)  # sampling on
    allowed = {"look", "none", "forward"}
    actions = {policy.decide(snapshot(), allowed=allowed).action for _ in range(40)}
    assert actions <= allowed
    assert actions  # something was chosen


def test_prediction_reports_the_masked_probabilities(trained: Path):
    policy = TrainedPolicy.load("policy", trained, temperature=1.0)
    prediction = policy.predict(snapshot(), allowed={"none"})
    assert prediction.action == "none"
    assert prediction.probs["none"] == pytest.approx(1.0)
