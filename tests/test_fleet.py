"""Premium accounts (Microsoft auth) and the multi-player fleet."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pymc_bot.config import AppSettings, ConfigStore, FleetBotSpec, FleetSettings, MinecraftSettings
from pymc_bot.events import EventLog
from pymc_bot.fleet import BotFleet, FleetError, expand_names, validate_username


# ------------------------------------------------------------------ validation
def test_offline_usernames_follow_minecraft_rules():
    assert validate_username("PyMC_Bot_1", "offline") == "PyMC_Bot_1"
    for bad in ["", "  ", "has space", "way_way_way_too_long", "emoji🙂", "semi;colon"]:
        with pytest.raises(FleetError):
            validate_username(bad, "offline")


def test_offline_auth_rejects_an_email():
    with pytest.raises(FleetError) as excinfo:
        validate_username("player@example.com", "offline")
    assert "offline" in str(excinfo.value).lower()


def test_premium_usernames_must_be_an_email():
    assert validate_username("player@example.com", "microsoft") == "player@example.com"
    for bad in ["PyMC_Bot", "player@localhost", "no-at-sign.com"]:
        with pytest.raises(FleetError):
            validate_username(bad, "microsoft")


def test_config_accepts_premium_and_rejects_a_bad_email():
    settings = MinecraftSettings(username="someone@example.com", auth="microsoft")
    assert settings.account_label == "premium (microsoft)"
    with pytest.raises(ValidationError):
        MinecraftSettings(username="PyMC_Bot", auth="microsoft")


def test_config_offline_rules_still_apply():
    with pytest.raises(ValidationError):
        MinecraftSettings(username="someone@example.com", auth="offline")
    assert MinecraftSettings(username="Bot_1").account_label == "offline (cracked)"


def test_fleet_settings_validation():
    with pytest.raises(ValidationError):
        FleetSettings(name_pattern="PyMC_Bot")  # needs {n}
    with pytest.raises(ValidationError):
        FleetSettings(count=0)
    with pytest.raises(ValidationError):
        FleetSettings(max_bots=10_000)
    assert FleetSettings(name_pattern="Bot_{n}").name_pattern == "Bot_{n}"


# ------------------------------------------------------------------- name gen
def test_expand_names_patterns():
    assert expand_names("Bot_{n}", 3) == ["Bot_1", "Bot_2", "Bot_3"]
    assert expand_names("Player{i}", 2) == ["Player1", "Player2"]
    assert expand_names("{name}_alt{n}", 2, base_name="Main") == ["Main_alt1", "Main_alt2"]


def test_expand_names_skips_taken_names():
    names = expand_names("Bot_{n}", 3, taken=["Bot_1", "BOT_2"])
    assert names == ["Bot_3", "Bot_4", "Bot_5"]


def test_expand_names_handles_empty_pattern():
    with pytest.raises(FleetError):
        expand_names("Bot", 2, taken=["Bot", "Bot"])


# --------------------------------------------------------------------- fixtures
@pytest.fixture()
def fleet(tmp_path, log, wait_for):
    store = ConfigStore(tmp_path / "fleet.json")
    store.load()
    store.update(
        {
            "minecraft": {"backend": "simulated", "username": "MainBot"},
            "fleet": {"max_bots": 8, "count": 3, "name_pattern": "Pop_{n}", "stagger_seconds": 0.05},
        }
    )
    instance = BotFleet(store, log)
    instance.primary.start()
    assert wait_for(lambda: instance.primary.connected, timeout=5)
    yield instance
    instance.shutdown()


def _wait_connected(wait_for, fleet: BotFleet, expected: int, timeout: float = 8.0) -> bool:
    return wait_for(lambda: fleet.status()["connected"] >= expected, timeout=timeout)


# ------------------------------------------------------------------- the fleet
def test_populate_adds_many_players(fleet: BotFleet, wait_for):
    members = fleet.spawn_many(count=3)
    assert [m.username for m in members] == ["Pop_1", "Pop_2", "Pop_3"]
    assert fleet.status()["size"] == 4  # primary + 3
    assert _wait_connected(wait_for, fleet, 4)

    status = fleet.status()
    assert status["connected"] == 4
    assert {b["username"] for b in status["bots"]} >= {"Pop_1", "Pop_2", "Pop_3", "MainBot"}
    assert all(b["auth"] == "offline" for b in status["bots"])
    assert status["bots"][0]["sponsor"] == "primary"


def test_spawn_single_premium_player(fleet: BotFleet, wait_for):
    member = fleet.spawn("premium.player@example.com", auth="microsoft", ai="off")
    assert member.auth == "microsoft"
    assert fleet.status()["size"] == 2
    assert wait_for(lambda: member.state in ("connected", "error"), timeout=8)
    entry = next(b for b in fleet.status()["bots"] if b["configured_username"] == "premium.player@example.com")
    assert entry["auth"] == "microsoft"
    assert entry["ai"] == "off"


def test_spawn_rejects_bad_input(fleet: BotFleet):
    with pytest.raises(FleetError):
        fleet.spawn("invalid name!", auth="offline")
    with pytest.raises(FleetError):
        fleet.spawn("not-an-email", auth="microsoft")
    with pytest.raises(FleetError):
        fleet.spawn("Dancer", auth="offline", ai="disco")


def test_duplicate_names_are_rejected(fleet: BotFleet):
    fleet.spawn("Twin", start=False)
    with pytest.raises(FleetError) as excinfo:
        fleet.spawn("twin", start=False)
    assert "already in the fleet" in str(excinfo.value)
    with pytest.raises(FleetError):
        fleet.spawn("MainBot", start=False)  # the primary bot's name


def test_fleet_respects_max_bots(fleet: BotFleet):
    fleet.store.update({"fleet": {"max_bots": 3}})
    fleet.spawn("One", start=False)
    fleet.spawn("Two", start=False)
    with pytest.raises(FleetError) as excinfo:
        fleet.spawn("Three", start=False)
    assert "full" in str(excinfo.value)


def test_populate_beyond_capacity_is_rejected(fleet: BotFleet):
    fleet.store.update({"fleet": {"max_bots": 3}})
    with pytest.raises(FleetError):
        fleet.spawn_many(count=5)


def test_select_and_find_control_the_right_bot(fleet: BotFleet, wait_for):
    fleet.spawn_many(count=2)
    assert _wait_connected(wait_for, fleet, 3)

    assert fleet.selected_bot() is fleet.primary
    fleet.select("Pop_2")
    assert fleet.selected_bot().username == "Pop_2"
    assert fleet.status()["selected"] == "Pop_2"
    assert fleet.find("pop_1") is not None          # case-insensitive
    assert fleet.find("nobody") is None
    with pytest.raises(FleetError):
        fleet.select("nobody")


def test_actions_can_target_one_bot_or_all(fleet: BotFleet, wait_for):
    fleet.spawn_many(count=2)
    assert _wait_connected(wait_for, fleet, 3)

    assert fleet.action_target("selected") == [fleet.primary]
    assert len(fleet.action_target("all")) == 3
    assert fleet.action_target("Pop_1")[0].username == "Pop_1"
    with pytest.raises(FleetError):
        fleet.action_target("ghost")

    assert fleet.broadcast("hello everyone") == 3


def test_per_bot_ai_mode_is_honoured(fleet: BotFleet, wait_for, log: EventLog):
    fleet.spawn_many(count=2, ai="heuristic")
    assert _wait_connected(wait_for, fleet, 3)
    assert wait_for(lambda: all(entry["ai_running"] for entry in fleet.status()["bots"][1:]), timeout=5)

    # the primary bot has its own agent; a fleet member's agent prefers heuristics
    member = fleet.get("Pop_1")
    assert member is not None and member.agent is not None
    assert member.agent.prefer_ollama is False


def test_stop_remove_and_restart_members(fleet: BotFleet, wait_for):
    fleet.spawn_many(count=2)
    assert _wait_connected(wait_for, fleet, 3)

    assert fleet.stop_bot("Pop_1") is True
    assert wait_for(lambda: fleet.get("Pop_1").state == "stopped", timeout=5)
    assert fleet.stop_bot("ghost") is False

    fleet.start_bot("Pop_1")
    assert _wait_connected(wait_for, fleet, 3)

    assert fleet.remove_bot("Pop_2") is True
    assert fleet.get("Pop_2") is None
    assert fleet.remove_bot("Pop_2") is False
    assert fleet.status()["size"] == 2


def test_stop_all_and_start_all(fleet: BotFleet, wait_for):
    fleet.spawn_many(count=3)
    assert _wait_connected(wait_for, fleet, 4)

    assert fleet.stop_all() == 3
    assert fleet.status()["connected"] == 1  # only the primary is left connected

    assert fleet.start_all() == 3
    assert _wait_connected(wait_for, fleet, 4)


def test_roster_is_persisted_and_restored(tmp_path, log, wait_for):
    path = tmp_path / "roster.json"
    store = ConfigStore(path)
    store.load()
    store.update({"minecraft": {"backend": "simulated"}, "fleet": {"restore_on_start": True}})

    first = BotFleet(store, log)
    first.spawn_many(count=2)
    assert wait_for(lambda: first.status()["connected"] == 2, timeout=8)
    saved = ConfigStore(path).load().fleet.roster
    assert [spec.username for spec in saved] == ["PyMC_Bot_1", "PyMC_Bot_2"]
    first.shutdown()

    # a fresh instance (like a restarted panel) brings the players back
    second = BotFleet(ConfigStore(path), log)
    assert second.status()["roster_size"] == 2
    assert second.restore_roster() == 2
    assert wait_for(lambda: second.status()["connected"] == 2, timeout=8)
    second.shutdown()


def test_restore_is_opt_in(tmp_path, log):
    path = tmp_path / "roster_off.json"
    store = ConfigStore(path)
    store.load()
    store.update({"minecraft": {"backend": "simulated"}, "fleet": {"restore_on_start": False, "roster": [
        FleetBotSpec(username="Remembered").model_dump()
    ]}})
    instance = BotFleet(store, log)
    assert instance.restore_roster() == 0
    assert instance.status()["extra_bots"] == 0
    instance.shutdown()


def test_roster_survives_a_bad_spec(tmp_path, log):
    store = ConfigStore(tmp_path / "r.json")
    store.load()
    store.update({"minecraft": {"backend": "simulated"}, "fleet": {
        "restore_on_start": True,
        "roster": [{"username": "bad name!", "auth": "offline", "ai": "heuristic"}],
    }})
    instance = BotFleet(store, log)
    assert instance.restore_roster() == 0  # invalid name is skipped, not fatal
    instance.shutdown()


def test_fleet_disabled_blocks_new_players(fleet: BotFleet):
    fleet.store.update({"fleet": {"enabled": False}})
    with pytest.raises(FleetError) as excinfo:
        fleet.spawn("Nope", start=False)
    assert "disabled" in str(excinfo.value)


def test_chatter_uses_the_bots(fleet: BotFleet, wait_for, log: EventLog):
    fleet.store.update({"fleet": {"chatter": True, "chatter_interval": 1.0}})
    fleet.spawn_many(count=2)
    assert _wait_connected(wait_for, fleet, 3)
    fleet._ensure_chatter()
    assert wait_for(
        lambda: any("hey everyone" in event["message"] or "diamonds" in event["message"] or "nice server" in event["message"]
                   or "build something" in event["message"] or "brb mining" in event["message"]
                   or "looks great" in event["message"] for event in log.tail(200)),
        timeout=6,
    )


def test_shutdown_stops_everything(fleet: BotFleet, wait_for):
    fleet.spawn_many(count=2)
    assert _wait_connected(wait_for, fleet, 3)
    fleet.shutdown()
    assert fleet.members == []
    assert fleet.primary.state == "disconnected"


def test_unique_names_are_generated_for_repeated_batches(fleet: BotFleet, wait_for):
    fleet.spawn_many(count=2)
    assert _wait_connected(wait_for, fleet, 3)
    second = fleet.spawn_many(count=2)
    assert [m.username for m in second] == ["Pop_3", "Pop_4"]
    assert _wait_connected(wait_for, fleet, 5)


def test_status_reports_premium_and_offline_side_by_side(fleet: BotFleet, wait_for):
    fleet.spawn("buyer@example.com", auth="microsoft", ai="off", start=False)
    fleet.spawn("Cracked_1", auth="offline", start=False)
    labels = {b["configured_username"]: b["auth"] for b in fleet.status()["bots"]}
    assert labels["buyer@example.com"] == "microsoft"
    assert labels["Cracked_1"] == "offline"


def test_store_and_fleet_share_one_config_file(tmp_path, log):
    """The roster write must not clobber other settings."""
    path = tmp_path / "shared.json"
    store = ConfigStore(path)
    store.load()
    store.update({"minecraft": {"backend": "simulated", "host": "mc.example.net"}, "ollama": {"model": "llama3.2"}})
    instance = BotFleet(store, log)
    instance.spawn("Persisted", start=False)
    reloaded = ConfigStore(path).load()
    assert reloaded.minecraft.host == "mc.example.net"
    assert reloaded.ollama.model == "llama3.2"
    assert [spec.username for spec in reloaded.fleet.roster] == ["Persisted"]
    instance.shutdown()


def test_fleet_does_not_duplicate_primary_name(tmp_path, log):
    store = ConfigStore(tmp_path / "prim.json")
    store.load()
    store.update({"minecraft": {"backend": "simulated", "username": "Solo"}})
    instance = BotFleet(store, log)
    with pytest.raises(FleetError):
        instance.spawn("Solo", start=False)
    instance.shutdown()


def test_app_settings_include_fleet_defaults():
    settings = AppSettings()
    assert settings.fleet.enabled is True
    assert settings.fleet.name_pattern == "PyMC_Bot_{n}"
    assert settings.fleet.ai_mode == "heuristic"
    assert settings.fleet.chatter is False
    assert settings.minecraft.auto_rejoin is True


def test_generated_names_stay_unique_across_batches():
    seen: set[str] = set()
    for _ in range(3):
        for name in expand_names("B{n}", 3, taken=seen):
            assert name not in seen
            seen.add(name)
    assert len(seen) == 9
