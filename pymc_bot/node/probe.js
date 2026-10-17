#!/usr/bin/env node
/*
 * PyMC_Bot server probe - a Minecraft "server list ping" without joining.
 *
 * Used by the bridge before every join and by the Python side (`python -m pymc_bot probe`,
 * the panel's "Check server" button). It follows SRV records like the real client does, so
 * addresses such as `myserver.aternos.me` work without a port, and it recognises the
 * placeholder that free hosts (Aternos, exaroton, Minehut...) answer with while a server is
 * asleep or starting ("● Offline", "● Starting", "This server is offline").
 *
 *   node probe.js <host> [port] [timeout_ms]   -> prints one JSON object and exits
 *
 * Made with love by codero-sus - https://github.com/codero-sus
 */
'use strict';

const SLEEP_WORDS = /\b(offline|starting|waiting|loading|preparing|stopping|saving|restarting|queued?|sleeping|hibernat\w*|asleep)\b/i;
const SLEEP_MOTD = /\b(?:server|it) is (?:currently )?(offline|starting|sleeping|stopping|asleep|hibernating)\b|\b(hibernat\w*|asleep)\b/i;

/** Plain text of a chat component / legacy-formatted string (drops `§x` codes). */
function plainText(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value.replace(/\u00a7[0-9a-fk-orx]/gi, '').trim();
  if (Array.isArray(value)) return value.map(plainText).join('').trim();
  if (typeof value === 'object') {
    let text = plainText(value.text || value.translate || '');
    if (Array.isArray(value.extra)) text += value.extra.map(plainText).join('');
    return text.replace(/\u00a7[0-9a-fk-orx]/gi, '').trim();
  }
  return String(value);
}

/**
 * Returns the sleeping state word ("offline", "starting", ...) when a ping answer is a free-host
 * placeholder rather than a real server, otherwise null. A real server always reports a numeric
 * Minecraft version in its version name and a positive protocol.
 */
function sleepingState(versionName, protocol, motd) {
  const name = plainText(versionName);
  const text = plainText(motd);
  const hasNumber = /\d+\.\d+/.test(name);
  const badProtocol = !(Number(protocol) > 0);
  const nameWord = name.match(SLEEP_WORDS);
  if (nameWord && (!hasNumber || badProtocol || /[\u25cf\u2022]/.test(name))) return nameWord[1].toLowerCase();
  const motdWord = text.match(SLEEP_MOTD);
  if (motdWord && (!hasNumber || badProtocol)) return (motdWord[1] || motdWord[2]).toLowerCase();
  return null;
}

/** Ping `host:port`. Resolves { ok, ... } and never rejects. */
function probe(host, port, timeoutMs = 8000, minecraftProtocol = null) {
  let mc = minecraftProtocol;
  if (!mc) {
    try {
      mc = require('minecraft-protocol');
    } catch (err) {
      return Promise.resolve({ ok: false, error: `minecraft-protocol is not installed (run npm install): ${err.message}` });
    }
  }
  const options = { host, port: Number(port) || 25565, closeTimeout: timeoutMs };
  const started = Date.now();
  return new Promise((resolve) => {
    let finished = false;
    const finish = (value) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      resolve(value);
    };
    const timer = setTimeout(() => finish({ ok: false, error: `no answer within ${Math.round(timeoutMs / 1000)}s` }), timeoutMs);
    const onResult = (err, res) => {
      if (err || !res) {
        finish({ ok: false, error: err ? String(err.message || err) : 'empty answer' });
        return;
      }
      const version = res.version || {};
      const players = res.players || {};
      const motd = plainText(res.description);
      const state = sleepingState(version.name, version.protocol, res.description);
      // minecraft-protocol rewrites options.host/port when it followed an SRV record
      const resolved = options.host !== host || options.port !== (Number(port) || 25565)
        ? { host: options.host, port: options.port } : null;
      finish({
        ok: true,
        online: !state,
        asleep: Boolean(state),
        state: state || 'online',
        version: plainText(version.name),
        protocol: Number(version.protocol),
        players: {
          online: Number(players.online) || 0,
          max: Number(players.max) || 0,
          sample: Array.isArray(players.sample) ? players.sample.map((p) => plainText(p.name)).filter(Boolean) : [],
        },
        motd,
        latency: Number(res.latency) || Date.now() - started,
        resolved,
      });
    };
    try {
      const maybe = mc.ping(options, onResult);
      if (maybe && typeof maybe.catch === 'function') maybe.catch((err) => onResult(err));
    } catch (err) {
      onResult(err);
    }
  });
}

module.exports = { probe, sleepingState, plainText };

if (require.main === module) {
  const [host, port, timeout] = process.argv.slice(2);
  if (!host) {
    process.stdout.write(JSON.stringify({ ok: false, error: 'usage: probe.js <host> [port] [timeout_ms]' }) + '\n');
    process.exit(2);
  }
  probe(host, Number(port) || 25565, Number(timeout) || 8000).then((result) => {
    process.stdout.write(JSON.stringify(result) + '\n');
    process.exit(0);
  });
}
