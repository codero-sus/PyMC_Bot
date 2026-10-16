// End-to-end check of the bridge's version handling, run in CI after `npm install`.
//
// Starts a local offline-mode server with minecraft-protocol that speaks the newest version the
// installed mineflayer supports but *advertises* a newer one in the server list ping (like a 26.3
// server with ViaBackwards). The bridge must detect that, report `version_fallback`, and still log in
// instead of failing with "No data available for version ...".
//
//   node scripts/check_version_fallback.mjs
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const mc = require('minecraft-protocol');
const { latestSupportedVersion } = require('mineflayer/lib/version');
const mcData = require('minecraft-data');

// newer than anything mineflayer knows; override with e.g. PYMC_ADVERTISE=26.3:777
const [advName, advProtocol] = (process.env.PYMC_ADVERTISE || '99.1:99999').split(':');
const advertised = { name: advName, protocol: Number(advProtocol) };
const port = 25000 + Math.floor(Math.random() * 2000);
const server = mc.createServer({
  host: '127.0.0.1',
  port,
  'online-mode': false,
  version: latestSupportedVersion,
  beforePing: (response) => ({ ...response, version: advertised }),
});

let joined = false;
server.on('playerJoin', () => { joined = true; });
server.on('connection', (client) => client.on('error', () => {})); // the bridge hangs up at the end

const bridge = spawn(process.execPath, [path.join(root, 'pymc_bot/node/minecraft_bridge.js')], {
  cwd: root,
  stdio: ['pipe', 'pipe', 'inherit'],
});
const events = [];
let buffer = '';
bridge.stdout.on('data', (chunk) => {
  buffer += chunk;
  let index;
  while ((index = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (!line) continue;
    const message = JSON.parse(line);
    events.push(message);
    if (message.event === 'ready') {
      bridge.stdin.write(JSON.stringify({
        id: 1, cmd: 'connect', params: { host: '127.0.0.1', port, username: 'FallbackCheck', auth: 'offline' },
      }) + '\n');
    }
  }
});

function finish(code, why) {
  console.log(why);
  try { bridge.kill(); } catch (err) { /* already gone */ }
  server.close();
  setTimeout(() => process.exit(code), 200);
}

const started = Date.now();
const timer = setInterval(() => {
  const ready = events.find((e) => e.event === 'ready');
  const fallback = events.find((e) => e.event === 'version_fallback');
  const error = events.find((e) => e.event === 'error');
  if (error) {
    clearInterval(timer);
    finish(1, `FAIL: bridge error: ${error.message}`);
  } else if (fallback && joined) {
    clearInterval(timer);
    const caps = ready.caps.minecraft || {};
    if (fallback.using !== latestSupportedVersion || fallback.server_version !== advertised.name) {
      finish(1, `FAIL: unexpected fallback ${JSON.stringify(fallback)}`);
    } else if (caps.latest !== latestSupportedVersion || !mcData(caps.latest)) {
      finish(1, `FAIL: ready caps report ${JSON.stringify(caps)}`);
    } else {
      finish(0, `OK: server advertised ${advertised.name}; the bot joined as ${fallback.using} `
        + `(mineflayer supports ${caps.oldest} - ${caps.latest}).`);
    }
  } else if (Date.now() - started > 30000) {
    clearInterval(timer);
    finish(1, `FAIL: timed out (joined=${joined}); events: ${events.map((e) => e.event).join(', ')}`);
  }
}, 100);
