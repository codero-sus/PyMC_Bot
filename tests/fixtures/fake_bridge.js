#!/usr/bin/env node
/**
 * fake_bridge.js -- a minimal stand-in for minecraft_bridge.js that speaks the
 * exact same newline-delimited JSON protocol but never touches Minecraft.
 *
 * It lets the Python side (NodeBridgeBackend) be tested without installing
 * mineflayer or running a server:
 *
 *   node tests/fixtures/fake_bridge.js
 */
'use strict';

const readline = require('readline');

let bot = { connected: false, controls: {}, position: { x: 0, y: 64, z: 0 }, yaw: 0, pitch: 0, chats: [], digs: [] };
let profilesFolder = null;
let stateTimer = null;

const send = (payload) => process.stdout.write(JSON.stringify(payload) + '\n');

function state() {
  return {
    status: bot.connected ? 'connected' : 'disconnected',
    username: bot.ingameName || '',
    position: bot.position,
    yaw: bot.yaw,
    pitch: bot.pitch,
    health: bot.connected ? 19 : 20,
    food: 18,
    dimension: 'overworld',
    time_of_day: 'day',
    players: bot.connected ? [{ name: 'Steve', x: 5, y: 64, z: 0, distance: 5 }] : [],
    inventory: bot.digs.map((block) => ({ name: block, count: 1 })),
  };
}

const COMMANDS = {
  ping: () => ({ pong: true, connected: bot.connected }),
  state: () => state(),
  connect: (params) => {
    profilesFolder = params.profiles_folder || null;
    const premium = params.auth === 'microsoft';
    // offline servers use the given name; premium accounts keep their own
    bot.ingameName = premium ? 'PremiumPlayer' : params.username;
    if (params.host === 'badhost' || (premium && params.username === 'bad@example.com')) {
      const error = premium
        ? 'Failed to obtain profile data for bad@example.com, does the account own minecraft?'
        : 'getaddrinfo ENOTFOUND badhost';
      setTimeout(() => send({ event: 'error', message: error }), 20);
      return { connecting: true, connect_timeout_ms: 1000 };
    }
    if (premium) {
      send({
        event: 'msa_code',
        user_code: 'FAKE-CODE-1234',
        verification_uri: 'https://www.microsoft.com/link',
        expires_in: 900,
        message: 'Sign in to Microsoft to finish the premium login.',
      });
    }
    setTimeout(() => {
      bot.connected = true;
      send({ event: 'login', username: bot.ingameName });
      send({ event: 'spawn' });
      stateTimer = setInterval(() => send({ event: 'state', state: state() }), 100);
    }, 50);
    return { connecting: true, profiles_folder: profilesFolder };
  },
  say: (params) => { bot.chats.push(String(params.text)); setTimeout(() => send({ event: 'chat', username: 'Steve', message: 'nice' }), 10); return { sent: true }; },
  command: (params) => ({ sent: true, command: params.text }),
  control: (params) => { bot.controls[params.name] = params.state; if (params.name === 'forward' && params.state) bot.position.z += 1; return { control: params.name }; },
  look: (params) => { bot.yaw = params.yaw; bot.pitch = params.pitch; return { looked: true }; },
  stop_motion: () => { bot.controls = {}; return { stopped: true }; },
  pathfind_to: () => { throw new Error('pathfinder_unavailable'); },
  stop_path: () => ({ stopped: true }),
  dig: (params) => { bot.digs.push(String(params.block)); return { dug: true, block: params.block }; },
  eat: () => ({ ate: true, item: 'bread' }),
  attack: (params) => ({ attacked: params.player }),
  jump: () => ({ jumped: true }),
  disconnect: () => { bot.connected = false; if (stateTimer) clearInterval(stateTimer); stateTimer = null; return { disconnected: true }; },
};

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  line = line.trim();
  if (!line) return;
  let request;
  try { request = JSON.parse(line); } catch (err) { send({ event: 'log', level: 'warn', message: 'bad json' }); return; }
  const handler = COMMANDS[request.cmd];
  if (!handler) { send({ id: request.id, ok: false, error: `unknown command: ${request.cmd}` }); return; }
  try {
    const result = handler(request.params || {});
    send({ id: request.id, ok: true, result: result || {} });
  } catch (err) {
    send({ id: request.id, ok: false, error: err.message });
  }
});

rl.on('close', () => process.exit(0));

send({ event: 'ready', caps: { pathfinder: false, mineflayer: false }, node: process.version });
send({ event: 'log', level: 'info', message: 'fake bridge ready' });
