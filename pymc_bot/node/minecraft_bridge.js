#!/usr/bin/env node
/**
 * minecraft_bridge.js -- Node-side Minecraft protocol adapter for PyMC_Bot.
 *
 * Why this file exists: the mature Minecraft *client* protocol library
 * (mineflayer) only ships for Node.js, so this ~300 line adapter is the one
 * piece of JS. Everything else -- configuration, movement logic, the AI brain,
 * the web control panel -- is Python.
 *
 * Protocol (newline-delimited JSON):
 *   stdin   {"id": 1, "cmd": "connect", "params": {...}}
 *   stdout  {"id": 1, "ok": true, "result": {...}}      // command reply
 *   stdout  {"event": "state", "state": {...}}          // unsolicited events
 *           events: ready, log, state, chat, spawn, end, kicked, error, exit
 */

'use strict';

const readline = require('readline');

let mineflayer = null;
let pathfinder = null;   // optional, enables real A* pathfinding
let goals = null;
let bot = null;
let stateTimer = null;

const CONNECT_TIMEOUT_MS = 30000;

/* ------------------------------------------------------------------ output */
function send(payload) {
  process.stdout.write(JSON.stringify(payload) + '\n');
}

function log(level, message) {
  send({ event: 'log', level, message: String(message) });
}

// Keep stdout strictly for the JSON protocol.
console.log = (...args) => log('debug', args.join(' '));
console.warn = (...args) => log('warn', args.join(' '));
console.error = (...args) => log('error', args.join(' '));

process.on('uncaughtException', (err) => {
  send({ event: 'error', message: `uncaught: ${err && err.stack ? err.stack : err}` });
});
process.on('unhandledRejection', (err) => {
  send({ event: 'error', message: `unhandled rejection: ${err && err.stack ? err.stack : err}` });
});
process.on('exit', () => send({ event: 'exit' }));

/* -------------------------------------------------------------- module load */
try {
  mineflayer = require('mineflayer');
} catch (err) {
  send({ event: 'error', message: 'mineflayer is not installed (run `npm install`). ' + err.message });
  process.exit(1);
}
try {
  pathfinder = require('mineflayer-pathfinder');
  goals = pathfinder.goals;
} catch (err) {
  pathfinder = null;
  goals = null;
  log('warn', 'mineflayer-pathfinder not found; Python-side steering will be used.');
}

/* ------------------------------------------------------------------- helpers */
function playerPosition(name) {
  if (!bot || !bot.players) return null;
  const player = bot.players[name];
  if (!player || !player.entity) return null;
  const pos = player.entity.position;
  return { x: pos.x, y: pos.y, z: pos.z };
}

function buildState() {
  if (!bot || !bot.entity) {
    return { status: bot ? 'connecting' : 'disconnected' };
  }
  const pos = bot.entity.position;
  const players = [];
  for (const name of Object.keys(bot.players)) {
    const player = bot.players[name];
    if (!player.entity || name === bot.username) continue;
    const p = player.entity.position;
    players.push({
      name,
      x: Number(p.x.toFixed(2)),
      y: Number(p.y.toFixed(2)),
      z: Number(p.z.toFixed(2)),
      distance: Number(pos.distanceTo(p).toFixed(2)),
    });
  }
  players.sort((a, b) => a.distance - b.distance);

  const inventory = bot.inventory
    ? bot.inventory.items().map((item) => ({ name: item.name, count: item.count }))
    : [];

  const timeOfDay = bot.time && typeof bot.time.timeOfDay === 'number' ? bot.time.timeOfDay : 0;

  return {
    status: 'connected',
    position: { x: Number(pos.x.toFixed(2)), y: Number(pos.y.toFixed(2)), z: Number(pos.z.toFixed(2)) },
    yaw: Number(bot.entity.yaw.toFixed(4)),
    pitch: Number(bot.entity.pitch.toFixed(4)),
    health: bot.health === undefined ? 20 : Number(bot.health),
    food: bot.food === undefined ? 20 : Number(bot.food),
    dimension: bot.game && bot.game.dimension ? bot.game.dimension : 'overworld',
    time_of_day: timeOfDay < 13000 ? 'day' : 'night',
    players,
    inventory,
  };
}

function startStateLoop() {
  stopStateLoop();
  stateTimer = setInterval(() => {
    try {
      send({ event: 'state', state: buildState() });
    } catch (err) {
      /* never let a state hiccup kill the bot */
    }
  }, 250);
  stateTimer.unref?.();
}

function stopStateLoop() {
  if (stateTimer) clearInterval(stateTimer);
  stateTimer = null;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function withTimeout(promise, ms, label) {
  return Promise.race([
    promise,
    sleep(ms).then(() => {
      throw new Error(`${label || 'operation'} timed out after ${ms}ms`);
    }),
  ]);
}

function stopPath() {
  if (bot && pathfinder && bot.pathfinder) {
    try {
      bot.pathfinder.setGoal(null);
    } catch (err) {
      /* ignore */
    }
  }
}

/* ------------------------------------------------------------------ commands */
async function cmdConnect(params) {
  if (bot) {
    try {
      bot.quit('reconnecting');
    } catch (err) {
      /* ignore */
    }
    bot = null;
  }
  const options = {
    host: String(params.host || '127.0.0.1'),
    port: Number(params.port || 25565),
    username: String(params.username || 'PyMC_Bot'),
    auth: params.auth === 'microsoft' ? 'microsoft' : 'offline', // cracked servers use offline
    viewDistance: params.view_distance || 'normal',
    hideErrors: true,
    checkTimeoutInterval: 60000,
  };
  if (params.version) options.version = String(params.version);

  log('info', `Connecting to ${options.host}:${options.port} as ${options.username} (version: ${params.version || 'auto'}).`);
  bot = mineflayer.createBot(options);

  bot.once('spawn', () => {
    if (pathfinder) {
      try {
        bot.loadPlugin(pathfinder.pathfinder);
      } catch (err) {
        log('warn', `Could not load pathfinder plugin: ${err.message}`);
      }
    }
    startStateLoop();
    send({ event: 'spawn' });
  });
  bot.on('end', (reason) => {
    stopStateLoop();
    send({ event: 'end', reason: reason || 'connection closed' });
  });
  bot.on('kicked', (reason) => {
    stopStateLoop();
    send({ event: 'kicked', reason: typeof reason === 'string' ? reason : JSON.stringify(reason) });
  });
  bot.on('error', (err) => {
    send({ event: 'error', message: err && err.message ? err.message : String(err) });
  });
  bot.on('chat', (username, message) => {
    if (username === bot.username) return;
    send({ event: 'chat', username, message });
  });
  bot.on('message', (jsonMsg) => {
    const text = jsonMsg.toString();
    if (text) log('debug', text);
  });

  // Load resource pack / login plugins asynchronously; the 'spawn' event above
  // is what tells Python the bot is actually in the world.
  return { connecting: true, connect_timeout_ms: CONNECT_TIMEOUT_MS };
}

function cmdSay(params) {
  if (!bot) throw new Error('not connected');
  bot.chat(String(params.text || ''));
  return { sent: true };
}

function cmdCommand(params) {
  if (!bot) throw new Error('not connected');
  const text = String(params.text || '');
  bot.chat(text.startsWith('/') ? text : `/${text}`);
  return { sent: true };
}

function cmdControl(params) {
  if (!bot) throw new Error('not connected');
  const name = String(params.name);
  const allowed = ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint'];
  if (!allowed.includes(name)) throw new Error(`unknown control: ${name}`);
  bot.setControlState(name, Boolean(params.state));
  return { control: name, state: Boolean(params.state) };
}

function cmdLook(params) {
  if (!bot) throw new Error('not connected');
  bot.look(Number(params.yaw) || 0, Number(params.pitch) || 0, Boolean(params.force));
  return { looked: true };
}

function cmdStopMotion() {
  if (!bot) return { stopped: false };
  for (const name of ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint']) {
    try {
      bot.setControlState(name, false);
    } catch (err) {
      /* ignore */
    }
  }
  stopPath();
  return { stopped: true };
}

function cmdPathfindTo(params) {
  if (!bot) throw new Error('not connected');
  if (!pathfinder || !bot.pathfinder) throw new Error('pathfinder_unavailable');
  const goal = new goals.GoalNear(Number(params.x), Number(params.y), Number(params.z), Number(params.range || 1));
  bot.pathfinder.setGoal(goal);
  return { goal: true };
}

function cmdStopPath() {
  stopPath();
  return { stopped: true };
}

async function cmdDig(params) {
  if (!bot) throw new Error('not connected');
  const blockName = String(params.block || 'stone').trim().toLowerCase();
  const budgetMs = Number(params.timeout || 20000);
  const deadline = Date.now() + budgetMs;

  const mcData = require('minecraft-data')(bot.version);
  const blockInfo = mcData.blocksByName[blockName];
  if (!blockInfo) throw new Error(`unknown block: ${blockName}`);

  let block = bot.findBlock({ matching: blockInfo.id, maxDistance: 48 });
  if (!block) {
    return { dug: false, reason: `no ${blockName} within 48 blocks` };
  }

  const distance = bot.entity.position.distanceTo(block.position);
  if (distance > 4.0) {
    if (pathfinder && bot.pathfinder) {
      try {
        bot.pathfinder.setGoal(new goals.GoalNear(block.position.x, block.position.y, block.position.z, 2));
        while (Date.now() < deadline) {
          await sleep(200);
          block = bot.findBlock({ matching: blockInfo.id, maxDistance: 48 });
          if (!block) break;
          if (bot.entity.position.distanceTo(block.position) <= 4.0) break;
        }
        stopPath();
      } catch (err) {
        log('warn', `pathfinding to ${blockName} failed: ${err.message}`);
      }
    }
  }

  block = bot.findBlock({ matching: blockInfo.id, maxDistance: 48 });
  if (!block) return { dug: false, reason: `lost sight of ${blockName}` };
  if (bot.entity.position.distanceTo(block.position) > 4.5) {
    return { dug: false, reason: `${blockName} is too far away` };
  }
  try {
    await withTimeout(bot.dig(block), Math.max(3000, deadline - Date.now()), `digging ${blockName}`);
    return { dug: true, block: blockName };
  } catch (err) {
    return { dug: false, reason: err.message };
  }
}

async function cmdEat(params) {
  if (!bot) throw new Error('not connected');
  const mcData = require('minecraft-data')(bot.version);
  const item = (bot.inventory ? bot.inventory.items() : []).find(
    (it) => mcData.foodsByName && mcData.foodsByName[it.name]
  );
  if (!item) return { ate: false, reason: 'no food in inventory' };
  await bot.equip(item, 'hand');
  await withTimeout(bot.consume(), 8000, 'eating');
  return { ate: true, item: item.name };
}

function cmdAttack(params) {
  if (!bot) throw new Error('not connected');
  const name = String(params.player);
  const player = bot.players[name];
  if (!player || !player.entity) throw new Error(`player ${name} is not visible`);
  bot.attack(player.entity);
  return { attacked: name };
}

function cmdJump() {
  if (!bot) throw new Error('not connected');
  bot.setControlState('jump', true);
  setTimeout(() => bot && bot.setControlState('jump', false), 250);
  return { jumped: true };
}

async function cmdDisconnect() {
  stopStateLoop();
  if (bot) {
    try {
      bot.quit('bye');
    } catch (err) {
      /* ignore */
    }
    bot = null;
  }
  return { disconnected: true };
}

const COMMANDS = {
  connect: cmdConnect,
  say: cmdSay,
  command: cmdCommand,
  control: cmdControl,
  look: cmdLook,
  stop_motion: cmdStopMotion,
  pathfind_to: cmdPathfindTo,
  stop_path: cmdStopPath,
  dig: cmdDig,
  eat: cmdEat,
  attack: cmdAttack,
  jump: cmdJump,
  disconnect: cmdDisconnect,
  ping: () => ({ pong: true, connected: Boolean(bot) }),
  state: () => buildState(),
};

/* --------------------------------------------------------------- main loop */
async function handle(request) {
  const id = request.id;
  const cmd = String(request.cmd || '');
  const params = request.params || {};
  const handler = COMMANDS[cmd];
  if (!handler) {
    send({ id, ok: false, error: `unknown command: ${cmd}` });
    return;
  }
  try {
    const result = await handler(params);
    send({ id, ok: true, result: result === undefined ? {} : result });
  } catch (err) {
    send({ id, ok: false, error: err && err.message ? err.message : String(err) });
  }
}

const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let request;
  try {
    request = JSON.parse(trimmed);
  } catch (err) {
    log('warn', `bridge received invalid JSON: ${trimmed.slice(0, 200)}`);
    return;
  }
  handle(request);
});
rl.on('close', () => {
  cmdDisconnect().finally(() => process.exit(0));
});

send({
  event: 'ready',
  caps: { pathfinder: Boolean(pathfinder), mineflayer: Boolean(mineflayer) },
  node: process.version,
});
