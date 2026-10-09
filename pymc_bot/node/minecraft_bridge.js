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
 *           events: ready, log, state, chat, spawn, end, kicked, error, exit,
 *                   msa_code (premium login: user code + verification URL)
 */

'use strict';

const readline = require('readline');

let mineflayer = null;
let pathfinder = null;   // optional, enables real A* pathfinding
let goals = null;
let bot = null;
let stateTimer = null;

const CONNECT_TIMEOUT_MS = 30000;
/** How many entities the bridge reports to the trained brain (matches the trainer default). */
const ADVANCED_ENTITY_LIMIT = 24;
/** Mobs the trained brain should treat as dangerous (a cheap fallback for mob hostility). */
const HOSTILE_ENTITIES = new Set([
  'zombie', 'husk', 'drowned', 'skeleton', 'stray', 'wither_skeleton', 'creeper', 'spider',
  'cave_spider', 'enderman', 'witch', 'slime', 'magma_cube', 'pillager', 'vindicator',
  'evoker', 'ravager', 'warden', 'blaze', 'ghast', 'hoglin', 'zoglin', 'piglin',
  'piglin_brute', 'phantom', 'silverfish', 'endermite', 'guardian', 'elder_guardian',
  'shulker', 'vex', 'wither', 'ender_dragon',
]);

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

  // Entities and the held item: the trained brain's "advanced" inputs. mineflayer knows
  // every entity the client tracks, so the bot can learn to react to what is around it.
  const entities = [];
  const tracked = bot.entities || {};
  for (const id of Object.keys(tracked)) {
    const entity = tracked[id];
    if (!entity || !entity.position || entity === bot.entity) continue;
    const ex = entity.position.x - pos.x;
    const ey = entity.position.y - pos.y;
    const ez = entity.position.z - pos.z;
    const isPlayer = entity.type === 'player' || Boolean(entity.username);
    const kind = isPlayer ? 'player' : String(entity.name || entity.displayName || entity.type || 'unknown');
    entities.push({
      type: (isPlayer || kind.includes(':')) ? (isPlayer ? 'minecraft:player' : kind) : `minecraft:${kind}`,
      dx: Number(ex.toFixed(2)),
      dy: Number(ey.toFixed(2)),
      dz: Number(ez.toFixed(2)),
      dist: Number(Math.sqrt(ex * ex + ey * ey + ez * ez).toFixed(2)),
      hostile: !isPlayer && HOSTILE_ENTITIES.has(kind),
      player: isPlayer,
      health: typeof entity.health === 'number' ? Number(entity.health.toFixed(1)) : 0,
      on_ground: Boolean(entity.onGround),
      yaw: entity.yaw === undefined ? 0 : Number((entity.yaw * 180 / Math.PI).toFixed(1)),
      count: entity.itemStack ? entity.itemStack.count : 0,
      held_item: entity.heldItem && entity.heldItem.name ? `minecraft:${entity.heldItem.name}` : '',
    });
  }
  entities.sort((a, b) => a.dist - b.dist);
  entities.length = Math.min(entities.length, ADVANCED_ENTITY_LIMIT);
  const heldItem = bot.heldItem && bot.heldItem.name ? `minecraft:${bot.heldItem.name}` : '';

  const timeOfDay = bot.time && typeof bot.time.timeOfDay === 'number' ? bot.time.timeOfDay : 0;

  return {
    status: 'connected',
    username: bot.username,
    position: { x: Number(pos.x.toFixed(2)), y: Number(pos.y.toFixed(2)), z: Number(pos.z.toFixed(2)) },
    yaw: Number(bot.entity.yaw.toFixed(4)),
    pitch: Number(bot.entity.pitch.toFixed(4)),
    health: bot.health === undefined ? 20 : Number(bot.health),
    food: bot.food === undefined ? 20 : Number(bot.food),
    dimension: bot.game && bot.game.dimension ? bot.game.dimension : 'overworld',
    time_of_day: timeOfDay < 13000 ? 'day' : 'night',
    players,
    inventory,
    held_item: heldItem,
    entities,
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
  const premium = params.auth === 'microsoft';
  const options = {
    host: String(params.host || '127.0.0.1'),
    port: Number(params.port || 25565),
    // offline: the in-game name. microsoft: the account email (the real name
    // is decided by the account and reported back in the state events).
    username: String(params.username || 'PyMC_Bot'),
    auth: premium ? 'microsoft' : 'offline', // cracked servers use offline
    viewDistance: params.view_distance || 'normal',
    hideErrors: true,
    checkTimeoutInterval: premium ? 120000 : 60000,
  };
  if (params.version) options.version = String(params.version);
  if (premium) {
    // Cache refresh tokens per account so only the first join needs a device code.
    options.profilesFolder = params.profiles_folder || '.pymc_profiles';
    options.onMsaCode = (data) => {
      send({
        event: 'msa_code',
        user_code: data && data.user_code,
        verification_uri: (data && (data.verification_uri || data.verificationUri)) || 'https://www.microsoft.com/link',
        expires_in: (data && data.expires_in) || null,
        message: (data && data.message) || 'Sign in to Microsoft to finish the premium login.',
      });
    };
  }

  log('info', `Connecting to ${options.host}:${options.port} as ${options.username} `
    + `(auth: ${options.auth}, version: ${params.version || 'auto'}).`
    + (premium ? ' First premium login shows a device code - watch the event stream.' : ''));
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
  bot.once('login', () => {
    log('success', `Logged in${premium ? ' (Microsoft)' : ''} as ${bot.username}.`);
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

function cmdSwingArm() {
  if (!bot) throw new Error('not connected');
  bot.swingArm('right');
  return { swung: true };
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
  swing_arm: cmdSwingArm,
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
  caps: {
    pathfinder: Boolean(pathfinder),
    mineflayer: Boolean(mineflayer),
    auth: ['offline', 'microsoft'],
    multiple_bots: 'one process per bot - spawn several to populate a server',
  },
  node: process.version,
});
