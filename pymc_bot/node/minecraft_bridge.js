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
 *                   msa_code (premium login: user code + verification URL),
 *                   version_fallback (server is newer than mineflayer supports),
 *                   server_asleep (a free host's sleeping/starting placeholder answered the ping)
 */

'use strict';

const readline = require('readline');

let mineflayer = null;
let minecraftData = null;     // the version database mineflayer reads (comes with mineflayer)
let minecraftProtocol = null; // used to ping the server for its version before joining
let mineflayerVersions = null; // { testedVersions, latestSupportedVersion, oldestSupportedVersion }
let pathfinder = null;   // optional, enables real A* pathfinding
const { probe } = require('./probe');
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
  minecraftData = require('minecraft-data');
  minecraftProtocol = require('minecraft-protocol');
  mineflayerVersions = require('mineflayer/lib/version');
} catch (err) {
  log('warn', `Could not load the Minecraft version tables (${err.message}); the server version is not checked before joining.`);
}
try {
  pathfinder = require('mineflayer-pathfinder');
  goals = pathfinder.goals;
} catch (err) {
  pathfinder = null;
  goals = null;
  log('warn', 'mineflayer-pathfinder not found; Python-side steering will be used.');
}

/* ------------------------------------------------------------ version check */
/** Supported Minecraft range of the installed mineflayer, e.g. { oldest: '1.8.8', latest: '26.1' }. */
function versionSupport() {
  if (!mineflayerVersions) return null;
  return {
    oldest: mineflayerVersions.oldestSupportedVersion,
    latest: mineflayerVersions.latestSupportedVersion,
    tested: mineflayerVersions.testedVersions,
  };
}

/** Whether minecraft-data has protocol data for `version` (what createBot needs to work). */
function hasData(version) {
  if (!minecraftData || !version) return false;
  try {
    const data = minecraftData(version);
    if (!data || !data.protocol) return false;
    const info = minecraftData.versionsByMinecraftVersion.pc[version];
    return !info || info.version === data.version.version;
  } catch (err) {
    return false;
  }
}

/**
 * Picks the version to join with. A version mineflayer knows is used as-is. When the server runs a
 * newer Minecraft than the installed mineflayer supports (e.g. 26.3 while mineflayer stops at 26.1),
 * the bot joins as the newest supported version instead of failing with "No data available for
 * version 26.3" - servers with ViaVersion + ViaBackwards accept that, others kick with a clear
 * "outdated client" message that the Python side explains.
 */
function resolveVersion(requested, ping) {
  if (requested && hasData(requested)) return { version: requested };
  const support = versionSupport();
  if (!support || !hasData(support.latest)) return requested ? { version: requested } : {};

  let serverName = requested || null;
  let serverProtocol = null;
  if (ping && ping.ok && ping.online) {
    serverName = String(ping.version || serverName || '');
    serverProtocol = Number(ping.protocol);
    if (!requested) {
      // Same lookup as minecraft-protocol's auto-version, but only versions that have data.
      const byProtocol = (minecraftData.postNettyVersionsByProtocolVersion.pc[serverProtocol] || [])
        .map((info) => info.minecraftVersion);
      // proxies advertise ranges like "Waterfall 1.8.x-1.21.x": prefer the newest number
      const byName = [serverName].concat((serverName.match(/\d+(?:\.\d+)+/g) || []).reverse());
      const usable = byProtocol.concat(byName).find(hasData);
      if (usable) return { version: usable, server: serverName, protocol: serverProtocol };
    }
  } else if (!requested) {
    return {}; // server unreachable: let mineflayer report the connection error
  }

  const latestProtocol = minecraftData.versionsByMinecraftVersion.pc[support.latest].version;
  const older = serverProtocol !== null && serverProtocol > 0 && serverProtocol < 47;
  const using = older ? support.oldest : support.latest;
  return {
    version: using,
    fallback: true,
    server: serverName || 'unknown',
    protocol: serverProtocol,
    newer: serverProtocol === null || serverProtocol > latestProtocol,
  };
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
  const inv = inventoryState();
  const groundItems = [];
  for (const id of Object.keys(tracked)) {
    const entity = tracked[id];
    if (!entity || !entity.position || entity.name !== 'item') continue;
    let dropped = null;
    try {
      dropped = typeof entity.getDroppedItem === 'function' ? entity.getDroppedItem() : null;
    } catch (err) {
      dropped = null;
    }
    groundItems.push({
      item: dropped && dropped.name ? `minecraft:${dropped.name}` : 'minecraft:unknown',
      count: dropped ? dropped.count : 1,
      dist: Number(pos.distanceTo(entity.position).toFixed(2)),
    });
  }
  groundItems.sort((a, b) => a.dist - b.dist);
  groundItems.length = Math.min(groundItems.length, 12);

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
    // everyone on the tab list (also players too far away to be tracked as entities)
    online_players: Object.keys(bot.players).filter((name) => name !== bot.username),
    inventory,
    held_item: heldItem,
    entities,
    ground_items: groundItems,
    // Slot layout as a player sees it: 0-8 hotbar, 9-35 backpack (pymc_bot.inventory).
    slots: inv.slots,
    selected_slot: inv.selected,
    armor_slots: inv.armor,
    armor: Object.values(inv.armor).filter(Boolean).map((stack) => stack.name),
    offhand: inv.offhand ? inv.offhand.name : '',
    sleeping: Boolean(bot.isSleeping),
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
  const requested = params.version ? String(params.version) : null;
  // Ping first (follows SRV records, e.g. name.aternos.me): a sleeping free-host server answers
  // with a placeholder, and joining it would only produce a confusing kick.
  const ping = minecraftProtocol ? await probe(options.host, options.port, 8000, minecraftProtocol) : null;
  if (ping && ping.ok && ping.asleep) {
    send({
      event: 'server_asleep',
      state: ping.state,
      motd: ping.motd,
      version_name: ping.version,
      resolved: ping.resolved,
    });
    throw new Error(`The server is ${ping.state}: ${ping.motd || ping.version}`);
  }
  if (ping && ping.ok && ping.resolved) {
    log('info', `${options.host} points to ${ping.resolved.host}:${ping.resolved.port} (SRV record).`);
  }
  const resolved = resolveVersion(requested, ping);
  if (resolved.version) options.version = resolved.version;
  if (resolved.fallback) {
    const support = versionSupport();
    send({
      event: 'version_fallback',
      server_version: resolved.server,
      server_protocol: resolved.protocol,
      using: resolved.version,
      latest: support && support.latest,
      oldest: support && support.oldest,
    });
    log('warn', `The server runs Minecraft ${resolved.server}, which the installed mineflayer does not support yet `
      + `(it goes up to ${support && support.latest}). Joining as ${resolved.version} instead - this works when `
      + 'the server has ViaVersion + ViaBackwards; otherwise the server will refuse the bot.');
  }
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
    + `(auth: ${options.auth}, version: ${options.version || 'auto'}${requested ? '' : ' (detected)'}).`
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
  const rawTimeout = Number(params.timeout || 20);
  const budgetMs = rawTimeout <= 1000 ? rawTimeout * 1000 : rawTimeout; // seconds (ms accepted too)
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
  const foods = mcData.foodsByName || {};
  let item = params.slot === undefined || params.slot === null ? null : slotItem(params.slot);
  if (item && !foods[item.name]) throw new Error(`${item.name} is not food`);
  if (!item) {
    const edible = (bot.inventory ? bot.inventory.items() : []).filter((it) => foods[it.name]);
    edible.sort((a, b) => (foods[b.name].foodPoints || 0) - (foods[a.name].foodPoints || 0));
    item = edible[0];
  }
  if (!item) throw new Error('no food in the inventory');
  if (bot.food >= 20 && !['golden_apple', 'enchanted_golden_apple', 'chorus_fruit'].includes(item.name)) {
    throw new Error('not hungry');
  }
  const before = bot.quickBarSlot;
  await bot.equip(item, 'hand');
  await withTimeout(bot.consume(), 8000, 'eating');
  if (params.restore !== false && before !== bot.quickBarSlot) bot.setQuickBarSlot(before);
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

/* -------------------------------------------------------- inventory & life */
// Python numbers the inventory like a player sees it: 0-8 hotbar, 9-35 backpack.
// mineflayer's window: 5-8 armour, 9-35 backpack, 36-44 hotbar, 45 off-hand.
const ARMOR_WINDOW = { head: 5, chest: 6, legs: 7, feet: 8 };
const DESTINATIONS = { hand: 'hand', offhand: 'off-hand', head: 'head', chest: 'torso', legs: 'legs', feet: 'feet' };

function windowSlot(index) {
  const i = Number(index);
  if (!Number.isInteger(i) || i < 0 || i > 35) throw new Error(`slot ${index} does not exist (0-35)`);
  return i < 9 ? 36 + i : i;
}

function stackOf(item) {
  return item ? { name: item.name, count: item.count } : null;
}

function inventoryState() {
  const slots = [];
  const window = bot && bot.inventory ? bot.inventory.slots : [];
  for (let i = 0; i < 36; i += 1) slots.push(stackOf(window[windowSlot(i)]));
  const armor = {};
  for (const [name, slot] of Object.entries(ARMOR_WINDOW)) armor[name] = stackOf(window[slot]);
  return { slots, armor, offhand: stackOf(window[45]), selected: bot ? bot.quickBarSlot || 0 : 0 };
}

function slotItem(index) {
  const item = bot.inventory.slots[windowSlot(index)];
  if (!item) throw new Error(`slot ${index} is empty`);
  return item;
}

function cmdSelectSlot(params) {
  if (!bot) throw new Error('not connected');
  const slot = Number(params.slot);
  if (!Number.isInteger(slot) || slot < 0 || slot > 8) throw new Error('hotbar slots are 0-8');
  bot.setQuickBarSlot(slot);
  return { selected: slot };
}

async function cmdMoveSlot(params) {
  if (!bot) throw new Error('not connected');
  const from = windowSlot(params.from);
  const to = windowSlot(params.to);
  if (!bot.inventory.slots[from]) throw new Error(`slot ${params.from} is empty`);
  await withTimeout(bot.moveSlotItem(from, to), 5000, 'moving the item');
  return { moved: true };
}

async function cmdEquip(params) {
  if (!bot) throw new Error('not connected');
  const destination = DESTINATIONS[String(params.destination || 'hand')];
  if (!destination) throw new Error('destination: hand, offhand, head, chest, legs or feet');
  const item = slotItem(params.slot);
  await withTimeout(bot.equip(item, destination), 5000, 'equipping');
  return { equipped: item.name, destination: params.destination || 'hand' };
}

async function cmdUnequip(params) {
  if (!bot) throw new Error('not connected');
  const destination = DESTINATIONS[String(params.destination || '')];
  if (!destination || destination === 'hand') throw new Error('unequip: offhand, head, chest, legs or feet');
  await withTimeout(bot.unequip(destination), 5000, 'unequipping');
  return { unequipped: params.destination };
}

async function cmdSwapHands() {
  if (!bot) throw new Error('not connected');
  await withTimeout(bot.moveSlotItem(36 + bot.quickBarSlot, 45), 5000, 'swapping hands');
  return { swapped: true };
}

async function cmdToss(params) {
  if (!bot) throw new Error('not connected');
  const item = slotItem(params.slot);
  const count = params.count === undefined || params.count === null ? item.count : Math.max(1, Number(params.count));
  if (count >= item.count) await withTimeout(bot.tossStack(item), 5000, 'dropping');
  else await withTimeout(bot.toss(item.type, item.metadata === undefined ? null : item.metadata, count), 5000, 'dropping');
  return { tossed: item.name, count: Math.min(count, item.count) };
}

async function cmdPlace(params) {
  if (!bot) throw new Error('not connected');
  const item = slotItem(params.slot);
  const Vec3 = bot.entity.position.constructor;
  await withTimeout(bot.equip(item, 'hand'), 5000, 'equipping');
  const base = bot.entity.position.floored();
  const yaw = bot.entity.yaw;
  const ahead = [Math.round(-Math.sin(yaw)), Math.round(-Math.cos(yaw))];
  const options = [ahead, [1, 0], [-1, 0], [0, 1], [0, -1]].filter(([dx, dz]) => dx || dz);
  for (const [dx, dz] of options) {
    const target = base.offset(dx, 0, dz);
    const here = bot.blockAt(target);
    const below = bot.blockAt(target.offset(0, -1, 0));
    if (!here || !below || here.boundingBox !== 'empty' || below.boundingBox !== 'block') continue;
    await bot.lookAt(target.offset(0.5, 0.5, 0.5));
    await withTimeout(bot.placeBlock(below, new Vec3(0, 1, 0)), 5000, 'placing');
    return { placed: item.name, x: target.x, y: target.y, z: target.z };
  }
  throw new Error('no free spot next to you to place a block');
}

async function cmdUseItem(params) {
  if (!bot) throw new Error('not connected');
  const seconds = Math.max(0.1, Math.min(5, Number(params.seconds || 0.3)));
  bot.activateItem(Boolean(params.offhand));
  await sleep(seconds * 1000);
  bot.deactivateItem();
  return { used: bot.heldItem ? bot.heldItem.name : 'hand', seconds };
}

async function walkNear(position, range, budgetMs) {
  if (bot.entity.position.distanceTo(position) <= range + 0.5) return true;
  if (!pathfinder || !bot.pathfinder) return false;
  bot.pathfinder.setGoal(new goals.GoalNear(position.x, position.y, position.z, range));
  const deadline = Date.now() + budgetMs;
  while (Date.now() < deadline && bot.entity.position.distanceTo(position) > range + 0.5) await sleep(200);
  stopPath();
  return bot.entity.position.distanceTo(position) <= range + 0.5;
}

async function cmdSleep() {
  if (!bot) throw new Error('not connected');
  if (bot.isSleeping) return { sleeping: true };
  const bed = bot.findBlock({ matching: (block) => bot.isABed(block), maxDistance: 32 });
  if (!bed) throw new Error('no bed within 32 blocks');
  if (!(await walkNear(bed.position, 2, 15000))) throw new Error('could not reach the bed');
  await withTimeout(bot.sleep(bed), 5000, 'getting into bed');
  return { sleeping: true };
}

async function cmdWake() {
  if (!bot) throw new Error('not connected');
  if (bot.isSleeping) await withTimeout(bot.wake(), 5000, 'getting up');
  return { sleeping: false };
}

async function cmdCollect(params) {
  if (!bot) throw new Error('not connected');
  const radius = Math.max(1, Math.min(32, Number(params.radius || 12)));
  const deadline = Date.now() + Math.max(2000, Number(params.seconds || 15) * 1000);
  let picked = 0;
  while (Date.now() < deadline) {
    const drops = Object.values(bot.entities).filter(
      (e) => e && e.name === 'item' && e.position && bot.entity.position.distanceTo(e.position) <= radius
    );
    if (!drops.length) break;
    drops.sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position));
    const target = drops[0];
    if (!pathfinder || !bot.pathfinder) throw new Error('pathfinder_unavailable');
    bot.pathfinder.setGoal(new goals.GoalNear(target.position.x, target.position.y, target.position.z, 0));
    const until = Math.min(deadline, Date.now() + 6000);
    while (Date.now() < until && bot.entities[target.id]) await sleep(150);
    if (!bot.entities[target.id]) picked += 1;
    else break;
  }
  stopPath();
  if (!picked) throw new Error(`nothing to pick up within ${radius} blocks`);
  return { picked };
}

async function cmdAttackMob(params) {
  if (!bot) throw new Error('not connected');
  const wanted = params.mob ? String(params.mob).toLowerCase().replace(/^minecraft:/, '') : '';
  const range = Math.max(4, Math.min(24, Number(params.range || 16)));
  const candidates = Object.values(bot.entities).filter((e) => {
    if (!e || e === bot.entity || !e.position || e.type === 'player' || e.username || e.name === 'item') return false;
    if (bot.entity.position.distanceTo(e.position) > range) return false;
    const name = String(e.name || '').toLowerCase();
    return wanted ? name === wanted : HOSTILE_ENTITIES.has(name);
  });
  if (!candidates.length) throw new Error(wanted ? `no ${wanted} within ${range} blocks` : `no hostile mob within ${range} blocks`);
  candidates.sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position));
  const target = candidates[0];
  const deadline = Date.now() + Math.max(2000, Math.min(30000, Number(params.seconds || 15) * 1000));
  let hits = 0;
  while (Date.now() < deadline && bot.entities[target.id] && target.isValid !== false) {
    const distance = bot.entity.position.distanceTo(target.position);
    if (distance > 3 && pathfinder && bot.pathfinder) {
      bot.pathfinder.setGoal(new goals.GoalFollow(target, 2), true);
    }
    await bot.lookAt(target.position.offset(0, (target.height || 1.6) * 0.85, 0));
    if (distance <= 3.2) {
      bot.attack(target);
      hits += 1;
    }
    await sleep(600 + Math.random() * 150); // attack cooldown, like a player timing the swing
  }
  stopPath();
  const killed = !bot.entities[target.id] || target.isValid === false;
  return { target: target.name, hits, killed };
}

/* ------------------------------------------------------------------ vision */
/**
 * The bot's first-person view as a JPEG, for vision models (Minecraftism VL, any Ollama/Cortex VL
 * model). Optional: rendering headlessly needs three extra packages that are NOT installed with the
 * bridge because they compile native WebGL code:
 *   npm install --prefix pymc_bot/node prismarine-viewer node-canvas-webgl three@0.128.0
 * Without them `snapshot` answers with that hint and the Python side simply plays without pictures.
 */
const VISION_PACKAGES = 'npm install --prefix pymc_bot/node prismarine-viewer node-canvas-webgl three@0.128.0';
let eyes = null; // { bot, viewer, worldView, renderer, canvas, width, height }

function openEyes(width, height, viewDistance) {
  let viewerLib;
  let createCanvas;
  let THREE;
  try {
    viewerLib = require('prismarine-viewer').viewer;
    ({ createCanvas } = require('node-canvas-webgl/lib'));
    THREE = require('three');
  } catch (err) {
    throw new Error(`vision is not installed (${err.message.split('\n')[0]}). Run: ${VISION_PACKAGES}`);
  }
  if (typeof global.Worker === 'undefined') global.Worker = require('worker_threads').Worker;
  const canvas = createCanvas(width, height);
  const renderer = new THREE.WebGLRenderer({ canvas });
  const viewer = new viewerLib.Viewer(renderer);
  if (viewer.setVersion(bot.version) === false) {
    throw new Error(`prismarine-viewer cannot render Minecraft ${bot.version}`);
  }
  const center = bot.entity.position;
  const worldView = new viewerLib.WorldView(bot.world, viewDistance, center);
  viewer.listen(worldView);
  worldView.init(center);
  return { bot, viewer, worldView, renderer, canvas, width, height };
}

async function cmdSnapshot(params) {
  if (!bot || !bot.entity) throw new Error('not connected');
  const width = Math.max(64, Math.min(1024, Number(params.width || 448)));
  const height = Math.max(64, Math.min(1024, Number(params.height || Math.round(width * 9 / 16))));
  if (!eyes || eyes.bot !== bot || eyes.width !== width || eyes.height !== height) {
    eyes = openEyes(width, height, Math.max(2, Math.min(8, Number(params.view_distance || 4))));
  }
  const eye = bot.entity.position.offset(0, bot.entity.height || 1.62, 0);
  eyes.worldView.updatePosition(bot.entity.position);
  eyes.viewer.setFirstPersonCamera(eye, bot.entity.yaw, bot.entity.pitch);
  // chunks are meshed in worker threads; give them a moment (the first picture can be sky only)
  const settle = Math.max(0, Math.min(5000, Number(params.settle_ms ?? 600)));
  if (eyes.viewer.world && typeof eyes.viewer.world.waitForChunksToRender === 'function') {
    await Promise.race([eyes.viewer.world.waitForChunksToRender(), sleep(settle)]);
  } else {
    await sleep(settle);
  }
  if (typeof eyes.viewer.update === 'function') eyes.viewer.update();
  eyes.renderer.render(eyes.viewer.scene, eyes.viewer.camera);
  const quality = Math.max(0.3, Math.min(0.95, Number(params.quality || 0.8)));
  const buffer = eyes.canvas.toBuffer('image/jpeg', { quality });
  return {
    mime: 'image/jpeg',
    width,
    height,
    image: buffer.toString('base64'),
    yaw: bot.entity.yaw,
    pitch: bot.entity.pitch,
  };
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
  select_slot: cmdSelectSlot,
  move_slot: cmdMoveSlot,
  equip: cmdEquip,
  unequip: cmdUnequip,
  swap_hands: cmdSwapHands,
  toss: cmdToss,
  place: cmdPlace,
  use_item: cmdUseItem,
  sleep: cmdSleep,
  wake: cmdWake,
  collect: cmdCollect,
  attack_mob: cmdAttackMob,
  snapshot: cmdSnapshot,
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
    minecraft: versionSupport(),
  },
  node: process.version,
});
