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
  send({ event: 'error', message: 'mineflayer is not installed (run `python -m pymc_bot setup` - no npm needed). ' + err.message });
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
  // what the autopilot reacts to: light at the feet, air, fire, water, rain, death
  let feet = null;
  try {
    feet = bot.blockAt(pos);
  } catch (err) {
    feet = null;
  }
  const meta = bot.entity.metadata ? bot.entity.metadata[0] : 0;

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
    world_time: Math.round(timeOfDay),
    block_light: feet && typeof feet.light === 'number' ? feet.light : null,
    sky_light: feet && typeof feet.skyLight === 'number' ? feet.skyLight : null,
    oxygen: typeof bot.oxygenLevel === 'number' ? bot.oxygenLevel : 20,
    in_water: Boolean(bot.entity.isInWater),
    on_fire: typeof meta === 'number' && (meta & 0x01) === 0x01,
    raining: Boolean(bot.isRaining),
    dead: typeof bot.health === 'number' && bot.health <= 0,
    riding: bot.vehicle ? String(bot.vehicle.name || 'vehicle') : '',
    xp_level: bot.experience ? bot.experience.level : 0,
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

/* ------------------------------------------------------------------ skills */
// One command per skill of pymc_bot/skills.py: bigger jobs a player does (chop a tree, craft,
// farm, store things in a chest...). Each one answers with a short `message` the Python side
// shows in the log, plus numbers (mined, placed_count, kills) for the stats.
function registry() {
  return bot.registry || require('minecraft-data')(bot.version);
}

function need() {
  if (!bot || !bot.entity) throw new Error('not connected');
}

function cleanName(value) {
  return String(value || '').toLowerCase().trim().replace(/^minecraft:/, '').replace(/\s+/g, '_');
}

function Vec3() {
  return bot.entity.position.constructor;
}

function invItems() {
  return bot.inventory ? bot.inventory.items() : [];
}

/** First inventory item whose name matches (exact first, then a part like "pickaxe" or "_log"). */
function findItem(match) {
  const items = invItems();
  if (typeof match === 'function') return items.find(match) || null;
  const name = cleanName(match);
  if (!name) return null;
  return items.find((it) => it.name === name) || items.find((it) => it.name.includes(name)) || null;
}

function countItem(name) {
  return invItems().filter((it) => it.name === name).reduce((sum, it) => sum + it.count, 0);
}

async function holdItem(item, destination) {
  if (!item) return false;
  await withTimeout(bot.equip(item, destination || 'hand'), 5000, 'equipping');
  return true;
}

function blockIds(test) {
  return Object.values(registry().blocksByName).filter((block) => test(block.name)).map((block) => block.id);
}

function nearestBlocks(test, radius, count) {
  const ids = blockIds(test);
  if (!ids.length) return [];
  return bot.findBlocks({ matching: ids, maxDistance: radius, count: count || 64 })
    .map((pos) => bot.blockAt(pos))
    .filter(Boolean)
    .sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position));
}

/** Walk until the block can be reached (looked at) - pathfinder when available. */
async function reachBlock(pos, budgetMs) {
  const eye = bot.entity.position.offset(0, bot.entity.height || 1.62, 0);
  if (eye.distanceTo(pos.offset(0.5, 0.5, 0.5)) <= 4.2) return true;
  if (!pathfinder || !bot.pathfinder) return false;
  try {
    await withTimeout(bot.pathfinder.goto(new goals.GoalLookAtBlock(pos, bot.world, { reach: 4 })), budgetMs || 20000, 'walking');
  } catch (err) {
    stopPath();
    const near = bot.entity.position.offset(0, bot.entity.height || 1.62, 0).distanceTo(pos.offset(0.5, 0.5, 0.5));
    return near <= 4.5;
  }
  return true;
}

async function reachEntity(entity, range, budgetMs) {
  if (bot.entity.position.distanceTo(entity.position) <= range) return true;
  if (!pathfinder || !bot.pathfinder) return false;
  try {
    await withTimeout(bot.pathfinder.goto(new goals.GoalFollow(entity, Math.max(1, range - 1))), budgetMs || 15000, 'walking');
  } catch (err) {
    stopPath();
  }
  return bot.entities[entity.id] !== undefined && bot.entity.position.distanceTo(entity.position) <= range + 0.5;
}

/** The fastest tool in the inventory for a block (keeps the hand when nothing helps). */
async function equipBestTool(block) {
  let best = null;
  let bestTime = Infinity;
  const hand = () => { try { return block.digTime(null, false, false, false, [], []); } catch (err) { return Infinity; } };
  const handTime = hand();
  for (const item of invItems()) {
    if (block.harvestTools && !block.harvestTools[item.type]) continue;
    let time = Infinity;
    try {
      time = block.digTime(item.type, false, false, false, [], []);
    } catch (err) {
      continue;
    }
    if (time < bestTime) {
      best = item;
      bestTime = time;
    }
  }
  if (best && (bestTime < handTime || block.harvestTools)) await holdItem(best);
  return best;
}

async function digBlock(block) {
  if (!(await reachBlock(block.position, 20000))) throw new Error(`cannot reach the ${block.name}`);
  const fresh = bot.blockAt(block.position);
  if (!fresh || fresh.name !== block.name) return false;
  await equipBestTool(fresh);
  if (fresh.harvestTools && !(bot.heldItem && fresh.harvestTools[bot.heldItem.type])) {
    throw new Error(`${fresh.name} needs a better tool`);
  }
  if (!bot.canDigBlock(fresh)) throw new Error(`cannot dig the ${fresh.name} from here`);
  await withTimeout(bot.dig(fresh, true), 30000, `digging ${fresh.name}`);
  return true;
}

/** Walk over dropped items nearby (never throws). */
async function pickUpNear(radius, budgetMs) {
  if (!pathfinder || !bot.pathfinder) return 0;
  const deadline = Date.now() + budgetMs;
  let picked = 0;
  while (Date.now() < deadline) {
    const drops = Object.values(bot.entities).filter(
      (e) => e && e.name === 'item' && e.position && bot.entity.position.distanceTo(e.position) <= radius
    );
    if (!drops.length) break;
    drops.sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position));
    const target = drops[0];
    bot.pathfinder.setGoal(new goals.GoalNear(target.position.x, target.position.y, target.position.z, 0));
    const until = Math.min(deadline, Date.now() + 5000);
    while (Date.now() < until && bot.entities[target.id]) await sleep(150);
    if (bot.entities[target.id]) break;
    picked += 1;
  }
  stopPath();
  return picked;
}

/** Mine up to `count` blocks matching `test`; shared by chop_tree, mine_ore and gather. */
async function mineMatching(test, count, radius, budgetMs, label) {
  const deadline = Date.now() + budgetMs;
  const skipped = new Set();
  let mined = 0;
  let lastError = null;
  while (mined < count && Date.now() < deadline) {
    const blocks = nearestBlocks(test, radius, 32).filter((b) => !skipped.has(b.position.toString()));
    if (!blocks.length) break;
    const block = blocks[0];
    try {
      if (await digBlock(block)) mined += 1;
    } catch (err) {
      lastError = err.message;
      skipped.add(block.position.toString());
      if (skipped.size > 6) break;
    }
  }
  if (mined) await pickUpNear(8, 8000);
  if (!mined) throw new Error(lastError || `no ${label} within ${radius} blocks`);
  return mined;
}

/** Place `item` so it fills `pos` (needs a solid neighbour to click on). */
async function placeAt(pos, item) {
  const V = Vec3();
  const here = bot.blockAt(pos);
  if (!here || (here.boundingBox !== 'empty' && !['water', 'lava', 'short_grass', 'grass', 'tall_grass', 'snow'].includes(here.name))) return false;
  const faces = [[0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
  for (const [dx, dy, dz] of faces) {
    const ref = bot.blockAt(pos.offset(dx, dy, dz));
    if (!ref || ref.boundingBox !== 'block') continue;
    await holdItem(item);
    try {
      await withTimeout(bot.placeBlock(ref, new V(-dx, -dy, -dz)), 4000, 'placing');
      return true;
    } catch (err) {
      const now = bot.blockAt(pos);
      if (now && now.boundingBox === 'block') return true; // placed, the server was just slow to confirm
    }
  }
  return false;
}

function buildingItem(wanted) {
  if (wanted) {
    const item = findItem(wanted);
    if (item) return item;
  }
  const plain = ['cobblestone', 'cobbled_deepslate', 'dirt', 'stone', 'netherrack', 'deepslate', 'andesite', 'diorite',
    'granite', 'tuff', 'sandstone', 'blackstone'];
  const items = invItems();
  const pile = (name) => countItem(name);
  const candidates = items.filter((it) => plain.includes(it.name) || /_planks$|_bricks$|_wool$/.test(it.name));
  candidates.sort((a, b) => pile(b.name) - pile(a.name));
  return candidates[0] || null;
}

function entitiesNear(test, radius) {
  return Object.values(bot.entities)
    .filter((e) => e && e !== bot.entity && e.position && bot.entity.position.distanceTo(e.position) <= radius && test(e))
    .sort((a, b) => bot.entity.position.distanceTo(a.position) - bot.entity.position.distanceTo(b.position));
}

function mobName(entity) {
  return String(entity.name || entity.displayName || '').toLowerCase();
}

/* --- survival */
async function cmdDrink(params) {
  need();
  const wanted = cleanName(params.item);
  const drinkable = (it) => (wanted ? it.name === wanted || it.name.includes(wanted)
    : ['potion', 'milk_bucket', 'honey_bottle'].includes(it.name));
  const item = findItem(drinkable);
  if (!item) throw new Error('nothing to drink');
  await holdItem(item);
  await withTimeout(bot.consume(), 8000, 'drinking');
  return { drank: item.name, message: `drank ${item.name.replace(/_/g, ' ')}` };
}

async function cmdSurface() {
  need();
  const deadline = Date.now() + 10000;
  await bot.look(bot.entity.yaw, Math.PI / 2, true).catch(() => {});
  bot.setControlState('jump', true);
  bot.setControlState('forward', true);
  try {
    while (Date.now() < deadline && (bot.entity.isInWater || (bot.oxygenLevel ?? 20) < 20)) {
      const head = bot.blockAt(bot.entity.position.offset(0, 1.6, 0));
      if (head && head.name !== 'water' && (bot.oxygenLevel ?? 20) >= 18) break;
      await sleep(150);
    }
  } finally {
    bot.setControlState('jump', false);
    bot.setControlState('forward', false);
  }
  return { oxygen: bot.oxygenLevel, message: 'swam up for air' };
}

function onFire() {
  const flags = bot.entity && bot.entity.metadata ? bot.entity.metadata[0] : 0;
  return typeof flags === 'number' && (flags & 0x01) === 0x01;
}

async function cmdExtinguish() {
  need();
  if (!onFire() && !bot.entity.isInLava) throw new Error('not on fire');
  const bucket = findItem('water_bucket');
  if (bucket) {
    await holdItem(bucket);
    const below = bot.blockAt(bot.entity.position.offset(0, -1, 0));
    if (below) await bot.lookAt(below.position.offset(0.5, 1, 0.5), true);
    bot.activateItem();
    await sleep(400);
    bot.deactivateItem();
    await sleep(600);
    bot.activateItem(); // scoop the water back up, like players do
    await sleep(250);
    bot.deactivateItem();
    return { message: 'put the fire out with a water bucket' };
  }
  const water = nearestBlocks((name) => name === 'water', 16, 8)[0];
  if (!water) throw new Error('no water bucket and no water nearby');
  await walkNear(water.position, 0, 8000);
  return { message: 'jumped into water to put the fire out' };
}

async function cmdRespawn() {
  if (!bot) throw new Error('not connected');
  if (bot.health > 0 && bot.entity) throw new Error('not dead');
  bot.respawn();
  return { message: 'respawned' };
}

/* --- gathering */
function matchBlockName(wanted) {
  const name = cleanName(wanted);
  if (registry().blocksByName[name]) return (n) => n === name;
  return (n) => n.includes(name);
}

async function cmdFindBlock(params) {
  need();
  const wanted = cleanName(params.block || 'oak_log');
  const radius = Math.max(4, Math.min(128, Number(params.radius || 64)));
  const found = nearestBlocks(matchBlockName(wanted), radius, 5).slice(0, 5).map((b) => ({
    block: b.name, x: b.position.x, y: b.position.y, z: b.position.z,
    distance: Number(bot.entity.position.distanceTo(b.position).toFixed(1)),
  }));
  if (!found.length) throw new Error(`no ${wanted} within ${radius} blocks`);
  const first = found[0];
  return { found, message: `${first.block} at ${first.x} ${first.y} ${first.z} (${Math.round(first.distance)} blocks away)` };
}

async function cmdChopTree(params) {
  need();
  const count = Math.max(1, Math.min(64, Number(params.count || 16)));
  const radius = Math.max(4, Math.min(64, Number(params.radius || 32)));
  const isLog = (n) => /_log$|_stem$/.test(n) && !n.startsWith('stripped_');
  const first = nearestBlocks(isLog, radius, 16)[0];
  if (!first) throw new Error(`no tree within ${radius} blocks`);
  // the tree = logs connected to the first one (up, sideways, diagonally), lowest first
  const tree = [];
  const seen = new Set([first.position.toString()]);
  const queue = [first.position];
  while (queue.length && tree.length < 40) {
    const pos = queue.shift();
    tree.push(pos);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = 0; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const next = pos.offset(dx, dy, dz);
          const key = next.toString();
          if (seen.has(key)) continue;
          seen.add(key);
          const block = bot.blockAt(next);
          if (block && isLog(block.name)) queue.push(next);
        }
      }
    }
  }
  tree.sort((a, b) => a.y - b.y);
  let mined = 0;
  for (const pos of tree) {
    if (mined >= count) break;
    const block = bot.blockAt(pos);
    if (!block || !isLog(block.name)) continue;
    try {
      if (await digBlock(block)) mined += 1;
    } catch (err) {
      /* too high to reach - leave the top of the tree */
    }
  }
  if (!mined) throw new Error('could not reach the tree');
  await pickUpNear(10, 10000);
  return { mined, message: `chopped a tree: ${mined} log(s)` };
}

const ORES = ['coal', 'iron', 'copper', 'gold', 'redstone', 'lapis', 'diamond', 'emerald', 'quartz', 'ancient_debris'];

async function cmdMineOre(params) {
  need();
  const ore = cleanName(params.ore).replace(/_ore$/, '').replace(/^deepslate_/, '');
  const count = Math.max(1, Math.min(32, Number(params.count || 4)));
  const radius = Math.max(4, Math.min(48, Number(params.radius || 24)));
  const test = ore
    ? (n) => n === `${ore}_ore` || n === `deepslate_${ore}_ore` || n === `nether_${ore}_ore` || (ore === 'ancient_debris' && n === ore)
    : (n) => n.endsWith('_ore') || n === 'ancient_debris';
  if (ore && !ORES.includes(ore)) throw new Error(`unknown ore ${ore} (${ORES.join(', ')})`);
  const mined = await mineMatching(test, count, radius, 140000, ore ? `${ore} ore` : 'ore');
  return { mined, message: `mined ${mined} ${ore || 'ore'} block(s)` };
}

async function cmdGather(params) {
  need();
  const wanted = cleanName(params.block || 'dirt');
  const count = Math.max(1, Math.min(64, Number(params.count || 8)));
  const radius = Math.max(4, Math.min(48, Number(params.radius || 24)));
  const mined = await mineMatching(matchBlockName(wanted), count, radius, 140000, wanted);
  return { mined, message: `mined ${mined} ${wanted.replace(/_/g, ' ')}` };
}

/* --- farming */
const CROPS = { wheat: [7, 'wheat_seeds'], carrots: [7, 'carrot'], potatoes: [7, 'potato'], beetroots: [3, 'beetroot_seeds'],
  nether_wart: [3, 'nether_wart'] };
const SEEDS = { wheat_seeds: 'wheat', carrot: 'carrots', potato: 'potatoes', beetroot_seeds: 'beetroots',
  melon_seeds: 'melon_stem', pumpkin_seeds: 'pumpkin_stem', nether_wart: 'nether_wart' };

function cropAge(block) {
  const props = typeof block.getProperties === 'function' ? block.getProperties() : {};
  return Number(props.age ?? block.metadata ?? 0);
}

async function plantOn(soil, seedName) {
  const seed = findItem(seedName);
  if (!seed) return false;
  const V = Vec3();
  await holdItem(seed);
  try {
    await withTimeout(bot.placeBlock(soil, new V(0, 1, 0)), 3000, 'planting');
  } catch (err) {
    const above = bot.blockAt(soil.position.offset(0, 1, 0));
    if (!above || above.boundingBox !== 'empty' || above.name === 'air') return Boolean(above && above.name !== 'air');
  }
  return true;
}

async function cmdHarvest(params) {
  need();
  const radius = Math.max(2, Math.min(32, Number(params.radius || 12)));
  const replant = params.replant !== false;
  const ripe = nearestBlocks((n) => n in CROPS || n === 'melon' || n === 'pumpkin', radius, 64)
    .filter((b) => !(b.name in CROPS) || cropAge(b) >= CROPS[b.name][0]);
  if (!ripe.length) throw new Error(`no ripe crops within ${radius} blocks`);
  let harvested = 0;
  let replanted = 0;
  const deadline = Date.now() + 100000;
  for (const crop of ripe.slice(0, 48)) {
    if (Date.now() > deadline) break;
    try {
      if (!(await digBlock(crop))) continue;
      harvested += 1;
      if (replant && crop.name in CROPS) {
        const soil = bot.blockAt(crop.position.offset(0, -1, 0));
        if (soil && (soil.name === 'farmland' || soil.name === 'soul_sand') && (await plantOn(soil, CROPS[crop.name][1]))) replanted += 1;
      }
    } catch (err) {
      /* next crop */
    }
  }
  await pickUpNear(radius, 12000);
  if (!harvested) throw new Error('could not reach the crops');
  return { harvested, mined: harvested, placed_count: replanted, message: `harvested ${harvested} crop(s)` + (replanted ? `, replanted ${replanted}` : '') };
}

async function cmdPlant(params) {
  need();
  const radius = Math.max(2, Math.min(24, Number(params.radius || 10)));
  const count = Math.max(1, Math.min(64, Number(params.count || 16)));
  const wanted = cleanName(params.seed);
  const seed = wanted ? findItem(wanted) : findItem((it) => it.name in SEEDS);
  if (!seed) throw new Error('no seeds to plant');
  const soilName = seed.name === 'nether_wart' ? 'soul_sand' : 'farmland';
  const spots = nearestBlocks((n) => n === soilName, radius, 128).filter((b) => {
    const above = bot.blockAt(b.position.offset(0, 1, 0));
    return above && above.name === 'air';
  });
  if (!spots.length) throw new Error(`no empty ${soilName} within ${radius} blocks`);
  let planted = 0;
  for (const soil of spots) {
    if (planted >= count || !findItem(seed.name)) break;
    if (!(await reachBlock(soil.position, 10000))) continue;
    if (await plantOn(soil, seed.name)) planted += 1;
  }
  if (!planted) throw new Error('could not plant anything');
  return { planted, placed_count: planted, message: `planted ${planted} ${seed.name.replace(/_/g, ' ')}` };
}

async function cmdTill(params) {
  need();
  const hoe = findItem((it) => it.name.endsWith('_hoe'));
  if (!hoe) throw new Error('I need a hoe');
  const radius = Math.max(2, Math.min(16, Number(params.radius || 6)));
  const count = Math.max(1, Math.min(32, Number(params.count || 8)));
  const spots = nearestBlocks((n) => ['dirt', 'grass_block', 'dirt_path', 'coarse_dirt', 'rooted_dirt'].includes(n), radius, 64)
    .filter((b) => {
      const above = bot.blockAt(b.position.offset(0, 1, 0));
      return above && above.name === 'air';
    });
  if (!spots.length) throw new Error(`no dirt to till within ${radius} blocks`);
  let tilled = 0;
  for (const block of spots) {
    if (tilled >= count) break;
    if (!(await reachBlock(block.position, 10000))) continue;
    await holdItem(findItem((it) => it.name.endsWith('_hoe')));
    await bot.activateBlock(block).catch(() => {});
    await sleep(200);
    const now = bot.blockAt(block.position);
    if (now && now.name === 'farmland') tilled += 1;
  }
  if (!tilled) throw new Error('could not till the ground');
  return { tilled, message: `tilled ${tilled} block(s) of farmland` };
}

async function cmdBonemeal(params) {
  need();
  if (!findItem('bone_meal')) throw new Error('I need bone meal');
  const radius = Math.max(2, Math.min(16, Number(params.radius || 8)));
  const count = Math.max(1, Math.min(32, Number(params.count || 4)));
  const targets = nearestBlocks((n) => n in CROPS || n.endsWith('_sapling') || n.endsWith('_stem') || n === 'sweet_berry_bush', radius, 32)
    .filter((b) => !(b.name in CROPS) || cropAge(b) < CROPS[b.name][0]);
  if (!targets.length) throw new Error('nothing to grow nearby');
  let used = 0;
  for (const block of targets) {
    if (used >= count || !findItem('bone_meal')) break;
    if (!(await reachBlock(block.position, 8000))) continue;
    await holdItem(findItem('bone_meal'));
    await bot.activateBlock(block).catch(() => {});
    used += 1;
    await sleep(250);
  }
  return { used, message: `used ${used} bone meal` };
}

async function cmdFish(params) {
  need();
  const rod = findItem('fishing_rod');
  if (!rod) throw new Error('I need a fishing rod');
  const water = nearestBlocks((n) => n === 'water', 16, 64).find((b) => {
    const above = bot.blockAt(b.position.offset(0, 1, 0));
    return above && above.name === 'air';
  });
  if (!water) throw new Error('no open water within 16 blocks');
  if (bot.entity.position.distanceTo(water.position) > 6) await walkNear(water.position, 4, 15000);
  const deadline = Date.now() + Math.max(10, Math.min(300, Number(params.seconds || 60))) * 1000;
  let caught = 0;
  while (Date.now() < deadline) {
    await holdItem(findItem('fishing_rod'));
    await bot.lookAt(water.position.offset(0.5, 1, 0.5), true);
    try {
      await withTimeout(bot.fish(), Math.max(5000, deadline - Date.now()), 'fishing');
      caught += 1;
      await sleep(500);
    } catch (err) {
      bot.activateItem(); // reel in
      break;
    }
  }
  if (!caught) throw new Error('nothing bit');
  await pickUpNear(6, 5000);
  return { caught, message: `caught ${caught} fish` };
}

async function feedOrUse(entity, item) {
  if (!(await reachEntity(entity, 3, 12000))) return false;
  await holdItem(item);
  await bot.lookAt(entity.position.offset(0, (entity.height || 1) * 0.6, 0), true);
  await bot.activateEntity(entity);
  await sleep(300);
  return true;
}

async function cmdShear(params) {
  need();
  if (!findItem('shears')) throw new Error('I need shears');
  const radius = Math.max(2, Math.min(32, Number(params.radius || 16)));
  const count = Math.max(1, Math.min(16, Number(params.count || 4)));
  const sheep = entitiesNear((e) => mobName(e) === 'sheep', radius);
  if (!sheep.length) throw new Error(`no sheep within ${radius} blocks`);
  let sheared = 0;
  for (const target of sheep.slice(0, count)) {
    if (await feedOrUse(target, findItem('shears'))) sheared += 1;
  }
  await pickUpNear(8, 6000);
  return { sheared, message: `sheared ${sheared} sheep` };
}

async function cmdMilk(params) {
  need();
  const bucket = findItem((it) => it.name === 'bucket');
  if (!bucket) throw new Error('I need an empty bucket');
  const radius = Math.max(2, Math.min(32, Number(params.radius || 16)));
  const cow = entitiesNear((e) => ['cow', 'mooshroom', 'goat'].includes(mobName(e)), radius)[0];
  if (!cow) throw new Error(`no cow within ${radius} blocks`);
  await feedOrUse(cow, bucket);
  if (!findItem('milk_bucket')) throw new Error('the cow did not give milk');
  return { message: `milked a ${mobName(cow)}` };
}

const BREED_FOOD = {
  cow: ['wheat'], sheep: ['wheat'], mooshroom: ['wheat'], goat: ['wheat'],
  pig: ['carrot', 'potato', 'beetroot'], chicken: ['wheat_seeds', 'beetroot_seeds', 'melon_seeds', 'pumpkin_seeds'],
  rabbit: ['carrot', 'golden_carrot', 'dandelion'], horse: ['golden_carrot', 'golden_apple'], donkey: ['golden_carrot', 'golden_apple'],
  wolf: ['cooked_beef', 'beef', 'cooked_porkchop', 'porkchop', 'cooked_chicken', 'chicken', 'rotten_flesh'],
  cat: ['cod', 'salmon'], llama: ['hay_block'], fox: ['sweet_berries', 'glow_berries'], turtle: ['seagrass'],
  frog: ['slime_ball'], camel: ['cactus'], armadillo: ['spider_eye'], strider: ['warped_fungus'], hoglin: ['crimson_fungus'],
  bee: ['dandelion', 'poppy', 'sunflower'], panda: ['bamboo'], axolotl: ['tropical_fish_bucket'], sniffer: ['torchflower_seeds'],
};

async function cmdBreed(params) {
  need();
  const radius = Math.max(2, Math.min(32, Number(params.radius || 16)));
  const wanted = cleanName(params.animal);
  const kinds = wanted ? [wanted] : Object.keys(BREED_FOOD);
  for (const kind of kinds) {
    const foods = BREED_FOOD[kind];
    if (!foods) throw new Error(`I don't know what ${kind}s eat`);
    const food = foods.map((f) => findItem((it) => it.name === f)).find(Boolean);
    if (!food) {
      if (wanted) throw new Error(`I need ${foods[0].replace(/_/g, ' ')} to breed ${kind}s`);
      continue;
    }
    const animals = entitiesNear((e) => mobName(e) === kind, radius);
    if (animals.length < 2) {
      if (wanted) throw new Error(`need two ${kind}s nearby`);
      continue;
    }
    let fed = 0;
    for (const animal of animals.slice(0, 2)) {
      const item = foods.map((f) => findItem((it) => it.name === f)).find(Boolean);
      if (item && (await feedOrUse(animal, item))) fed += 1;
    }
    return { fed, message: `fed ${fed} ${kind}(s)` + (fed >= 2 ? ' - a baby is on the way' : '') };
  }
  throw new Error('no animals I have food for');
}

/* --- crafting & storage */
async function ensureStation(names, itemName, label) {
  const ids = blockIds((n) => names.includes(n));
  let block = bot.findBlock({ matching: ids, maxDistance: 32 });
  if (!block) {
    const item = findItem((it) => it.name === itemName);
    if (!item) return null;
    const base = bot.entity.position.floored();
    for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, -1]]) {
      if (await placeAt(base.offset(dx, 0, dz), item)) break;
    }
    block = bot.findBlock({ matching: ids, maxDistance: 6 });
    if (block) log('info', `placed a ${label}`);
  }
  if (block && !(await reachBlock(block.position, 20000))) throw new Error(`cannot reach the ${label}`);
  return block;
}

async function cmdCraft(params) {
  need();
  const name = cleanName(params.item);
  if (!name) throw new Error('craft what? (item)');
  const info = registry().itemsByName[name];
  if (!info) throw new Error(`unknown item ${name}`);
  const count = Math.max(1, Math.min(64, Number(params.count || 1)));
  let recipe = bot.recipesFor(info.id, null, 1, null)[0];
  let table = null;
  if (!recipe) {
    table = await ensureStation(['crafting_table'], 'crafting_table', 'crafting table');
    if (!table && countItem('crafting_table') === 0) {
      // no table anywhere: make one from planks (or logs -> planks) the way a player starts out
      const planks = invItems().find((it) => it.name.endsWith('_planks'));
      if (planks || invItems().find((it) => /_log$|_stem$/.test(it.name))) {
        if (!planks) {
          const logItem = invItems().find((it) => /_log$|_stem$/.test(it.name));
          const plankName = logItem.name.replace(/_log$|_stem$/, '_planks');
          const plankInfo = registry().itemsByName[plankName];
          const plankRecipe = plankInfo && bot.recipesFor(plankInfo.id, null, 1, null)[0];
          if (plankRecipe) await bot.craft(plankRecipe, 1, null);
        }
        const tableRecipe = bot.recipesFor(registry().itemsByName.crafting_table.id, null, 1, null)[0];
        if (tableRecipe) await bot.craft(tableRecipe, 1, null);
        table = await ensureStation(['crafting_table'], 'crafting_table', 'crafting table');
      }
    }
    recipe = bot.recipesFor(info.id, null, 1, table)[0];
  }
  if (!recipe) throw new Error(`missing ingredients for ${name}` + (table ? '' : ' (or a crafting table)'));
  const per = recipe.result && recipe.result.count ? recipe.result.count : 1;
  const times = Math.max(1, Math.ceil(count / per));
  const before = countItem(name);
  await withTimeout(bot.craft(recipe, times, table || undefined), 30000, 'crafting');
  const made = countItem(name) - before;
  return { crafted: made, item: name, message: `crafted ${made} ${name.replace(/_/g, ' ')}` };
}

const FUELS = [['coal', 8], ['charcoal', 8], ['coal_block', 80], ['blaze_rod', 12], ['lava_bucket', 100]];

function pickFuel(amount, input) {
  for (const [name, per] of FUELS) {
    const item = findItem((it) => it.name === name);
    if (item) return [item, Math.min(item.count, Math.ceil(amount / per))];
  }
  const wood = findItem((it) => /_planks$|_log$|_stem$/.test(it.name) && it.name !== input);
  if (wood) return [wood, Math.min(wood.count, Math.ceil(amount / 1.5))];
  const stick = findItem((it) => it.name === 'stick');
  return stick ? [stick, Math.min(stick.count, amount * 2)] : [null, 0];
}

async function cmdSmelt(params) {
  need();
  const name = cleanName(params.item);
  const input = name ? findItem((it) => it.name === name) || findItem(name) : null;
  if (!input) throw new Error(name ? `I have no ${name}` : 'smelt what? (item)');
  const count = Math.max(1, Math.min(input.count, Number(params.count || 8)));
  const block = await ensureStation(['furnace', 'blast_furnace', 'smoker'], 'furnace', 'furnace');
  if (!block) throw new Error('no furnace nearby (and none to place)');
  const furnace = await withTimeout(bot.openFurnace(block), 8000, 'opening the furnace');
  let taken = 0;
  try {
    if (!furnace.fuelItem()) {
      const [fuel, amount] = pickFuel(count, input.name);
      if (!fuel) throw new Error('no fuel (coal, charcoal, planks or logs)');
      await furnace.putFuel(fuel.type, null, amount);
    }
    await furnace.putInput(input.type, null, count);
    const budget = Math.max(10, Math.min(600, Number(params.seconds || 90))) * 1000;
    const deadline = Date.now() + Math.min(budget, count * 10500 + 3000);
    while (Date.now() < deadline && taken < count) {
      await sleep(1000);
      const out = furnace.outputItem();
      if (out) {
        taken += out.count;
        await furnace.takeOutput();
      }
    }
  } finally {
    furnace.close();
  }
  if (!taken) throw new Error('nothing came out yet - it keeps smelting in the furnace');
  const left = count - taken;
  return { smelted: taken, message: `smelted ${taken} ${input.name.replace(/_/g, ' ')}` + (left > 0 ? ` (${left} still in the furnace)` : '') };
}

const KEEP = /(_sword|_pickaxe|_axe|_shovel|_hoe|_helmet|_chestplate|_leggings|_boots|^bow$|^crossbow$|^shield$|^totem_of_undying$|^torch$|^fishing_rod$|^shears$|^bucket$|^water_bucket$|^elytra$)/;

async function openNearestChest() {
  const ids = blockIds((n) => ['chest', 'trapped_chest', 'barrel', 'ender_chest'].includes(n) || n.endsWith('shulker_box'));
  const block = bot.findBlock({ matching: ids, maxDistance: 32 });
  if (!block) throw new Error('no chest within 32 blocks');
  if (!(await reachBlock(block.position, 20000))) throw new Error('cannot reach the chest');
  return withTimeout(bot.openContainer(block), 8000, 'opening the chest');
}

async function cmdDeposit(params) {
  need();
  const wanted = cleanName(params.item);
  const foods = registry().foodsByName || {};
  const chest = await openNearestChest();
  let stored = 0;
  try {
    for (const item of invItems()) {
      if (wanted && !item.name.includes(wanted)) continue;
      if (!wanted && !params.all && (KEEP.test(item.name) || foods[item.name])) continue;
      try {
        await chest.deposit(item.type, null, item.count);
        stored += item.count;
      } catch (err) {
        if (/full|no space/i.test(err.message)) break;
      }
    }
  } finally {
    chest.close();
  }
  if (!stored) throw new Error('nothing to put away (or the chest is full)');
  return { stored, message: `put ${stored} item(s) into the chest` };
}

async function cmdWithdraw(params) {
  need();
  const wanted = cleanName(params.item);
  const chest = await openNearestChest();
  try {
    const items = chest.containerItems();
    const item = items.find((it) => it.name === wanted) || items.find((it) => wanted && it.name.includes(wanted));
    if (!item) throw new Error(`the chest has no ${wanted || 'items'}`);
    const available = items.filter((it) => it.name === item.name).reduce((s, it) => s + it.count, 0);
    const count = Math.max(1, Math.min(available, Number(params.count || 16)));
    await chest.withdraw(item.type, null, count);
    return { took: count, item: item.name, message: `took ${count} ${item.name.replace(/_/g, ' ')} from the chest` };
  } finally {
    chest.close();
  }
}

async function cmdChestContents() {
  need();
  const chest = await openNearestChest();
  try {
    const totals = {};
    for (const it of chest.containerItems()) totals[it.name] = (totals[it.name] || 0) + it.count;
    const items = Object.entries(totals).map(([name, count]) => ({ name, count })).sort((a, b) => b.count - a.count);
    const summary = items.slice(0, 8).map((i) => `${i.count} ${i.name}`).join(', ') || 'nothing';
    return { items, message: `the chest has ${summary}${items.length > 8 ? ' ...' : ''}` };
  } finally {
    chest.close();
  }
}

function describeStack(item) {
  return item ? `${item.name} x${item.count}` : '';
}

async function cmdTrade(params) {
  need();
  const villager = entitiesNear((e) => ['villager', 'wandering_trader'].includes(mobName(e)), 16)[0];
  if (!villager) throw new Error('no villager within 16 blocks');
  if (!(await reachEntity(villager, 3, 15000))) throw new Error('cannot reach the villager');
  const window = await withTimeout(bot.openVillager(villager), 8000, 'opening trades');
  try {
    const offers = (window.trades || []).map((t, index) => ({
      index,
      give: [describeStack(t.inputItem1), describeStack(t.inputItem2)].filter(Boolean).join(' + '),
      get: describeStack(t.outputItem),
      disabled: Boolean(t.tradeDisabled),
    }));
    if (params.index === undefined || params.index === null) {
      return { trades: offers, message: 'the villager offers: ' + offers.map((o) => `#${o.index} ${o.give} -> ${o.get}`).join('; ') };
    }
    const index = Number(params.index);
    if (!offers[index]) throw new Error(`the villager has no offer #${index}`);
    if (offers[index].disabled) throw new Error(`offer #${index} is sold out`);
    const times = Math.max(1, Math.min(12, Number(params.times || 1)));
    await withTimeout(bot.trade(window, index, times), 15000, 'trading');
    return { traded: times, message: `traded ${offers[index].give} for ${offers[index].get} x${times}` };
  } finally {
    window.close();
  }
}

/* --- building */
async function torchSpot(pos) {
  const below = bot.blockAt(pos.offset(0, -1, 0));
  const here = bot.blockAt(pos);
  return Boolean(below && here && below.boundingBox === 'block' && here.name === 'air');
}

async function cmdPlaceTorch() {
  need();
  const torch = findItem((it) => it.name === 'torch' || it.name === 'soul_torch');
  if (!torch) throw new Error('I have no torches');
  const base = bot.entity.position.floored();
  for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]]) {
    const pos = base.offset(dx, 0, dz);
    if ((await torchSpot(pos)) && (await placeAt(pos, torch))) {
      return { placed_count: 1, message: 'placed a torch' };
    }
  }
  throw new Error('no spot for a torch next to me');
}

async function cmdLightUp(params) {
  need();
  const radius = Math.max(3, Math.min(16, Number(params.radius || 8)));
  const count = Math.max(1, Math.min(32, Number(params.count || 6)));
  if (!findItem((it) => it.name === 'torch')) throw new Error('I have no torches');
  const floors = bot.findBlocks({
    matching: (block) => block && block.boundingBox === 'block',
    maxDistance: radius,
    count: 400,
  });
  const dark = [];
  for (const pos of floors) {
    const above = bot.blockAt(pos.offset(0, 1, 0));
    const head = bot.blockAt(pos.offset(0, 2, 0));
    if (!above || above.name !== 'air' || !head || head.boundingBox !== 'empty') continue;
    if ((above.light ?? 15) > 7) continue;
    dark.push(above.position);
  }
  dark.sort((a, b) => bot.entity.position.distanceTo(a) - bot.entity.position.distanceTo(b));
  const chosen = [];
  for (const pos of dark) {
    if (chosen.length >= count) break;
    if (chosen.every((other) => other.distanceTo(pos) >= 6)) chosen.push(pos);
  }
  if (!chosen.length) throw new Error('it is already bright enough here');
  let placed = 0;
  for (const pos of chosen) {
    const torch = findItem((it) => it.name === 'torch');
    if (!torch) break;
    if (!(await reachBlock(pos, 12000))) continue;
    if (await placeAt(pos, torch)) placed += 1;
  }
  if (!placed) throw new Error('could not place torches');
  return { placed_count: placed, message: `placed ${placed} torch(es) on dark spots` };
}

async function cmdBuildShelter(params) {
  need();
  stopPath();
  bot.clearControlStates();
  const base = bot.entity.position.floored();
  const ring = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const plan = [];
  for (const [dx, dz] of ring) plan.push(base.offset(dx, 0, dz));
  for (const [dx, dz] of ring) plan.push(base.offset(dx, 1, dz));
  for (const [dx, dz] of ring) plan.push(base.offset(dx, 2, dz));
  plan.push(base.offset(0, 2, 0));
  let placed = 0;
  let solid = 0;
  for (const pos of plan) {
    const here = bot.blockAt(pos);
    if (here && here.boundingBox === 'block') {
      solid += 1;
      continue;
    }
    const block = buildingItem(cleanName(params.block));
    if (!block) break;
    if (await placeAt(pos, block)) {
      placed += 1;
      solid += 1;
    }
  }
  if (!placed && solid < plan.length) throw new Error('I have no blocks to build with');
  const complete = solid >= plan.length;
  return { placed_count: placed, complete,
    message: `built a shelter (${placed} block(s))` + (complete ? '' : ' - ran out of blocks, it has gaps') };
}

async function cmdPillarUp(params) {
  need();
  const height = Math.max(1, Math.min(32, Number(params.height || 4)));
  const V = Vec3();
  let placed = 0;
  stopPath();
  await bot.look(bot.entity.yaw, -Math.PI / 2, true);
  for (let i = 0; i < height; i += 1) {
    const block = buildingItem(cleanName(params.block));
    if (!block) break;
    const head = bot.blockAt(bot.entity.position.offset(0, 2.2, 0));
    if (head && head.boundingBox === 'block') break; // ceiling
    await holdItem(block);
    const startY = bot.entity.position.y;
    const ground = bot.blockAt(bot.entity.position.offset(0, -0.5, 0));
    if (!ground || ground.boundingBox !== 'block') break;
    bot.setControlState('jump', true);
    const until = Date.now() + 1500;
    while (Date.now() < until && bot.entity.position.y < startY + 1.05) await sleep(25);
    try {
      await withTimeout(bot.placeBlock(ground, new V(0, 1, 0)), 2500, 'placing');
      placed += 1;
    } catch (err) {
      const now = bot.blockAt(ground.position.offset(0, 1, 0));
      if (now && now.boundingBox === 'block') placed += 1;
      else break;
    } finally {
      bot.setControlState('jump', false);
    }
    await sleep(250);
  }
  if (!placed) throw new Error('could not tower up (blocks? ceiling?)');
  return { placed_count: placed, message: `towered up ${placed} block(s)` };
}

const USABLE = /(_door|_trapdoor|_fence_gate|_button|^lever$|^bell$|^note_block$|^repeater$|^comparator$|^daylight_detector$)/;

async function cmdActivate(params) {
  need();
  let block = null;
  if ([params.x, params.y, params.z].every((v) => v !== undefined && v !== null && !Number.isNaN(Number(v)))) {
    const V = Vec3();
    block = bot.blockAt(new V(Math.floor(Number(params.x)), Math.floor(Number(params.y)), Math.floor(Number(params.z))));
  } else {
    const wanted = cleanName(params.block);
    block = nearestBlocks((n) => USABLE.test(n) && (!wanted || n.includes(wanted)), 16, 16)[0] || null;
  }
  if (!block) throw new Error('nothing to use nearby (door, button, lever, trapdoor, gate)');
  if (!(await reachBlock(block.position, 15000))) throw new Error(`cannot reach the ${block.name}`);
  await bot.activateBlock(block);
  return { block: block.name, message: `used the ${block.name.replace(/_/g, ' ')}` };
}

/* --- moving & combat */
const RIDEABLE = /(boat|raft|minecart|horse|donkey|mule|pig|strider|camel|llama)/;

async function cmdRide() {
  need();
  if (bot.vehicle) throw new Error('already riding');
  const target = entitiesNear((e) => RIDEABLE.test(mobName(e)) && !/chest_|hopper|tnt|furnace|command|spawner/.test(mobName(e)), 8)[0];
  if (!target) throw new Error('nothing to ride within 8 blocks');
  if (!(await reachEntity(target, 3, 10000))) throw new Error(`cannot reach the ${mobName(target)}`);
  bot.mount(target);
  await sleep(600);
  return { riding: mobName(target), message: `got on the ${mobName(target).replace(/_/g, ' ')}` };
}

async function cmdDismount() {
  need();
  if (!bot.vehicle) throw new Error('not riding anything');
  bot.dismount();
  return { message: 'got off' };
}

async function cmdShoot(params) {
  need();
  const bow = findItem((it) => it.name === 'bow' || it.name === 'crossbow');
  if (!bow) throw new Error('I need a bow');
  if (!findItem((it) => it.name.endsWith('arrow'))) throw new Error('I have no arrows');
  const wanted = cleanName(params.mob);
  const range = Math.max(4, Math.min(48, Number(params.range || 24)));
  const shots = Math.max(1, Math.min(10, Number(params.shots || 3)));
  const targets = entitiesNear((e) => {
    if (e.type === 'player' || e.username || e.name === 'item') return false;
    return wanted ? mobName(e) === wanted : HOSTILE_ENTITIES.has(mobName(e));
  }, range);
  const target = targets[0];
  if (!target) throw new Error(wanted ? `no ${wanted} within ${range} blocks` : `no hostile mob within ${range} blocks`);
  await holdItem(bow);
  let fired = 0;
  for (let i = 0; i < shots && bot.entities[target.id] && target.isValid !== false; i += 1) {
    if (!findItem((it) => it.name.endsWith('arrow'))) break;
    const distance = bot.entity.position.distanceTo(target.position);
    const drop = 0.0025 * distance * distance; // arrows fall: aim a little higher further away
    await bot.lookAt(target.position.offset(0, (target.height || 1.6) * 0.7 + drop, 0), true);
    bot.activateItem();
    await sleep(bow.name === 'crossbow' ? 1300 : 1050);
    await bot.lookAt(target.position.offset(0, (target.height || 1.6) * 0.7 + drop, 0), true);
    bot.deactivateItem();
    fired += 1;
    await sleep(350);
  }
  const killed = !bot.entities[target.id] || target.isValid === false;
  return { target: mobName(target), shots: fired, kills: killed ? 1 : 0,
    message: `shot ${fired} arrow(s) at a ${mobName(target)}` + (killed ? ' and got it' : '') };
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
  // skills (pymc_bot/skills.py)
  drink: cmdDrink,
  surface: cmdSurface,
  extinguish: cmdExtinguish,
  respawn: cmdRespawn,
  find_block: cmdFindBlock,
  chop_tree: cmdChopTree,
  mine_ore: cmdMineOre,
  gather: cmdGather,
  harvest: cmdHarvest,
  plant: cmdPlant,
  till: cmdTill,
  bonemeal: cmdBonemeal,
  fish: cmdFish,
  shear: cmdShear,
  milk: cmdMilk,
  breed: cmdBreed,
  craft: cmdCraft,
  smelt: cmdSmelt,
  deposit: cmdDeposit,
  withdraw: cmdWithdraw,
  chest_contents: cmdChestContents,
  trade: cmdTrade,
  place_torch: cmdPlaceTorch,
  light_up: cmdLightUp,
  build_shelter: cmdBuildShelter,
  pillar_up: cmdPillarUp,
  activate: cmdActivate,
  ride: cmdRide,
  dismount: cmdDismount,
  shoot: cmdShoot,
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
