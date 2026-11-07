'use strict';
// Exercise the real bridge command functions with deterministic Mineflayer doubles.
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('async function cmdCriticalHit(');
const end = source.indexOf('/* ------------------------------------------------------------------ vision */', start);
assert(start >= 0 && end > start);
let clock = 0;
let weapon = 'crossbow';
let inventoryItems = [];
let towered = false;
let uses = [];
const target = { id: 1, name: 'zombie', type: 'mob', height: 1.6, isValid: true,
  position: { offset: () => ({}) } };
const bot = {
  entity: { onGround: true, isInWater: false, yaw: 0, velocity: { y: 0 }, position: { distanceTo: () => 2 } },
  isRaining: false, health: 20, inventory: { slots: { 45: {name: 'totem_of_undying'} } },
  look: async () => {},
  entities: { 1: target },
  setControlState: (name, value) => uses.push([name, value]),
  lookAt: async () => {},
  activateItem: () => uses.push('activate'),
  deactivateItem: () => uses.push('deactivate'),
  attack: () => uses.push('attack'),
};
const context = { bot, Date: { now: () => clock }, Math, Number, String, Boolean, Error,
  need: () => {}, cleanName: (v) => String(v || ''), mobName: (e) => e.name,
  HOSTILE_ENTITIES: new Set(['zombie']), entitiesNear: (predicate) => predicate(target) ? [target] : [], stopPath: () => {},
  cmdPillarUp: async () => { towered = true; },
  findItem: (predicate) => [{name: weapon}, {name: 'arrow'}, ...inventoryItems].find(predicate), holdItem: async () => {},
  sleep: async (ms) => { clock += ms; if (ms === 25 || ms === 50) { bot.entity.onGround = false; bot.entity.velocity.y = -0.1; } },
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context);
(async () => {
  await context.cmdShoot({ mob: 'zombie', shots: 1, weapon: 'crossbow' });
  assert.deepStrictEqual(uses, ['activate', 'deactivate', 'activate', 'deactivate']);
  uses = []; weapon = 'bow';
  await context.cmdShoot({ mob: 'zombie', shots: 1, weapon: 'bow' });
  assert.deepStrictEqual(uses, ['activate', 'deactivate']);
  uses = [];
  const result = await context.cmdCriticalHit({ mob: 'zombie' });
  assert(result.message.includes('not confirmed'));
  assert(uses.includes('attack'));
  assert.deepStrictEqual(uses[uses.length - 1], ['jump', false]);
  inventoryItems = [{name: 'trident'}, {name: 'iron_spear'}, {name: 'snowball'}, {name: 'mace'}, {name: 'wind_charge'}];
  await assert.rejects(context.cmdThrowTrident({mob: 'zombie'}), /confirm=true/);
  bot.isRaining = true;
  await assert.rejects(context.cmdThrowTrident({mob: 'zombie', confirm: true}), /dry weather/);
  bot.isRaining = false; uses = [];
  assert((await context.cmdThrowTrident({mob: 'zombie', confirm: true})).message.includes('not confirmed'));
  assert.deepStrictEqual(uses, ['activate', 'deactivate']);
  uses = [];
  assert((await context.cmdSpearAttack({mob: 'zombie'})).message.includes('not confirmed'));
  assert(uses.includes('attack'));
  uses = [];
  assert((await context.cmdThrowProjectile({mob: 'zombie', item: 'snowball'})).message.includes('not confirmed'));
  assert.deepStrictEqual(uses, ['activate', 'deactivate']);
  await assert.rejects(context.cmdThrowProjectile({item: 'wind_charge'}), /snowball or egg/);
  bot.entity.onGround = true; uses = [];
  await assert.rejects(context.cmdMaceSmash({mob: 'zombie', mode: 'tower'}), /confirm=true/);
  bot.inventory.slots[45] = null;
  await assert.rejects(context.cmdMaceSmash({mob: 'zombie', mode: 'tower', confirm: true}), /totem/);
  bot.inventory.slots[45] = {name: 'totem_of_undying'};
  bot.entity.onGround = true;
  assert((await context.cmdMaceSmash({mob: 'zombie', mode: 'tower', confirm: true})).message.includes('not confirmed'));
  assert(towered && uses.includes('attack'));
  bot.entity.onGround = true; uses = [];
  assert((await context.cmdMaceSmash({mob: 'zombie', mode: 'wind_charge', confirm: true})).message.includes('not confirmed'));
  assert(uses.includes('attack') && uses.includes('activate'));
  target.type = 'player'; target.username = 'Steve';
  await assert.rejects(context.cmdSpearAttack({}), /no matching hostile/);
  target.type = 'mob'; delete target.username;
  await assert.rejects(context.cmdShoot({ weapon: 'trident' }), /weapon must be bow or crossbow/);
})().catch((error) => { console.error(error); process.exitCode = 1; });
