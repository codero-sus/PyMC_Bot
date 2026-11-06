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
let uses = [];
const target = { id: 1, name: 'zombie', type: 'mob', height: 1.6, isValid: true,
  position: { offset: () => ({}) } };
const bot = {
  entity: { onGround: true, velocity: { y: 0 }, position: { distanceTo: () => 2 } },
  entities: { 1: target },
  setControlState: (name, value) => uses.push([name, value]),
  lookAt: async () => {},
  activateItem: () => uses.push('activate'),
  deactivateItem: () => uses.push('deactivate'),
  attack: () => uses.push('attack'),
};
const context = { bot, Date: { now: () => clock }, Math, Number, String, Boolean, Error,
  need: () => {}, cleanName: (v) => String(v || ''), mobName: (e) => e.name,
  HOSTILE_ENTITIES: new Set(['zombie']), entitiesNear: () => [target], stopPath: () => {},
  findItem: (predicate) => [{name: weapon}, {name: 'arrow'}].find(predicate), holdItem: async () => {},
  sleep: async (ms) => { clock += ms; if (ms === 25) { bot.entity.onGround = false; bot.entity.velocity.y = -0.1; } },
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
  await assert.rejects(context.cmdShoot({ weapon: 'trident' }), /weapon must be bow or crossbow/);
})().catch((error) => { console.error(error); process.exitCode = 1; });
