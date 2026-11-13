'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const collect = source.slice(source.indexOf('async function cmdCollect('), source.indexOf('async function cmdAttackMob('));
const pos = x => ({x, distanceTo(other) { return Math.abs(this.x - other.x); }});
const bot = {entity: {position: pos(0)}, entities: {}, pathfinder: {setGoal(goal) {
  const match = Object.values(bot.entities).find(e => e.position.x === goal.x);
  if (match) delete bot.entities[match.id];
}}};
const context = vm.createContext({bot, pathfinder: {}, goals: {GoalNear: class {constructor(x) {this.x = x;}}},
  stopPath: () => {}, sleep: async () => {}, Date, Math, Number, String, Object, Error});
vm.runInContext(collect, context);
(async () => {
  bot.entities = {
    1: {id: 1, name: 'item', position: pos(1), getDroppedItem: () => ({name: 'dirt'})},
    2: {id: 2, name: 'item', position: pos(2), getDroppedItem: () => ({name: 'oak_log'})},
    3: {id: 3, name: 'item', position: pos(3), getDroppedItem: () => null},
    4: {id: 4, name: 'item', position: pos(4), getDroppedItem: () => {throw new Error('unknown');}},
  };
  const result = await context.cmdCollect({item: 'minecraft:oak_log'});
  assert.strictEqual(result.picked, 1);
  assert(bot.entities[1] && bot.entities[3] && bot.entities[4]);
  await assert.rejects(context.cmdCollect({item: 'oak_log'}), /nothing to pick up/);
  delete bot.entities[3]; delete bot.entities[4];
  assert.strictEqual((await context.cmdCollect({})).picked, 1);
})().catch(err => { console.error(err); process.exitCode = 1; });
