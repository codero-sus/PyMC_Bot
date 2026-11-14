'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2], 'utf8');
const open = src.slice(src.indexOf('async function openNearestChest('), src.indexOf('async function cmdDeposit('));
const inspect = src.slice(src.indexOf('async function cmdChestContents('), src.indexOf('function describeStack('));
let closed = false;
const position = {x: 4, y: 65, z: -9};
const chest = {containerItems: () => [{name: 'coal', count: 4}, {name: 'coal', count: 2}], close: () => {closed = true;}};
const bot = {findBlock: () => ({position}), openContainer: async () => chest, game: {dimension: 'minecraft:overworld'}};
const ctx = vm.createContext({bot, need: () => {}, blockIds: () => [1], reachBlock: async () => true,
  withTimeout: promise => promise, Object, String, Error});
vm.runInContext(open + inspect, ctx);
(async () => {
  const result = await ctx.cmdChestContents();
  assert.strictEqual(result.position.x, 4);
  assert.strictEqual(result.dimension, 'minecraft:overworld');
  assert.strictEqual(result.items[0].name, 'coal');
  assert.strictEqual(result.items[0].count, 6);
  assert(closed);
})().catch(err => { console.error(err); process.exitCode = 1; });
