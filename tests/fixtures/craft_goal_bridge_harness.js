'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const craft = source.slice(source.indexOf('async function cmdCraft('), source.indexOf('const FUELS ='));
let stations = 0, crafted = 0;
const bot = {recipesFor: () => [], craft: async () => {crafted++;}};
const context = vm.createContext({bot, need: () => {}, cleanName: s => s,
  registry: () => ({itemsByName: {torch: {id: 1}}}),
  ensureStation: async () => {stations++; return null;},
  Number, Math, Error});
vm.runInContext(craft, context);
(async () => {
  await assert.rejects(context.cmdCraft({item: 'torch', no_station: true}), /inventory-only recipe/);
  assert.strictEqual(stations, 0);
  assert.strictEqual(crafted, 0);
})().catch(err => { console.error(err); process.exitCode = 1; });
