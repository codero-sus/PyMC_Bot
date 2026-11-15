'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const code = source.slice(source.indexOf('function cmdRegistryLookup('), source.indexOf('function cleanName('));
const context = vm.createContext({bot: {version: '1.21.1', recipesAll: () => [{result: {id: 1, count: 2}, requiresTable: true,
  inShape: [[{id: 2, count: 1}, {id: -1}], [{id: 2, count: 1}]]}]}, need: () => {}, Object, Number, String, Math, Error,
  registry: () => ({items: {1: {name: 'apple'}, 2: {name: 'oak_log'}}, itemsByName: {apple: {name: 'apple', id: 1, displayName: 'Apple', stackSize: 64},
    'mod:apple_tool': {name: 'mod:apple_tool', id: 1024, displayName: 'Mod tool', stackSize: 1}},
    blocksByName: {stone: {name: 'stone', id: 2}}})});
vm.runInContext(code, context);
const result = context.cmdRegistryLookup({kind: 'item', query: 'apple', limit: 1, offset: 1});
assert.strictEqual(result.total, 2);
assert.strictEqual(result.entries[0].key, 'mod:apple_tool');
assert.strictEqual(result.entries[0].stack_size, 1);
assert.strictEqual(context.cmdRegistryLookup({kind: 'block'}).entries[0].name, 'stone');
assert.throws(() => context.cmdRegistryLookup({kind: 'recipe'}), /kind/);
assert.throws(() => context.cmdRegistryLookup({query: 'bad\nname'}), /invalid/);
assert.throws(() => context.cmdRegistryLookup({limit: 101}), /invalid/);
const recipes = context.cmdRecipeLookup({item: 'minecraft:apple'});
assert.strictEqual(recipes.total, 1);
assert.strictEqual(recipes.recipes[0].requires_table, true);
assert.strictEqual(recipes.recipes[0].shape[0][0].name, 'oak_log');
assert.strictEqual(recipes.recipes[0].shape[0][1], null);
assert.throws(() => context.cmdRecipeLookup({item: 'unknown'}), /not found/);
assert.throws(() => context.cmdRecipeLookup({item: 'apple', limit: 51}), /invalid/);
