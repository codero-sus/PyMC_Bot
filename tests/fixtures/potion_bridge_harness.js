'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const { EventEmitter } = require('events');
const { classifyPotion } = require('../../pymc_bot/node/potions');

const potion = name => ({nbt: {type: 'compound', value: {Potion: {type: 'string', value: `minecraft:${name}`}}}});
assert.strictEqual(classifyPotion(potion('strong_healing')).kind, 'good');
assert.strictEqual(classifyPotion(potion('long_poison')).kind, 'bad');
assert.strictEqual(classifyPotion(potion('water')).kind, 'unknown');
assert.strictEqual(classifyPotion({components: [{type: 'potion_contents', data: {potionId: 5, customEffects: []}}]}).kind, 'unknown');
const mixed = potion('healing');
mixed.nbt.value.CustomPotionEffects = {type: 'list', value: {type: 'compound', value: [{Id: {type: 'string', value: 'minecraft:poison'}}]}};
assert.strictEqual(classifyPotion(mixed).kind, 'bad');
mixed.nbt.value.CustomPotionEffects.value.value[0].Id.value = 999;
assert.strictEqual(classifyPotion(mixed).kind, 'unknown');

class Vec {
  constructor(x, y, z) { this.x = x; this.y = y; this.z = z; }
  offset(x, y, z) { return new Vec(this.x + x, this.y + y, this.z + z); }
  floored() { return new Vec(Math.floor(this.x), Math.floor(this.y), Math.floor(this.z)); }
  distanceTo(other) { return Math.hypot(this.x - other.x, this.y - other.y, this.z - other.z); }
  clone() { return new Vec(this.x, this.y, this.z); }
}
const oldMovements = {};
const moves = [];
const bot = Object.assign(new EventEmitter(), {entity: {position: new Vec(0, 64, 0)}, entities: {}, registry: {entitiesByName: {potion: {metadataKeys: ['item_stack']}}},
  blockAt: pos => pos.y < 64 ? {name: 'stone', boundingBox: 'block'} : {name: 'air', boundingBox: 'empty'},
  pathfinder: {movements: oldMovements, setMovements: m => moves.push(m), setGoal: g => moves.push(g), goto: async g => moves.push(g)}});
const source = fs.readFileSync(process.argv[2], 'utf8');
const observation = source.slice(source.indexOf('function potionObservation('), source.indexOf('function buildState('));
const reaction = source.slice(source.indexOf('async function cmdReactPotion('), source.indexOf('/* ------------------------------------------------------------------ vision */'));
const context = vm.createContext({bot, classifyPotion, Number, Math, String, Object, Error,
  pathfinder: {Movements: class {}}, goals: {GoalNear: class {constructor(x, y, z, r) {Object.assign(this, {x,y,z,r});}}},
  need: () => {}, checkAction: () => {}, withTimeout: promise => promise});
vm.runInContext(observation + reaction, context);
(async () => {
  const drop = {id: 2, name: 'item', position: new Vec(3, 64, 0), getDroppedItem: () => ({name: 'splash_potion', ...potion('healing')})};
  bot.entities[2] = drop;
  assert.strictEqual(context.potionObservation(drop).kind, 'good');
  assert((await context.cmdReactPotion({entity_id: 2, mode: 'seek'})).message.includes('pickup not confirmed'));
  assert.strictEqual(moves[0].canDig, false);
  assert.strictEqual(moves[0].allowParkour, false);
  assert.strictEqual(moves[moves.length - 1], oldMovements);
  const bad = {id: 3, name: 'potion', position: new Vec(2, 64, 0), metadata: [potion('poison')]};
  bot.entities[3] = bad;
  assert.strictEqual(context.potionObservation(bad).kind, 'bad');
  await assert.rejects(context.cmdReactPotion({entity_id: 2, mode: 'seek'}), /another potion hazard/);
  assert((await context.cmdReactPotion({entity_id: 3, mode: 'avoid'})).message.includes('moved away'));
  bad.metadata = [potion('healing')];
  await assert.rejects(context.cmdReactPotion({entity_id: 3, mode: 'avoid'}), /not a harmful/);
  bad.velocity = {y: -0.1};
  assert((await context.cmdReactPotion({entity_id: 3, mode: 'seek'})).message.includes('effect is not confirmed'));
  bot.pathfinder.goto = () => new Promise(() => {});
  const running = context.cmdReactPotion({entity_id: 3, mode: 'seek'});
  bot.emit('entityGone', bad);
  await assert.rejects(running, /landed or disappeared/);
  assert.strictEqual(bot.listenerCount('entityGone'), 0);
  assert.strictEqual(bot.listenerCount('entityMoved'), 0);
  bot.pathfinder.goto = async g => moves.push(g);
  bad.velocity.y = 0.5;
  await assert.rejects(context.cmdReactPotion({entity_id: 3, mode: 'seek'}), /not close enough/);
  bad.metadata = [];
  assert.strictEqual(context.potionObservation(bad).kind, 'unknown');
  assert((await context.cmdReactPotion({entity_id: 3, mode: 'avoid'})).message.includes('moved away'));
  bot.blockAt = () => null;
  await assert.rejects(context.cmdReactPotion({entity_id: 3, mode: 'avoid'}), /no known safe/);
})().catch(err => { console.error(err); process.exitCode = 1; });
