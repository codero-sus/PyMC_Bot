'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const { AsyncLocalStorage } = require('async_hooks');
const source = fs.readFileSync(process.argv[2], 'utf8');
function extract(start, end) {
  const a = source.indexOf(start), b = source.indexOf(end, a);
  if (a < 0 || b < 0) throw new Error(`missing ${start}`);
  return source.slice(a, b);
}
const replies = [];
let controlsCleared = false, diggingStopped = false, afterSleep = false;
const context = vm.createContext({
  actionContext: new AsyncLocalStorage(), activeActions: new Set(), AbortController, setTimeout, clearTimeout,
  bot: { stopDigging: () => { diggingStopped = true; } },
  cmdStopMotion: () => { controlsCleared = true; },
  send: (value) => replies.push(value),
  COMMANDS: {},
});
vm.runInContext(extract('function checkAction() {', 'function withTimeout('), context);
vm.runInContext(extract('function cmdCancelActions() {', 'function cmdPathfindTo('), context);
vm.runInContext(extract('async function handle(request) {', 'const rl = readline.createInterface'), context);
context.COMMANDS.cancel_actions = context.cmdCancelActions;
context.COMMANDS.test = async () => { await context.sleep(10000); afterSleep = true; };
context.COMMANDS.ping = () => ({ pong: true });
let digCalls = 0;
context.nearestBlocks = () => [{position: {toString: () => 'block'}}];
context.digBlock = async () => { digCalls++; await context.sleep(10000); return true; };
context.pickUpNear = async () => 0;
vm.runInContext(extract('async function mineMatching(', '/** Place `item`'), context);
context.COMMANDS.gather = async () => context.mineMatching(() => true, 5, 8, 10000, 'block');
(async () => {
  const long = context.handle({id: 1, cmd: 'test'});
  await new Promise((resolve) => setTimeout(resolve, 10));
  await context.handle({id: 2, cmd: 'cancel_actions'});
  await long;
  assert(controlsCleared && diggingStopped && !afterSleep);
  assert(replies.some((r) => r.id === 1 && !r.ok && /cancelled/.test(r.error)));
  assert(replies.some((r) => r.id === 2 && r.ok));
  await context.handle({id: 3, cmd: 'ping'});
  assert(replies.some((r) => r.id === 3 && r.ok));
  const gathering = context.handle({id: 4, cmd: 'gather'});
  await new Promise((resolve) => setTimeout(resolve, 10));
  await context.handle({id: 5, cmd: 'cancel_actions'});
  await gathering;
  assert.strictEqual(digCalls, 1);
  assert(replies.some((r) => r.id === 4 && !r.ok && /cancelled/.test(r.error)));
})().catch((error) => { console.error(error); process.exitCode = 1; });
