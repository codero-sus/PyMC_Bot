'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('async function cmdLook(params) {');
const end = source.indexOf('function cmdStopMotion()', start);
assert(start >= 0 && end > start);
let angles;
const ctx = vm.createContext({bot: {look: async (yaw, pitch, force) => { angles = [yaw, pitch, force]; }}});
vm.runInContext(source.slice(start, end), ctx);
(async () => {
  await ctx.cmdLook({yaw: 0, pitch: Math.PI / 4});
  assert(Math.abs(angles[0] - Math.PI) < 1e-9);
  assert(Math.abs(angles[1] + Math.PI / 4) < 1e-9);
  await ctx.cmdLook({yaw: Math.PI / 2, pitch: -0.1});
  assert(Math.abs(angles[0] - Math.PI / 2) < 1e-9 && Math.abs(angles[1] - 0.1) < 1e-9);
})().catch((error) => { console.error(error); process.exitCode = 1; });
