'use strict';
const assert = require('assert');
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const start = source.indexOf('const VISION_PACKAGES =');
const end = source.indexOf('const COMMANDS =', start);
assert(start >= 0 && end > start);
let attached = 0, detached = 0, terminated = 0, disposed = 0, position, initFinished = false;
const entity = {position: {x: 2, y: 70, z: 3}, yaw: 1, pitch: 0.2};
const bot = {version: '1.21.4', entity, world: {}};
class Viewer {
  constructor() { this.world = {workers: [{terminate: () => { terminated++; }}], waitForChunksToRender: async () => {}}; this.scene = {}; this.camera = {}; }
  setVersion(v) { this.version = v; return true; }
  listen() {}
  setFirstPersonCamera(pos) { position = pos; }
  update() {}
  resetAll() {}
}
class WorldView {
  listenToBot(owner) { assert.strictEqual(owner, bot); attached++; }
  removeListenersFromBot(owner) { assert.strictEqual(owner, bot); detached++; }
  async init() { await Promise.resolve(); initFinished = true; }
  async updatePosition() { assert(initFinished); }
}
const viewerLib = { Viewer, WorldView, getBufferFromStream: async () => Buffer.from([255,216,255,0]) };
const ctx = vm.createContext({bot, Buffer, Math, Promise, global: {},
  require: (name) => {
    if (name === 'prismarine-viewer') return {viewer: viewerLib};
    if (name === 'node-canvas-webgl/lib') return {createCanvas: () => ({createJPEGStream: () => ({})})};
    if (name === 'three') return {WebGLRenderer: class {render() {} dispose() {disposed++;}}};
    if (name === 'worker_threads') return {Worker: class {}};
    throw new Error(name);
  },
  sleep: async () => {}, checkAction: () => {}, withTimeout: (p) => p,
});
vm.runInContext(source.slice(start, end), ctx);
(async () => {
  const result = await ctx.cmdSnapshot({width: 448, settle_ms: 0});
  assert.strictEqual(result.image, '/9j/AA==');
  assert.strictEqual(position, entity.position, 'camera must receive feet position; viewer adds eye height');
  assert.strictEqual(attached, 1);
  await ctx.cmdSnapshot({width: 448, settle_ms: 0});
  assert.strictEqual(attached, 1, 'renderer should be reused');
  await ctx.cmdSnapshot({width: 300, settle_ms: 0});
  assert.strictEqual(attached, 2); assert.strictEqual(detached, 1);
  ctx.closeEyes();
  assert.strictEqual(detached, 2); assert.strictEqual(terminated, 2); assert.strictEqual(disposed, 2);
})().catch((error) => { console.error(error); process.exitCode = 1; });
