'use strict';
// This entire process and its children must run in the disposable sandbox.
const fs = require('fs');
const { spawn } = require('child_process');

async function main() {
  let data = '';
  for await (const chunk of process.stdin) {
    data += chunk;
    if (Buffer.byteLength(data) > 100000) throw new Error('request too large');
  }
  const request = JSON.parse(data);
  if (typeof request.source !== 'string' || Buffer.byteLength(request.source) > 16001) throw new Error('invalid source');
  fs.mkdirSync('/tmp/world');
  fs.writeFileSync('/tmp/world/eula.txt', 'eula=true\n');
  fs.writeFileSync('/tmp/world/server.properties', [
    'server-ip=127.0.0.1', 'server-port=25565', 'online-mode=false', 'spawn-protection=0',
    'view-distance=3', 'simulation-distance=3', 'max-players=1', 'level-type=minecraft:flat',
    'generator-settings={"layers":[{"block":"minecraft:bedrock","height":1},{"block":"minecraft:dirt","height":2},{"block":"minecraft:grass_block","height":1}],"biome":"minecraft:plains"}',
    'enable-rcon=false', 'enable-query=false', 'enable-status=true', 'allow-flight=false', '',
  ].join('\n'));
  fs.writeFileSync('/tmp/candidate.cjs', request.source, { mode: 0o600 });
  const env = { PATH: process.env.PATH, HOME: '/tmp', NODE_PATH: '/lab/node_modules' };
  const server = spawn('java', ['-XX:ActiveProcessorCount=2', '-Xms256M', '-Xmx1024M', '-jar', '/lab/server.jar', 'nogui'], {
    cwd: '/tmp/world', stdio: ['ignore', 'pipe', 'ignore'], env,
  });
  try {
    await new Promise((resolve, reject) => {
      let tail = '';
      const timer = setTimeout(() => reject(new Error('test server startup timed out')), 80000);
      server.once('error', (error) => { clearTimeout(timer); reject(error); });
      server.once('exit', () => { clearTimeout(timer); reject(new Error('test server exited')); });
      server.stdout.on('data', (chunk) => {
        tail = (tail + chunk.toString()).slice(-4096);
        if (/Done \([\d.]+s\)!/.test(tail)) { clearTimeout(timer); resolve(); }
      });
    });
    const result = await new Promise((resolve, reject) => {
      const worker = spawn('node', ['/lab/worker.js'], { cwd: '/tmp', stdio: 'ignore', env });
      const timer = setTimeout(() => { worker.kill('SIGKILL'); reject(new Error('candidate timed out')); }, 25000);
      worker.once('error', (error) => { clearTimeout(timer); reject(error); });
      worker.once('exit', (code) => { clearTimeout(timer); resolve(code); });
    });
    if (result !== 0) throw new Error('candidate exited unsuccessfully');
    // Candidate shares a sandbox with this reporter. It can tamper with the report.
    console.log(JSON.stringify({ status: 'completed_unverified' }));
  } finally {
    server.kill('SIGTERM');
    setTimeout(() => server.kill('SIGKILL'), 2000).unref();
  }
}
main().catch(() => { console.log(JSON.stringify({ status: 'failed' })); process.exitCode = 1; });
