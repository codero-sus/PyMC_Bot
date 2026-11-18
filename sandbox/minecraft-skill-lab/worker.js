'use strict';
const mineflayer = require('mineflayer');
const pathfinder = require('mineflayer-pathfinder');
const bot = mineflayer.createBot({ host: '127.0.0.1', port: 25565, username: 'SkillLab', auth: 'offline', version: '26.1' });
bot.on('error', () => process.exit(2));
bot.once('spawn', async () => {
  try {
    bot.loadPlugin(pathfinder.pathfinder);
    const skill = require('/tmp/candidate.cjs');
    if (typeof skill !== 'function') throw new Error('expected an exported async function');
    await skill(bot);
    bot.quit();
    process.exit(0);
  } catch (_) {
    try { bot.quit(); } catch (_) { /* disconnected */ }
    process.exit(3);
  }
});
