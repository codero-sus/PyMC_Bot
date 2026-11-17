'use strict';

// Conservative heuristics, not a guarantee against concealed traps or server plugins.
const DANGER = /^(?:lava|fire|soul_fire|magma_block|cactus|sweet_berry_bush|powder_snow|cobweb|web|pointed_dripstone|tnt|respawn_anchor|sculk_sensor|calibrated_sculk_sensor|sculk_shrieker|tripwire|tripwire_hook|end_portal|end_portal_frame|nether_portal|wither_rose|campfire|soul_campfire)$|_pressure_plate$|_trapdoor$/;
const ADJACENT_TRAP = /^(?:tnt|tripwire|tripwire_hook|sculk_sensor|calibrated_sculk_sensor|sculk_shrieker|dispenser|dropper|observer|redstone_wire|redstone_torch|redstone_wall_torch)$|_pressure_plate$/;
const FALLING = /^(?:sand|red_sand|gravel|anvil|chipped_anvil|damaged_anvil|dragon_egg|suspicious_sand|suspicious_gravel)$|_concrete_powder$/;
const LIQUID = /^(?:water|lava|flowing_water|flowing_lava)$/;

function nameOf(block) {
  return block && typeof block.name === 'string' ? block.name.replace(/^minecraft:/, '') : '';
}

function at(bot, position, dx, dy, dz) {
  try { return bot.blockAt(position.offset(dx, dy, dz)); } catch (_) { return null; }
}

function stepRisk(bot, block) {
  if (!block || !block.position) return 100;
  const pos = block.position;
  // Include the block underfoot, the body and adjacent redstone mechanisms.
  for (const [x, y, z] of [[0, -1, 0], [0, 0, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]) {
    const seen = at(bot, pos, x, y, z);
    if (!seen) return 100; // unloaded terrain is not evidence of a safe path
    const name = nameOf(seen);
    if (!name || (x === 0 && z === 0 && DANGER.test(name)) || ADJACENT_TRAP.test(name)) return 100;
  }
  return 0;
}

function digRisk(bot, block) {
  if (!block || !block.position || !bot || !bot.entity || !bot.entity.position) return 'missing block or position';
  const p = block.position, feet = bot.entity.position;
  if (Math.floor(feet.x) === Math.floor(p.x) && Math.floor(feet.z) === Math.floor(p.z)
      && Math.floor(feet.y) - 1 === Math.floor(p.y)) return 'block supports the bot';
  if (!nameOf(block)) return 'unknown block';
  if (DANGER.test(nameOf(block)) || ADJACENT_TRAP.test(nameOf(block))) return 'suspicious or dangerous block';
  for (const [x, y, z] of [[0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]]) {
    const neighbour = at(bot, p, x, y, z);
    if (!neighbour) return 'unobserved adjacent block';
    const name = nameOf(neighbour);
    if (!name) return 'unknown adjacent block';
    if (LIQUID.test(name)) return `adjacent ${name}`;
    if (y === 1 && FALLING.test(name)) return `falling block above (${name})`;
    if (DANGER.test(name) || ADJACENT_TRAP.test(name)) return `nearby hazard (${name})`;
  }
  return null;
}

function observedHazards(bot) {
  if (!bot || !bot.entity || !bot.entity.position) return [];
  const result = new Set();
  const origin = bot.entity.position;
  for (let x = -1; x <= 1; x++) {
    for (let y = -1; y <= 1; y++) {
      for (let z = -1; z <= 1; z++) {
        const block = at(bot, origin, x, y, z);
        const name = nameOf(block);
        if (DANGER.test(name) || ADJACENT_TRAP.test(name)) result.add(name);
      }
    }
  }
  return [...result].sort().slice(0, 12);
}

function safeMovements(bot, pathfinder) {
  const moves = new pathfinder.Movements(bot);
  // Route planning is not permission to modify the world. Explicit skills handle digging/placing.
  moves.canDig = false;
  moves.allow1by1towers = false;
  moves.allowParkour = false;
  moves.scafoldingBlocks = [];
  moves.maxDropDown = 3;
  moves.infiniteLiquidDropdownDistance = false;
  moves.exclusionAreasStep.push((block) => stepRisk(bot, block));
  return moves;
}

module.exports = { DANGER, digRisk, stepRisk, observedHazards, safeMovements };
