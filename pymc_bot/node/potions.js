'use strict';

// Classify only potion data fields, never display names or particle colors. Unsupported
// server/component formats remain unknown instead of being guessed from appearance.
const GOOD = new Set(['healing', 'regeneration', 'swiftness', 'strength', 'leaping', 'fire_resistance',
  'water_breathing', 'night_vision', 'invisibility', 'slow_falling', 'luck', 'speed', 'jump_boost',
  'instant_health', 'haste', 'resistance', 'absorption', 'health_boost', 'saturation', 'conduit_power']);
const BAD = new Set(['harming', 'poison', 'slowness', 'weakness', 'turtle_master', 'instant_damage',
  'wither', 'mining_fatigue', 'nausea', 'blindness', 'hunger', 'levitation', 'unluck', 'darkness',
  'bad_omen', 'trial_omen', 'raid_omen', 'wind_charged', 'weaving', 'oozing', 'infested']);

function unwrap(value) {
  for (let i = 0; i < 5 && value && typeof value === 'object' && !Array.isArray(value) &&
      typeof value.type === 'string' && Object.prototype.hasOwnProperty.call(value, 'value'); i++) value = value.value;
  return value;
}

function classifyPotion(item, effectNames = {}) {
  if (!item || typeof item !== 'object') return {kind: 'unknown', effects: []};
  const nbt = unwrap(item.nbtData || item.nbt) || {};
  let potion = unwrap(nbt.Potion || nbt.potion);
  let custom = unwrap(nbt.CustomPotionEffects || nbt.custom_potion_effects);
  // Modern network slots store potion data as a component. potionId is numeric; without
  // a verified potion registry its meaning varies by version, so don't assume its value.
  const components = Array.isArray(item.components) ? item.components : [];
  const component = components.find(c => c && (c.type === 'potion_contents' || c.type === 'minecraft:potion_contents'));
  if (component) {
    const contents = component.data || component.value || component;
    potion = contents.potion ?? contents.potionId ?? potion;
    custom = contents.customEffects ?? contents.custom_effects ?? custom;
  }
  const values = [];
  if (potion !== undefined && potion !== null && potion !== '') {
    const value = unwrap(potion);
    values.push(typeof value === 'number' ? null : value);
  }
  custom = unwrap(custom);
  if (custom && !Array.isArray(custom) && Array.isArray(custom.value)) custom = custom.value;
  if (Array.isArray(custom)) {
    for (const entry of custom) {
      const obj = unwrap(entry) || {};
      values.push(unwrap(obj.id ?? obj.Id ?? obj.effect ?? obj));
    }
  } else if (custom != null) {
    values.push(null); // malformed custom effects must not make a potion appear safe
  }
  let unknown = !values.length, bad = false;
  const effects = [];
  for (let value of values) {
    if (typeof value === 'number') value = effectNames[value];
    if (typeof value !== 'string') { unknown = true; continue; }
    const name = value.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase()
      .replace(/^minecraft:/, '').replace(/^(strong|long)_/, '');
    effects.push(name);
    if (BAD.has(name)) bad = true;
    else if (!GOOD.has(name)) unknown = true;
  }
  return {kind: bad ? 'bad' : unknown ? 'unknown' : 'good', effects: effects.slice(0, 8)};
}

module.exports = { classifyPotion };
