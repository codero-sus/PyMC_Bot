/* PyMC_Bot control panel -- vanilla JS, no build step. Talks to the FastAPI app
 * on the same origin (relative URLs), so it works behind any proxy/preview host. */
'use strict';

const MAX_LOG_LINES = 500;
const $ = (id) => document.getElementById(id);

/* Effects live in fx.js (themes, toasts, confetti, palette...). If it failed to
 * load the panel still works: every call below degrades to a no-op. */
const ui = window.FX || {
  toast() {}, confetti() {}, hit() {}, pulse() {}, busy: (b, w) => w, motionOn: () => false,
  cssVar: (name, fallback) => fallback,
  flashText(el, text) { const node = typeof el === 'string' ? $(el) : el; if (node) node.textContent = String(text); },
  countTo(el, value, options = {}) {
    const node = typeof el === 'string' ? $(el) : el;
    if (node) node.textContent = `${Number(value).toFixed(options.digits ?? 0)}${options.suffix ?? ''}`;
  },
};

/** Escape text for innerHTML: names, items and kick messages come from the Minecraft server. */
function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* A button whose click starts a slow request shows a spinner until it finishes. */
let clickedButton = null;
let clickedAt = 0;
const busyCount = new WeakMap();
document.addEventListener('click', (event) => {
  const button = event.target.closest && event.target.closest('button.btn');
  if (button) { clickedButton = button; clickedAt = performance.now(); }
}, true);
function trackBusy() {
  const button = clickedButton && performance.now() - clickedAt < 400 ? clickedButton : null;
  if (!button) return () => {};
  busyCount.set(button, (busyCount.get(button) || 0) + 1);
  const timer = setTimeout(() => { if (busyCount.get(button)) button.classList.add('busy'); }, 150);
  return () => {
    clearTimeout(timer);
    const left = (busyCount.get(button) || 1) - 1;
    busyCount.set(button, left);
    clickedAt = performance.now(); // chained calls from the same click keep the spinner
    if (!left) button.classList.remove('busy');
  };
}

let config = null;
let lastStatus = null;
let socket = null;
let socketRetry = 1500;
let logFilter = 'all';
let configPath = '';
let lastEventTs = 0;
let pollTimer = null;
let fleetStatus = null;

/* ------------------------------------------------------------------ helpers */
async function api(path, options = {}) {
  const done = trackBusy();
  try {
    return await request(path, options);
  } finally {
    done();
  }
}

async function request(path, options) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  let data = null;
  const text = await response.text();
  if (text) {
    try { data = JSON.parse(text); } catch (err) { data = { detail: text }; }
  }
  if (!response.ok) {
    const detail = data && data.detail !== undefined ? data.detail : response.statusText;
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return data ?? {};
}

function fmt(value) {
  return value === null || value === undefined ? '–' : String(value);
}

function num(value, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '–';
}

function fmtUptime(seconds) {
  if (!seconds) return '0s';
  const s = Math.floor(seconds % 60), m = Math.floor((seconds / 60) % 60), h = Math.floor(seconds / 3600);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${s}s`;
  return `${s}s`;
}

/* ---------------------------------------------------------------- log panel */
function appendLog(entry) {
  if (typeof entry.ts === 'number' && entry.ts > lastEventTs) lastEventTs = entry.ts;
  const log = $('log');
  const line = document.createElement('div');
  line.className = `log-line ${entry.level || 'info'}`;
  line.dataset.level = entry.level || 'info';
  const time = document.createElement('span');
  time.className = 't';
  time.textContent = entry.time || '';
  const source = document.createElement('span');
  source.className = 's';
  source.textContent = (entry.source || 'app').slice(0, 8);
  const message = document.createElement('span');
  message.className = 'm';
  message.textContent = entry.message || '';
  line.append(time, source, message);
  log.appendChild(line);
  while (log.childElementCount > MAX_LOG_LINES) log.removeChild(log.firstChild);
  applyLogFilter();
  if ($('log-autoscroll').checked) log.scrollTop = log.scrollHeight;
  $('log-count').textContent = `${log.childElementCount} event${log.childElementCount === 1 ? '' : 's'}`;
  // feedback for what *you* did in the panel also pops up as a toast
  if (entry.source === 'panel' && ['success', 'warn', 'error'].includes(entry.level)) {
    ui.toast(entry.message, entry.level);
  }
}

function applyLogFilter() {
  for (const line of $('log').children) {
    const level = line.dataset.level;
    let show = true;
    if (logFilter === 'chat') show = level === 'chat' || level === 'bot';
    else if (logFilter === 'ai') show = level === 'ai' || level === 'chat' || level === 'bot';
    else if (logFilter === 'warn') show = level === 'warn' || level === 'error';
    line.style.display = show ? '' : 'none';
  }
}

/* ------------------------------------------------------------------ render */
let previousState = null;
let previousHealth = null;

function announceState(state, bot) {
  if (previousState === null) { previousState = state; return; }
  if (state === previousState) return;
  const where = bot.server && bot.server.host ? `${bot.server.host}:${bot.server.port}` : 'the server';
  if (state === 'connected') {
    ui.toast(`${bot.username || bot.server?.username || 'The bot'} joined ${where}`, 'success', { icon: '🟢', title: 'Connected' });
    ui.confetti($('status-pill'), undefined, { count: 70, power: 7 });
  } else if (state === 'error') {
    ui.toast(bot.error || bot.last_error || `Could not stay on ${where}`, 'error', { title: 'Connection error' });
    ui.hit($('status-pill'));
  } else if (previousState === 'connected' && (state === 'disconnected' || state === 'stopped')) {
    ui.toast(`Left ${where}`, 'warn', { icon: '⏏', title: 'Disconnected' });
  }
  previousState = state;
}

function renderHero(status) {
  const bot = status.bot || {};
  const agent = status.agent || {};
  const stats = status.stats || {};
  const state = bot.state || 'disconnected';
  ui.flashText('hero-state', state);
  $('hero-state').style.color = state === 'connected' ? 'var(--accent)' : state === 'error' ? 'var(--error)'
    : state === 'connecting' || state === 'waiting' ? 'var(--warn)' : '';
  $('hero-server').textContent = bot.server && bot.server.host
    ? `${bot.server.host}:${bot.server.port} · ${bot.backend || 'auto'}` : 'not connected';
  const brain = agent.resolved_brain || agent.brain || 'auto';
  ui.flashText('hero-brain', brain);
  $('hero-brain-sub').textContent = agent.running ? 'AI loop running' : 'AI loop idle';
  ui.countTo('hero-decisions', agent.decisions || 0);
  const last = agent.last_decision;
  $('hero-last').textContent = last ? `last: ${last.action}${last.source ? ` (${last.source})` : ''}` : 'no decision yet';
  ui.countTo('hero-walked', stats.distance_walked || 0, { suffix: ' m' });
  $('hero-mined').textContent = `${stats.blocks_mined || 0} blocks mined · ${stats.chats_sent || 0} chats`;
  const fleet = status.fleet;
  if (fleet) {
    ui.countTo('hero-fleet', fleet.connected || 0);
    $('hero-fleet-sub').textContent = `of ${fleet.size} bot(s) · max ${fleet.max_bots}`;
  }
  const training = status.training || {};
  if (training.active_run !== undefined || training.running !== undefined) {
    ui.flashText('hero-model', training.running ? 'training…' : (training.active_run || '—'));
    $('hero-model-sub').textContent = training.running
      ? `step ${training.step ?? '?'}${training.total_steps ? ` / ${training.total_steps}` : ''}`
      : `${training.models ?? 0} checkpoint(s) · brain ${training.brain || 'auto'}`;
  }
}

function renderStatus(status) {
  lastStatus = status;
  const bot = status.bot || {};
  const state = bot.state || 'disconnected';
  announceState(state, bot);
  renderHero(status);

  const pill = $('status-pill');
  pill.textContent = state;
  pill.className = 'pill ' + (
    state === 'connected' ? 'pill-on' :
    state === 'connecting' || state === 'waiting' ? 'pill-connecting' :
    state === 'error' ? 'pill-error' : 'pill-off'
  );

  const busy = state === 'connected' || state === 'connecting' || state === 'waiting';
  $('btn-connect').disabled = busy;
  $('btn-disconnect').disabled = !busy;
  $('btn-reconnect').disabled = false;

  $('world-backend').textContent = bot.backend ? `backend: ${bot.backend}` : '';
  $('footer-conn').textContent =
    bot.server && bot.server.host ? `${bot.server.host}:${bot.server.port} as ${bot.server.username || '?'}` : '';

  const health = typeof bot.health === 'number' ? bot.health : null;
  const food = typeof bot.food === 'number' ? bot.food : null;
  $('bar-health').style.width = health === null ? '0%' : `${Math.max(0, Math.min(100, (health / 20) * 100))}%`;
  $('bar-food').style.width = food === null ? '0%' : `${Math.max(0, Math.min(100, (food / 20) * 100))}%`;
  $('val-health').textContent = health === null ? '–' : `${health.toFixed(1)} / 20`;
  $('val-food').textContent = food === null ? '–' : `${food.toFixed(1)} / 20`;
  const healthBar = $('bar-health').parentElement;
  if (health !== null && previousHealth !== null && health < previousHealth - 0.01) ui.hit(healthBar);
  healthBar.classList.toggle('low', health !== null && health <= 6);
  $('bar-food').parentElement.classList.toggle('low', food !== null && food <= 4);
  previousHealth = health;

  const pos = bot.position;
  $('val-pos').textContent = pos ? `${num(pos.x)} ${num(pos.y)} ${num(pos.z)}` : '–';
  const yawDeg = typeof bot.yaw === 'number' ? (bot.yaw * 180 / Math.PI).toFixed(0) : '–';
  const pitchDeg = typeof bot.pitch === 'number' ? (bot.pitch * 180 / Math.PI).toFixed(0) : '–';
  $('val-yaw').textContent = `${yawDeg}° / ${pitchDeg}°`;
  ui.flashText('val-dim', fmt(bot.dimension));
  $('val-time').textContent = fmt(bot.time_of_day);
  $('val-uptime').textContent = fmtUptime(bot.uptime);

  const stats = status.stats || {};
  ui.countTo('val-mined', stats.blocks_mined ?? 0);
  ui.countTo('val-walked', stats.distance_walked ?? 0, { suffix: ' m' });
  ui.countTo('val-chats', stats.chats_sent ?? 0);

  // players
  const list = $('player-list');
  list.innerHTML = '';
  const players = bot.players || [];
  if (!players.length) {
    list.innerHTML = '<li class="empty">nobody in range</li>';
  } else {
    for (const player of players) {
      const li = document.createElement('li');
      li.innerHTML = `<span>👤 ${esc(player.name)}</span><span>${num(player.distance)} m &middot; ${num(player.x, 0)},${num(player.z, 0)}</span>`;
      list.appendChild(li);
    }
  }

  // inventory: hotbar (click to select), armour / off-hand, backpack
  renderInventory(bot);

  // ai status
  const agent = status.agent || {};
  const last = agent.last_decision;
  const thinker = agent.resolved_brain && agent.resolved_brain !== agent.brain
    ? `${agent.brain} → ${agent.resolved_brain}` : (agent.brain || 'auto');
  $('ai-status').textContent = `brain: ${thinker} · ` + (agent.running
    ? `running · ${agent.decisions} decisions` + (last ? ` · last: ${last.action} ${JSON.stringify(last.params || {})}${last.source ? ` (${last.source})` : ''}` : '')
      + (last && last.thought ? ` · thinks: “${last.thought}”` : '') + (agent.memory && agent.memory.goal ? ` · goal: ${agent.memory.goal}` : '')
    : `idle · ${agent.decisions} decisions` + (agent.last_error ? ` · last error: ${agent.last_error}` : ''));
  $('btn-ai-start').disabled = agent.running || state !== 'connected';
  $('btn-ai-stop').disabled = !agent.running;
  $('btn-ai-step').disabled = state !== 'connected';

  // backend availability note
  const backends = status.backends || {};
  const node = backends.node || {};
  const mcRange = backends.minecraft_versions;
  $('backend-note').textContent = node.available
    ? `mineflayer bridge ready (node ${backends.node_version || '?'})`
      + (mcRange ? ` · joins Minecraft ${mcRange.oldest} – ${mcRange.latest}; newer servers need ViaBackwards.` : '.')
    : `${node.reason || 'Node bridge unavailable'} — the simulated world will be used.`;
  fillVersionList(mcRange);
  renderHosting(status.hosting);

  setMinimapTarget(bot);
}

/* Tips for Aternos & co plus the leave-when-alone countdown. Text only changes when it differs. */
function renderHosting(hosting) {
  const note = $('hosting-note');
  if (!hosting) return;
  const parts = [...(hosting.tips || [])];
  if (hosting.leave_when_alone && hosting.alone_for > 0) {
    const left = Math.max(0, Math.round(hosting.alone_minutes * 60 - hosting.alone_for));
    parts.push(`Nobody else is online - the bots leave in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}.`);
  }
  const text = parts.join(' ');
  if (note.textContent !== text) note.textContent = text;
}

/* Version suggestions follow the installed mineflayer (newest first) instead of a fixed list. */
let versionListKey = '';
function fillVersionList(range) {
  if (!range || !Array.isArray(range.tested) || !range.tested.length) return;
  const key = range.tested.join(',');
  if (key === versionListKey) return;
  versionListKey = key;
  const list = $('versions');
  list.replaceChildren(...['auto', ...range.tested.slice().reverse()].map((value) => {
    const option = document.createElement('option');
    option.value = value;
    return option;
  }));
}

/* The minimap eases between status frames (positions arrive every ~1 s) and
 * draws a radar sweep plus a pulse around the bot. Colours follow the theme. */
const minimap = { target: null, view: null, raf: 0, last: 0, colors: null };

function mapColors() {
  return {
    bg: ui.cssVar('--map-bg', '#080c11'),
    grid: ui.cssVar('--map-grid', '#16202b'),
    faint: ui.cssVar('--faint', '#3b4a5c'),
    muted: ui.cssVar('--muted', '#64748b'),
    accent: ui.cssVar('--accent', '#34d399'),
    chat: ui.cssVar('--chat', '#60a5fa'),
    error: ui.cssVar('--error', '#f87171'),
  };
}
window.addEventListener('fx:theme', () => { minimap.colors = null; drawMinimap(performance.now()); });
window.addEventListener('fx:motion', () => scheduleMinimap());

function lerpAngle(from, to, k) {
  let delta = (to - from) % (Math.PI * 2);
  if (delta > Math.PI) delta -= Math.PI * 2;
  if (delta < -Math.PI) delta += Math.PI * 2;
  return from + delta * k;
}

function setMinimapTarget(bot) {
  minimap.target = bot;
  const pos = bot.position;
  if (!pos) { minimap.view = null; }
  else if (!minimap.view || !ui.motionOn()) {
    minimap.view = { x: pos.x, z: pos.z, yaw: bot.yaw || 0 };
  }
  scheduleMinimap();
}

function scheduleMinimap() {
  if (minimap.raf) return;
  if (!ui.motionOn()) { drawMinimap(performance.now()); return; }
  const loop = (now) => {
    minimap.raf = 0;
    if (document.hidden || !ui.motionOn()) return;
    if (now - minimap.last >= 33) { minimap.last = now; drawMinimap(now); }
    minimap.raf = requestAnimationFrame(loop);
  };
  minimap.raf = requestAnimationFrame(loop);
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) scheduleMinimap(); });

function drawMinimap(now) {
  const canvas = $('minimap');
  const bot = minimap.target || {};
  const colors = minimap.colors || (minimap.colors = mapColors());
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 240;
  if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const cx = width / 2, cy = height / 2;
  const scale = Math.min(width, height) / 80; // ~80 blocks across
  const pos = bot.position;
  const view = minimap.view;
  if (pos && view && ui.motionOn()) {
    view.x += (pos.x - view.x) * 0.18;
    view.z += (pos.z - view.z) * 0.18;
    view.yaw = lerpAngle(view.yaw, typeof bot.yaw === 'number' ? bot.yaw : view.yaw, 0.2);
  }
  const ox = view ? view.x : 0, oz = view ? view.z : 0;

  // world grid scrolls with the bot
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  const step = 8 * scale;
  const shiftX = ((-ox * scale) % step + step) % step;
  const shiftY = ((oz * scale) % step + step) % step;
  for (let x = shiftX + (cx % step); x < width; x += step) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke(); }
  for (let y = shiftY + (cy % step); y < height; y += step) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke(); }

  // range rings
  ctx.strokeStyle = colors.grid;
  for (const r of [16, 32]) { ctx.beginPath(); ctx.arc(cx, cy, r * scale, 0, Math.PI * 2); ctx.stroke(); }

  ctx.fillStyle = colors.faint;
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText('+X', width - 26, cy - 6);
  ctx.fillText('+Z', cx + 6, 14);

  if (!pos || !view) {
    ctx.fillStyle = colors.muted;
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText('not connected — no world data', 12, 20);
    return;
  }

  // radar sweep
  if (ui.motionOn()) {
    const angle = (now / 1600) % (Math.PI * 2);
    const radius = Math.hypot(width, height) / 2;
    if (ctx.createConicGradient) {
      const sweep = ctx.createConicGradient(angle, cx, cy);
      sweep.addColorStop(0, `${colors.accent}55`);
      sweep.addColorStop(0.08, `${colors.accent}10`);
      sweep.addColorStop(0.2, 'transparent');
      sweep.addColorStop(1, 'transparent');
      ctx.fillStyle = sweep;
      ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.fill();
    }
  }

  const toScreen = (x, z) => [cx + (x - ox) * scale, cy - (z - oz) * scale];
  for (const player of bot.players || []) {
    const [px, py] = toScreen(player.x, player.z);
    ctx.fillStyle = colors.chat;
    ctx.shadowColor = colors.chat; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(String(player.name || ''), px + 7, py + 4);
  }

  // pulse ring around the bot
  if (ui.motionOn()) {
    const phase = (now % 1800) / 1800;
    ctx.strokeStyle = colors.accent;
    ctx.globalAlpha = 1 - phase;
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, 6 + phase * 22, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
  }

  // the bot: triangle pointing along yaw (yaw 0 => +Z, which is up on screen)
  const yaw = view.yaw;
  const heading = [-Math.sin(yaw), Math.cos(yaw)];
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(Math.atan2(heading[0], heading[1]));
  ctx.fillStyle = colors.accent;
  ctx.shadowColor = colors.accent; ctx.shadowBlur = 12;
  ctx.beginPath();
  ctx.moveTo(0, -10); ctx.lineTo(7, 8); ctx.lineTo(0, 3); ctx.lineTo(-7, 8);
  ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.fillStyle = colors.muted;
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(`${num(pos.x, 0)}, ${num(pos.z, 0)}`, 10, height - 10);
}

/* ------------------------------------------------------------------- config */
function fillForm(cfg) {
  const mc = cfg.minecraft || {}, ol = cfg.ollama || {}, ag = cfg.agent || {}, cx = cfg.cortex || {};
  const set = (id, value) => { const el = $(id); if (el && document.activeElement !== el) el.value = value; };
  const check = (id, value) => { const el = $(id); if (el && document.activeElement !== el) el.checked = Boolean(value); };

  set('cfg-host', mc.host || '');
  set('cfg-port', mc.port || 25565);
  set('cfg-username', mc.username || '');
  set('cfg-version', mc.version || 'auto');
  set('cfg-auth', mc.auth || 'offline');
  set('cfg-backend', mc.backend || 'auto');
  set('cfg-view', mc.view_distance || 'normal');
  set('cfg-timeout', mc.connect_timeout || 20);
  set('cfg-wait', mc.wait_for_server ? 'on' : 'off');
  set('cfg-wait-min', mc.wait_minutes || 10);
  set('cfg-leave', mc.leave_when_alone || 'auto');
  set('cfg-alone-min', mc.alone_minutes || 5);

  check('cfg-ollama-enabled', ol.enabled);
  set('cfg-ollama-url', ol.base_url || 'http://127.0.0.1:11434');
  set('cfg-ollama-model', ol.model || 'llama3.2');
  set('cfg-ollama-interval', ol.decision_interval || 6);
  set('cfg-ollama-temp', ol.temperature ?? 0.7);

  check('cfg-cortex-enabled', cx.enabled);
  set('cfg-cortex-url', cx.base_url || 'http://127.0.0.1:8624');
  set('cfg-cortex-path', cx.api_path ?? '/v1');
  set('cfg-cortex-model', cx.model || '');
  set('cfg-cortex-key-env', cx.api_key_env ?? 'CORTEX_API_KEY');
  set('cfg-cortex-interval', cx.decision_interval || 6);
  set('cfg-cortex-temp', cx.temperature ?? 0.7);
  set('cfg-cortex-max-tokens', cx.max_tokens || 320);
  check('cfg-cortex-json', cx.json_mode ?? true);

  set('cfg-agent-mode', ag.mode || 'auto');
  check('cfg-allow-movement', ag.allow_movement);
  check('cfg-allow-chat', ag.allow_chat);
  check('cfg-allow-mining', ag.allow_mining);
  check('cfg-allow-attack', ag.allow_attacking);
  check('cfg-allow-inventory', ag.allow_inventory !== false);
  check('cfg-allow-building', Boolean(ag.allow_building));
  check('cfg-allow-mob-fighting', ag.allow_mob_fighting !== false);
  check('cfg-greet', ag.greet_players);
  selectBehaviour(ag.persona || 'friendly');
  set('cfg-persona-notes', ag.persona_notes || '');
  check('cfg-human-chat', ag.human_chat ?? true);
  set('cfg-chat-cooldown', ag.chat_cooldown ?? 6);
  set('cfg-memory-turns', ag.memory_turns ?? 8);

  $('raw-config').value = JSON.stringify(cfg, null, 2);
  $('config-path').textContent = configPath;
  applyAuthLabels();

  const afk = cfg.antiafk || {};
  check('antiafk-enabled', afk.enabled !== false);
  set('antiafk-min', afk.interval_min ?? 8);
  set('antiafk-max', afk.interval_max ?? 25);
  check('antiafk-look', afk.look !== false);
  check('antiafk-walk', afk.walk !== false);
  check('antiafk-jump', afk.jump !== false);
  check('antiafk-sneak', afk.sneak !== false);
  check('antiafk-swing', afk.swing !== false);
  check('antiafk-hotbar', afk.hotbar !== false);

  const tr = cfg.training || {};
  set('train-dataset', tr.dataset || 'pymcbot/dataset.jsonl');
  set('train-engine', tr.engine || 'mlp');
  set('train-steps', tr.steps ?? 1000);
  set('train-batch', tr.batch_size ?? 64);
  set('train-lr', tr.lr ?? 0.003);
  set('train-checkpoint', tr.checkpoint_every ?? 200);
  check('train-advanced', tr.advanced);
  set('train-entity-slots', tr.entity_slots ?? 24);
  set('train-item-slots', tr.item_slots ?? 32);

  const fleet = cfg.fleet || {};
  set('fleet-count-input', fleet.count ?? 5);
  set('fleet-pattern', fleet.name_pattern || 'PyMC_Bot_{n}');
  set('fleet-auth', fleet.auth || 'offline');
  set('fleet-ai', fleet.ai_mode || 'heuristic');
  set('fleet-stagger', fleet.stagger_seconds ?? 1.5);
  set('fleet-max', fleet.max_bots ?? 25);
  check('fleet-chatter', fleet.chatter);
  check('fleet-restore', fleet.restore_on_start);
}

/* With Microsoft auth the username field is the account email, not a game name. */
function applyAuthLabels() {
  const premium = $('cfg-auth').value === 'microsoft';
  const label = $('username-field');
  label.childNodes[0].nodeValue = premium ? 'Microsoft email ' : 'Username ';
  $('cfg-username').placeholder = premium ? 'player@example.com' : 'PyMC_Bot';
  $('auth-note').textContent = premium
    ? 'Premium login: enable the AI brain below or keep it off. The first join prints a device code in the event stream - open the link and enter it; the token is then cached per account.'
    : 'Offline (cracked) servers accept any username of 1-16 characters (letters, digits, _). No account needed.';
}

let modelList = [];
let selectedModel = '';

function collectForm() {
  return {
    minecraft: {
      host: $('cfg-host').value.trim(),
      port: Number($('cfg-port').value) || 25565,
      username: $('cfg-username').value.trim() || 'PyMC_Bot',
      version: $('cfg-version').value.trim() || 'auto',
      auth: $('cfg-auth').value,
      backend: $('cfg-backend').value,
      view_distance: $('cfg-view').value,
      connect_timeout: Number($('cfg-timeout').value) || 20,
      wait_for_server: $('cfg-wait').value === 'on',
      wait_minutes: Number($('cfg-wait-min').value) || 10,
      leave_when_alone: $('cfg-leave').value,
      alone_minutes: Number($('cfg-alone-min').value) || 5,
    },
    ollama: {
      enabled: $('cfg-ollama-enabled').checked,
      base_url: $('cfg-ollama-url').value.trim(),
      model: $('cfg-ollama-model').value.trim() || 'llama3.2',
      decision_interval: Number($('cfg-ollama-interval').value) || 6,
      temperature: Number($('cfg-ollama-temp').value ?? 0.7),
    },
    cortex: {
      enabled: $('cfg-cortex-enabled').checked,
      base_url: $('cfg-cortex-url').value.trim() || 'http://127.0.0.1:8624',
      api_path: $('cfg-cortex-path').value.trim(),
      model: $('cfg-cortex-model').value.trim(),
      api_key_env: $('cfg-cortex-key-env').value.trim(),
      decision_interval: Number($('cfg-cortex-interval').value) || 6,
      temperature: Number($('cfg-cortex-temp').value ?? 0.7),
      max_tokens: Number($('cfg-cortex-max-tokens').value) || 320,
      json_mode: $('cfg-cortex-json').checked,
    },
    agent: {
      mode: $('cfg-agent-mode').value,
      allow_movement: $('cfg-allow-movement').checked,
      allow_chat: $('cfg-allow-chat').checked,
      allow_mining: $('cfg-allow-mining').checked,
      allow_attacking: $('cfg-allow-attack').checked,
      allow_inventory: $('cfg-allow-inventory').checked,
      allow_building: $('cfg-allow-building').checked,
      allow_mob_fighting: $('cfg-allow-mob-fighting').checked,
      greet_players: $('cfg-greet').checked,
      persona: selectedBehaviour(),
      persona_notes: $('cfg-persona-notes').value.slice(0, 600),
      human_chat: $('cfg-human-chat').checked,
      chat_cooldown: Math.max(0, Number($('cfg-chat-cooldown').value) || 0),
      memory_turns: Math.min(30, Math.max(1, Number($('cfg-memory-turns').value) || 8)),
    },
    antiafk: {
      enabled: $('antiafk-enabled').checked,
      interval_min: Number($('antiafk-min').value) || 8,
      interval_max: Number($('antiafk-max').value) || 25,
      look: $('antiafk-look').checked,
      look_at_players: $('antiafk-look').checked,
      walk: $('antiafk-walk').checked,
      jump: $('antiafk-jump').checked,
      sneak: $('antiafk-sneak').checked,
      swing: $('antiafk-swing').checked,
      hotbar: $('antiafk-hotbar').checked,
    },
    training: {
      dataset: $('train-dataset').value.trim() || 'pymcbot/dataset.jsonl',
      engine: $('train-engine').value,
      steps: Number($('train-steps').value) || 1000,
      batch_size: Number($('train-batch').value) || 64,
      lr: Number($('train-lr').value) || 0.003,
      checkpoint_every: Number($('train-checkpoint').value) || 200,
      advanced: $('train-advanced').checked,
      entity_slots: Number($('train-entity-slots').value) || 0,
      item_slots: Number($('train-item-slots').value) || 0,
    },
    fleet: {
      count: Number($('fleet-count-input').value) || 5,
      name_pattern: $('fleet-pattern').value.trim() || 'PyMC_Bot_{n}',
      auth: $('fleet-auth').value,
      ai_mode: $('fleet-ai').value,
      stagger_seconds: Number($('fleet-stagger').value) || 0,
      max_bots: Number($('fleet-max').value) || 25,
      chatter: $('fleet-chatter').checked,
      restore_on_start: $('fleet-restore').checked,
    },
  };
}

async function loadConfig() {
  const data = await api('/api/config');
  config = data.config;
  configPath = data.path || '';
  fillForm(config);
  $('config-path').textContent = configPath;
}

async function saveConfig(silent = false) {
  const data = await api('/api/config', { method: 'PUT', body: JSON.stringify(collectForm()) });
  config = data.config;
  fillForm(config);
  if (!silent) appendLog({ level: 'success', source: 'panel', message: 'Settings saved.', time: nowTime() });
  return data;
}

function nowTime() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

/* ---------------------------------------------------------------------- ws */
/* The panel prefers the /ws event stream, but some proxies refuse WebSocket
 * upgrades; in that case we fall back to polling the REST endpoints so the UI
 * keeps updating (slower, never frozen). */
function startPolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    try {
      const [status, logs] = await Promise.all([api('/api/status'), api('/api/logs?limit=80')]);
      renderStatus(status);
      renderFleet(status);
      renderAntiAfk(status.antiafk);
      renderTraining({ status: { ...(status.training || {}), running: status.training?.running }, brain: status.training?.brain, settings: { dataset: status.training?.dataset } });
      if (status.training?.running) {
        const detail = await api('/api/training/status');
        renderTraining(detail);
      }
      for (const entry of logs.events || []) {
        if ((entry.ts || 0) > lastEventTs) appendLog(entry);
      }
    } catch (err) { /* keep polling */ }
  }, 2500);
}

function stopPolling() {
  if (pollTimer) clearInterval(pollTimer);
  pollTimer = null;
}

function connectSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${proto}//${location.host}/ws`);

  socket.onopen = () => { socketRetry = 1500; stopPolling(); };
  socket.onmessage = (message) => {
    let frame;
    try { frame = JSON.parse(message.data); } catch (err) { return; }
    if (frame.type === 'event') appendLog(frame.data);
    else if (frame.type === 'status') {
      renderStatus(frame.data);
      renderFleet(frame.data);
      renderAntiAfk(frame.data.antiafk);
      if (frame.data.ollama) updateOllamaPill(frame.data.ollama);
      if (frame.data.cortex) updateCortexPill(frame.data.cortex);
      if (frame.data.training) refreshTraining();
    }
  };
  socket.onclose = () => {
    startPolling();
    setTimeout(connectSocket, socketRetry);
    socketRetry = Math.min(socketRetry * 1.6, 15000);
  };
  socket.onerror = () => { try { socket.close(); } catch (err) { /* ignore */ } };
}

function updateOllamaPill(health) {
  const pill = $('ollama-health');
  const detail = String(health.detail || '');
  if (health.ok) {
    pill.textContent = 'ollama up';
    pill.className = 'pill pill-on';
    if (/stub/i.test(detail)) $('stub-banner').classList.remove('hidden');
  } else if (health.ok === false) {
    pill.textContent = 'ollama offline';
    pill.className = 'pill pill-error';
    pill.title = detail;
  } else {
    pill.textContent = 'checking…';
    pill.className = 'pill pill-unknown';
  }
}

function updateCortexPill(health) {
  const pill = $('cortex-health');
  const detail = String(health.detail || '');
  const inUse = health.in_use ?? health.enabled;
  pill.title = `Cortex LLMHoster: ${detail || 'not checked yet'}`;
  if (health.ok) {
    pill.textContent = 'cortex up';
    pill.className = 'pill pill-on';
  } else if (health.ok === false) {
    // Only shout when the user actually turned Cortex on; otherwise just say it is off.
    pill.textContent = inUse ? 'cortex offline' : 'cortex off';
    pill.className = inUse ? 'pill pill-error' : 'pill pill-unknown';
  } else {
    pill.textContent = 'cortex…';
    pill.className = 'pill pill-unknown';
  }
  const keyNote = health.key_env
    ? (health.key_set ? `key from $${health.key_env}` : `$${health.key_env} not set (fine if Cortex has no key)`)
    : 'no key variable';
  if (!health.ok && !inUse) {
    // Optional service that is not selected: say so calmly instead of printing a connection error.
    $('cortex-summary').textContent = '— not running (optional: only needed for the Cortex brain)';
    return;
  }
  $('cortex-summary').textContent = `— ${health.ok ? 'ready' : 'unavailable'}: ${detail || '…'} · ${keyNote}`;
}

// ---------------------------------------------------------------- behaviours
let behaviourChoice = 'friendly';

function selectedBehaviour() {
  const checked = document.querySelector('input[name="behaviour"]:checked');
  return checked ? checked.value : behaviourChoice;
}

function selectBehaviour(key) {
  behaviourChoice = key || 'friendly';
  for (const input of document.querySelectorAll('input[name="behaviour"]')) {
    input.checked = input.value === behaviourChoice;
  }
  const picked = document.querySelector(`input[name="behaviour"][value="${behaviourChoice}"]`);
  const label = picked ? picked.dataset.label : behaviourChoice;
  $('mind-summary').textContent = `— ${label}`;
}

async function loadBehaviours() {
  try {
    const data = await request('/api/ai/behaviours');
    const grid = $('behaviour-grid');
    grid.innerHTML = '';
    for (const item of data.behaviours || []) {
      const card = document.createElement('label');
      card.className = 'behaviour';
      const input = document.createElement('input');
      input.type = 'radio';
      input.name = 'behaviour';
      input.value = item.key;
      input.dataset.label = item.label;
      input.addEventListener('change', async () => {
        selectBehaviour(item.key);
        try { await saveConfig(true); } catch (err) { reportError(err); }
      });
      const emoji = document.createElement('span');
      emoji.className = 'behaviour-emoji';
      emoji.setAttribute('aria-hidden', 'true');
      emoji.textContent = item.emoji || '🙂';
      const text = document.createElement('span');
      text.className = 'behaviour-text';
      const name = document.createElement('strong');
      name.textContent = item.label;
      const blurb = document.createElement('small');
      blurb.textContent = item.blurb || '';
      text.append(name, blurb);
      card.append(input, emoji, text);
      grid.appendChild(card);
    }
    selectBehaviour(data.selected || behaviourChoice);
  } catch (err) { reportError(err); }
}

async function checkCortex() {
  try {
    await saveConfig(true);
    const health = await api('/api/cortex/health?force=true');
    updateCortexPill(health);
    appendLog({ level: health.ok ? 'success' : 'warn', source: 'panel', time: nowTime(), message: `Cortex: ${health.detail}` });
    if (!health.ok) return;
    const data = await api('/api/cortex/models');
    const datalist = $('cortex-models');
    datalist.innerHTML = '';
    for (const entry of data.details || []) {
      const option = document.createElement('option');
      option.value = entry.id;
      option.label = `${entry.runtime || ''} ${entry.chat ? '' : '(no text_generation)'}`.trim();
      datalist.appendChild(option);
    }
  } catch (err) { reportError(err); }
}

async function useCortex() {
  $('cfg-agent-mode').value = 'cortex';
  try {
    await saveConfig(true);
    appendLog({ level: 'success', source: 'panel', time: nowTime(), message: 'Brain switched to Cortex LLMHoster.' });
    await checkCortex();
  } catch (err) { reportError(err); }
}

/* -------------------------------------------------------------------- fleet */
const STATE_CLASS = {
  connected: 'pill-on',
  connecting: 'pill-connecting',
  error: 'pill-error',
  disconnected: 'pill-off',
  stopped: 'pill-off',
  queued: 'pill-unknown',
};

let trainingWasRunning = false;

function renderTraining(training) {
  if (!training) return;
  const status = training.status || {};
  const dataset = training.dataset || {};
  const el = $('train-state');
  const running = Boolean(status.running);
  el.textContent = running ? `training step ${status.step || 0}/${status.total_steps || '?'}` : 'idle';
  el.className = 'pill ' + (running ? 'pill-on' : 'pill-unknown');
  const bar = $('train-progress-bar');
  const progress = status.progress ?? (status.total_steps ? (status.step || 0) / status.total_steps : 0);
  bar.style.width = `${Math.round(Math.min(1, Math.max(0, progress || 0)) * 100)}%`;
  $('train-progress').classList.toggle('running', running);
  if (trainingWasRunning && !running) {
    if (status.error) ui.toast(`Training stopped: ${status.error}`, 'error', { title: 'Training' });
    else {
      ui.toast(`Run '${status.run || training.active_run || 'model'}' finished${status.val_acc ? ` · val acc ${status.val_acc}` : ''}`,
        'success', { icon: '🎓', title: 'Training complete' });
      ui.confetti($('train-progress'), undefined, { count: 120 });
    }
    refreshTrainedModels();
  }
  trainingWasRunning = running;
  ui.flashText('hero-model', running ? 'training…' : (training.active_run || status.run || '—'));
  if (running) $('hero-model-sub').textContent = `step ${status.step || 0} / ${status.total_steps || '?'}`;

  const bits = [];
  if (dataset.exists) {
    bits.push(`${dataset.samples ?? 0} samples from ${dataset.episodes ?? 0} episode(s)`);
    const top = Object.entries(dataset.actions || {}).slice(0, 4).map(([name, count]) => `${name} ${count}`);
    if (top.length) bits.push(`mostly ${top.join(', ')}`);
  } else {
    bits.push(`no dataset at ${dataset.dataset || training.settings?.dataset || 'pymcbot/dataset.jsonl'}`);
  }
  if (running || status.summary) {
    bits.push(`loss ${status.loss ?? '—'} · val ${status.val_loss ?? '—'} · acc ${status.val_acc ?? '—'}`);
  }
  if (status.advanced) bits.push('advanced mode (entities + items)');
  if (status.error) bits.push(`error: ${status.error}`);
  if (status.run) bits.push(`run '${status.run}'`);
  bits.push(`brain: ${training.brain || 'auto'}${training.active_run ? ` (model '${training.active_run}')` : ''}`);
  $('train-status').textContent = bits.join(' · ');

  // Advanced training: what the dataset offers and what the newest model learned.
  const advanced = training.advanced || {};
  const pieces = [];
  if (dataset.entities_and_items) {
    pieces.push(`dataset has entity + item data`);
    if (dataset.entity_coverage !== undefined) pieces.push(`entity coverage ${(dataset.entity_coverage * 100).toFixed(0)}%`);
    if (dataset.item_coverage !== undefined) pieces.push(`item coverage ${(dataset.item_coverage * 100).toFixed(0)}%`);
  } else if (dataset.exists) {
    pieces.push('dataset has no entity/item data yet — record with /pymcbot advanced on');
  }
  if (advanced.trained && advanced.run) {
    pieces.push(`${advanced.run}: ${advanced.entity_slots || 0} entity word(s), ${advanced.item_slots || 0} item word(s)`);
    if (advanced.entities?.length) pieces.push(`entities: ${advanced.entities.slice(0, 6).join(', ')}`);
    if (advanced.items?.length) pieces.push(`items: ${advanced.items.slice(0, 6).join(', ')}`);
  } else if (dataset.exists) {
    pieces.push('no advanced model trained yet');
  }
  $('train-advanced-status').textContent = pieces.length ? pieces.join(' · ') : 'nothing learned yet';
}

function renderModels(payload) {
  if (!payload) return;
  modelList = payload.models || [];
  if (!selectedModel && payload.active_run) selectedModel = payload.active_run;
  if (!selectedModel && modelList.length) selectedModel = modelList[0].run;
  $('models-dir').textContent = payload.models_dir || 'models/';
  const body = $('model-body');
  body.innerHTML = '';
  if (!modelList.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty">nothing trained yet — press “Train on playtime”</td></tr>';
    return;
  }
  for (const card of modelList) {
    const row = document.createElement('tr');
    const active = card.run === payload.active_run;
    row.className = active ? 'row-active' : '';
    const metrics = card.metrics || {};
    const cells = [
      `${card.run}${active ? ' ✓' : ''}`,
      card.engine || 'mlp',
      card.advanced ? `advanced (${(card.entity_vocabulary || []).length}e/${(card.item_vocabulary || []).length}i)` : 'basic',
      card.step ?? 0,
      card.params ?? 0,
      metrics.val_loss === null || metrics.val_loss === undefined ? '—' : Number(metrics.val_loss).toFixed(4),
      metrics.val_accuracy === null || metrics.val_accuracy === undefined ? '—' : Number(metrics.val_accuracy).toFixed(3),
      `${card.checkpoint_exists ? '' : '⚠ '}${card.checkpoints?.length || 0} file(s)`,
    ];
    for (const value of cells) {
      const td = document.createElement('td');
      td.textContent = String(value);
      row.appendChild(td);
    }
    row.addEventListener('click', () => {
      selectedModel = card.run;
      renderModels({ models: modelList, active_run: payload.active_run, models_dir: payload.models_dir });
    });
    body.appendChild(row);
  }
}

async function refreshTraining() {
  try {
    const data = await api('/api/training/status');
    renderTraining(data);
  } catch (err) { /* keep the panel alive */ }
}

async function refreshTrainedModels() {
  try {
    renderModels(await api('/api/models'));
  } catch (err) { reportError(err); }
}

async function startTraining(extra = {}) {
  try {
    await saveConfig(true);
    const payload = {
      dataset: $('train-dataset').value.trim() || undefined,
      engine: $('train-engine').value,
      steps: Number($('train-steps').value) || 1000,
      batch_size: Number($('train-batch').value) || 64,
      lr: Number($('train-lr').value) || 0.003,
      checkpoint_every: Number($('train-checkpoint').value) || 200,
      advanced: $('train-advanced').checked,
      entity_slots: Number($('train-entity-slots').value) || undefined,
      item_slots: Number($('train-item-slots').value) || undefined,
      ...extra,
    };
    const data = await api('/api/training/start', { method: 'POST', body: JSON.stringify(payload) });
    appendLog({ level: 'success', source: 'panel', time: nowTime(),
      message: `Training '${data.run}' on ${payload.dataset || data.dataset?.dataset || 'the configured dataset'} (self-checkpointing).` });
    renderTraining(data);
    await refreshTrainedModels();
  } catch (err) { reportError(err); }
}

async function activateModel() {
  if (!selectedModel) { reportError(new Error('Pick a checkpoint row first.')); return; }
  try {
    const data = await api('/api/models/activate', { method: 'POST', body: JSON.stringify({ run: selectedModel }) });
    appendLog({ level: 'success', source: 'panel', time: nowTime(),
      message: `Brain = trained (${data.active_run}); the bot now plays with the model trained on playtime.` });
    await loadConfig();
    await refreshTrainedModels();
  } catch (err) { reportError(err); }
}

async function exportModelToOllama() {
  if (!selectedModel) { reportError(new Error('Pick a checkpoint row first.')); return; }
  try {
    const data = await api('/api/models/export-ollama', { method: 'POST', body: JSON.stringify({ run: selectedModel }) });
    appendLog({ level: 'success', source: 'panel', time: nowTime(), message: `Exported to Ollama as '${data.model}' (${Object.entries(data.priors || {}).slice(0, 3).map(([k, v]) => `${k}:${v}`).join(', ')}).` });
    await refreshModels();
  } catch (err) { reportError(err); }
}

async function previewPolicy() {
  try {
    const data = await api('/api/policy/preview');
    const prediction = data.prediction || {};
    const top = Object.entries(prediction.probs || {}).sort((a, b) => b[1] - a[1]).slice(0, 3)
      .map(([name, value]) => `${name} ${(value * 100).toFixed(0)}%`).join(' · ');
    ui.flashText('policy-preview',
      `🔮 model '${data.run}' would do: ${prediction.action} (${top}) — turn ${prediction.motion?.dyaw ?? 0}°, step ${prediction.motion?.forward ?? 0} blocks`);
  } catch (err) { reportError(err); }
}

function renderAntiAfk(antiafk) {
  if (!antiafk) return;
  const el = $('antiafk-state');
  if (!el) return;
  el.textContent = antiafk.enabled
    ? `on · ${antiafk.pokes} bursts so far${antiafk.running ? '' : ' (starting…)'}`
    : 'off';
}

function renderFleet(status) {
  const fleet = (status && status.fleet) || null;
  fleetStatus = fleet;
  if (!fleet) return;

  $('fleet-count').textContent = `${fleet.connected} online / ${fleet.size} of ${fleet.max_bots}`;
  $('fleet-count').className = 'pill ' + (fleet.connected ? 'pill-on' : 'pill-unknown');
  $('world-player').textContent = fleet.selected;

  // one shared player list for the top bar and the fleet card
  for (const select of [$('topbar-select'), $('fleet-select')]) {
    const previous = select.value;
    select.innerHTML = '';
    for (const bot of fleet.bots) {
      const option = document.createElement('option');
      option.value = bot.configured_username;
      option.textContent = `${bot.username}${bot.sponsor === 'primary' ? ' (main)' : ''} · ${bot.state}`;
      if (bot.selected) option.selected = true;
      select.appendChild(option);
    }
    if (previous && !fleet.bots.some((b) => b.configured_username === previous)) select.value = '';
  }

  const body = $('fleet-body');
  body.innerHTML = '';
  for (const bot of fleet.bots) {
    const row = document.createElement('tr');
    if (bot.selected) row.className = 'selected';
    const pos = bot.position ? `${num(bot.position.x, 0)}, ${num(bot.position.z, 0)}` : '–';
    const premium = bot.auth === 'microsoft';
    const idle = bot.state === 'connected'
      ? `${bot.idle_seconds === null || bot.idle_seconds === undefined ? '–' : Math.round(bot.idle_seconds) + 's'}` +
        (bot.antiafk_pokes ? `<div class="mini">afk ×${bot.antiafk_pokes}</div>` : '')
      : '–';
    row.innerHTML = `
      <td><span class="who">${esc(bot.username)}</span>${bot.sponsor === 'primary' ? ' <span class="mini">main</span>' : ''}
        ${bot.error ? `<div class="mini error-text">${esc(bot.error)}</div>` : ''}</td>
      <td>${premium ? '<span class="mini premium">premium</span>' : '<span class="mini">offline</span>'}</td>
      <td><span class="pill ${STATE_CLASS[bot.state] || 'pill-unknown'}">${esc(bot.state)}</span></td>
      <td>${esc(bot.ai)}${bot.ai_running ? ' <span class="ai-dot" title="AI running">●</span>' : ''}</td>
      <td class="idle-cell">${idle}</td>
      <td>${pos}</td>
      <td class="row-actions"></td>`;
    const actions = row.querySelector('.row-actions');
    const add = (label, handler, title) => {
      const button = document.createElement('button');
      button.className = 'btn ghost small';
      button.textContent = label;
      if (title) button.title = title;
      button.addEventListener('click', handler);
      actions.appendChild(button);
    };
    add(bot.selected ? 'active' : 'use', () => selectBot(bot.configured_username).catch(reportError));
    if (bot.sponsor !== 'primary') {
      add('start', () => fleetCall('/api/fleet/start', { bot: bot.configured_username }).catch(reportError));
      add('stop', () => fleetCall('/api/fleet/stop', { bot: bot.configured_username }).catch(reportError));
      add('drop', () => fleetCall('/api/fleet/remove', { bot: bot.configured_username }).catch(reportError),
          'Disconnect and forget this player');
    }
    body.appendChild(row);
  }
  if (!fleet.bots.length) {
    body.innerHTML = '<tr><td colspan="7" class="empty">no players yet</td></tr>';
  }
}

async function fleetCall(path, params, method = 'POST') {
  const result = await api(path, { method, body: JSON.stringify(params || {}) });
  if (result.fleet) renderFleet({ fleet: result.fleet });
  return result;
}

async function selectBot(username) {
  const result = await api('/api/fleet/select', { method: 'POST', body: JSON.stringify({ bot: username }) });
  appendLog({ level: 'info', source: 'panel', message: `Now controlling ${result.selected}`, time: nowTime() });
  renderStatus(await api('/api/status'));
}

/* ------------------------------------------------------------------ actions */
function wire() {
  $('btn-connect').addEventListener('click', async () => {
    try { await saveConfig(true); await api('/api/bot/start', { method: 'POST' }); }
    catch (err) { appendLog({ level: 'error', source: 'panel', message: err.message, time: nowTime() }); }
  });
  $('btn-save-connect').addEventListener('click', async () => {
    try { await saveConfig(true); await api('/api/bot/start', { method: 'POST' }); }
    catch (err) { appendLog({ level: 'error', source: 'panel', message: err.message, time: nowTime() }); }
  });
  $('btn-disconnect').addEventListener('click', () => api('/api/bot/stop', { method: 'POST' }).catch(reportError));
  $('btn-reconnect').addEventListener('click', () => api('/api/bot/reconnect', { method: 'POST' }).catch(reportError));
  $('btn-save').addEventListener('click', () => saveConfig().catch(reportError));
  $('btn-probe').addEventListener('click', () => probeServer().catch(reportError));

  $('btn-ai-start').addEventListener('click', async () => {
    try { await saveConfig(true); await api('/api/ai/start', { method: 'POST' }); }
    catch (err) { reportError(err); }
  });
  $('btn-ai-stop').addEventListener('click', () => api('/api/ai/stop', { method: 'POST' }).catch(reportError));
  $('btn-ai-step').addEventListener('click', () => api('/api/ai/step', { method: 'POST' }).catch(reportError));
  $('btn-refresh-models').addEventListener('click', () => refreshModels());
  $('btn-train-start').addEventListener('click', () => startTraining());
  $('btn-train-stop').addEventListener('click', () =>
    api('/api/training/stop', { method: 'POST' })
      .then(() => appendLog({ level: 'info', source: 'panel', message: 'Stopping training (it saves a checkpoint first).', time: nowTime() }))
      .catch(reportError));
  $('btn-train-resume').addEventListener('click', () => startTraining({ resume: true }));
  $('btn-train-simulate').addEventListener('click', () => startTraining({ simulate_minutes: 5, run_name: '' }));
  $('btn-model-activate').addEventListener('click', activateModel);
  $('btn-model-export').addEventListener('click', exportModelToOllama);
  $('btn-models-refresh').addEventListener('click', refreshTrainedModels);
  $('btn-policy-preview').addEventListener('click', previewPolicy);
  $('btn-pull').addEventListener('click', pullModel);
  $('btn-cortex-check').addEventListener('click', checkCortex);
  loadBehaviours();
  $('btn-cortex-use').addEventListener('click', useCortex);

  $('cfg-auth').addEventListener('change', applyAuthLabels);

  $('btn-populate').addEventListener('click', async () => {
    try {
      await saveConfig(true);
      const payload = {
        count: Number($('fleet-count-input').value) || 5,
        pattern: $('fleet-pattern').value.trim() || 'PyMC_Bot_{n}',
        auth: $('fleet-auth').value,
        ai: $('fleet-ai').value,
        stagger: Number($('fleet-stagger').value) || 0,
      };
      const result = await api('/api/fleet/populate', { method: 'POST', body: JSON.stringify(payload) });
      renderFleet(result);
      appendLog({
        level: 'success', source: 'panel', time: nowTime(),
        message: `Queued ${result.count} player(s): ${result.queued.slice(0, 6).join(', ')}${result.queued.length > 6 ? ' …' : ''}`,
      });
    } catch (err) { reportError(err); }
  });

  $('btn-add-premium').addEventListener('click', async () => {
    const username = $('premium-email').value.trim();
    if (!username) {
      appendLog({ level: 'warn', source: 'panel', message: 'Enter the Microsoft account email first.', time: nowTime() });
      return;
    }
    try {
      const result = await api('/api/fleet/spawn', {
        method: 'POST',
        body: JSON.stringify({ username, auth: 'microsoft', ai: $('premium-ai').value }),
      });
      $('premium-email').value = '';
      appendLog({
        level: 'success', source: 'panel', time: nowTime(),
        message: `Premium player ${username} is joining - the device code will appear in the log.`,
      });
      if (result.member) renderStatus(await api('/api/status'));
    } catch (err) { reportError(err); }
  });

  $('btn-antiafk-poke').addEventListener('click', async () => {
    try {
      const result = await api('/api/antiafk/poke', { method: 'POST', body: JSON.stringify({ bot: 'all' }) });
      const poked = result.results.filter((entry) => entry.poked);
      appendLog({
        level: poked.length ? 'success' : 'warn', source: 'panel', time: nowTime(),
        message: `Anti-AFK poke: ${poked.length}/${result.results.length} bots moved` +
          (poked.length ? ` (${[...new Set(poked.flatMap((entry) => entry.habits))].join(', ')})` : ''),
      });
      renderStatus(await api('/api/status'));
    } catch (err) { reportError(err); }
  });

  $('btn-fleet-start').addEventListener('click', () => fleetCall('/api/fleet/start', {}).catch(reportError));
  $('btn-fleet-stop').addEventListener('click', () => fleetCall('/api/fleet/stop', {}).catch(reportError));
  $('btn-fleet-remove-all').addEventListener('click', () => {
    if (!window.confirm('Disconnect and forget every extra player?')) return;
    fleetCall('/api/fleet/stop', { remove: true })
      .then(() => appendLog({ level: 'success', source: 'panel', message: 'Removed all extra players.', time: nowTime() }))
      .catch(reportError);
  });
  $('btn-fleet-select').addEventListener('click', () => selectBot($('fleet-select').value).catch(reportError));
  $('topbar-select').addEventListener('change', (event) => selectBot(event.target.value).catch(reportError));
  $('btn-fleet-broadcast').addEventListener('click', async () => {
    const message = $('fleet-broadcast-input').value.trim();
    if (!message) return;
    try {
      const result = await api('/api/fleet/broadcast', { method: 'POST', body: JSON.stringify({ message }) });
      $('fleet-broadcast-input').value = '';
      appendLog({ level: 'bot', source: 'panel', time: nowTime(), message: `${result.sent} bot(s) said: ${message}` });
    } catch (err) { reportError(err); }
  });

  $('btn-chat').addEventListener('click', sendChat);
  $('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') sendChat(); });
  $('btn-command').addEventListener('click', runCommand);
  $('command-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') runCommand(); });

  $('hotbar').addEventListener('click', (event) => {
    const cell = event.target.closest('.slot');
    if (cell) runAction({ action: 'select_slot', slot: Number(cell.dataset.slot) }).catch(reportError);
  });
  for (const button of document.querySelectorAll('#quick-actions button, #life-actions button')) {
    button.addEventListener('click', () => {
      const payload = JSON.parse(button.dataset.action);
      runAction(payload).catch(reportError);
    });
  }
  $('btn-action').addEventListener('click', () => {
    const raw = $('action-input').value.trim();
    if (!raw) return;
    runAction({ raw }).catch(reportError);
  });

  $('log-filter').addEventListener('change', (event) => { logFilter = event.target.value; applyLogFilter(); });
  $('btn-clear-logs').addEventListener('click', async () => {
    try { await api('/api/logs/clear', { method: 'POST' }); } catch (err) { /* ignore */ }
    $('log').innerHTML = '';
    $('log-count').textContent = '0 events';
  });

  $('btn-raw-save').addEventListener('click', async () => {
    try {
      const parsed = JSON.parse($('raw-config').value);
      const data = await api('/api/config', { method: 'PUT', body: JSON.stringify(parsed) });
      config = data.config;
      fillForm(config);
      $('raw-config-status').textContent = 'Saved.';
      ui.toast('Raw config saved', 'success');
    } catch (err) {
      $('raw-config-status').textContent = `Not saved: ${err.message}`;
      ui.toast(`Raw config not saved: ${err.message}`, 'error');
    }
  });
  $('btn-raw-reload').addEventListener('click', () => loadConfig().catch(reportError));
  $('btn-reset').addEventListener('click', async () => {
    if (!window.confirm('Reset every setting to its default? Your current config file is overwritten.')) return;
    try {
      const data = await api('/api/config/reset', { method: 'POST' });
      config = data.config;
      fillForm(config);
      $('raw-config-status').textContent = 'Reset to defaults.';
    } catch (err) { reportError(err); }
  });
}

/* "Check server": a server list ping without joining (asleep Aternos servers, versions, players). */
async function probeServer() {
  const note = $('probe-note');
  const host = $('cfg-host').value.trim();
  if (!host) { note.textContent = 'Enter a host first.'; return; }
  note.textContent = `Pinging ${host}…`;
  note.dataset.state = 'busy';
  const port = Number($('cfg-port').value) || 25565;
  const result = await api(`/api/server/probe?host=${encodeURIComponent(host)}&port=${port}`);
  const icon = !result.ok ? '🔴' : result.asleep ? '💤' : '🟢';
  note.textContent = `${icon} ${result.summary}`;
  note.dataset.state = !result.ok ? 'error' : result.asleep ? 'asleep' : 'online';
  ui.toast(result.summary, !result.ok ? 'error' : result.asleep ? 'warn' : 'success',
    { title: !result.ok ? 'Server unreachable' : result.asleep ? 'Server asleep' : 'Server online' });
}

function reportError(err) {
  appendLog({ level: 'error', source: 'panel', message: err.message, time: nowTime() });
}

async function sendChat() {
  const input = $('chat-input');
  const message = input.value.trim();
  if (!message) return;
  input.value = '';
  try { await api('/api/bot/chat', { method: 'POST', body: JSON.stringify({ message }) }); }
  catch (err) { reportError(err); }
}

async function runCommand() {
  const input = $('command-input');
  const command = input.value.trim();
  if (!command) return;
  input.value = '';
  try { await api('/api/bot/command', { method: 'POST', body: JSON.stringify({ command }) }); }
  catch (err) { reportError(err); }
}

const ITEM_ICONS = [
  ['_sword', '⚔️'], ['_pickaxe', '⛏️'], ['_axe', '🪓'], ['_shovel', '🥄'], ['_hoe', '🌾'], ['bow', '🏹'],
  ['shield', '🛡️'], ['_helmet', '⛑️'], ['_chestplate', '🦺'], ['_leggings', '👖'], ['_boots', '🥾'],
  ['torch', '🔥'], ['_log', '🪵'], ['_planks', '🟫'], ['bread', '🍞'], ['apple', '🍎'], ['cooked_', '🍖'],
  ['beef', '🥩'], ['porkchop', '🥩'], ['carrot', '🥕'], ['potato', '🥔'], ['fish', '🐟'], ['cod', '🐟'],
  ['salmon', '🐟'], ['bucket', '🪣'], ['coal', '⚫'], ['diamond', '💎'], ['iron_ingot', '🔩'], ['bed', '🛏️'],
  ['dirt', '🟫'], ['cobblestone', '🪨'], ['stone', '🪨'], ['sand', '🟨'], ['rotten_flesh', '🧟'], ['bone', '🦴'],
];

function itemIcon(name) {
  const hit = ITEM_ICONS.find(([needle]) => name.includes(needle));
  return hit ? hit[1] : '▪️';
}

function itemName(stack) {
  return String((stack && stack.name) || '').replace(/^minecraft:/, '');
}

function renderInventory(bot) {
  const bar = $('hotbar');
  const slots = Array.isArray(bot.slots) ? bot.slots : null;
  const selected = Number.isInteger(bot.selected_slot) ? bot.selected_slot : -1;
  bar.innerHTML = '';
  bar.hidden = !slots;
  if (slots) {
    for (let i = 0; i < 9; i += 1) {
      const stack = slots[i];
      const name = itemName(stack);
      const cell = document.createElement('button');
      cell.type = 'button';
      cell.className = 'slot' + (i === selected ? ' selected' : '') + (stack ? '' : ' empty-slot');
      cell.dataset.slot = String(i + 1);
      cell.title = `${i + 1}: ${name ? name.replace(/_/g, ' ') + (stack.count > 1 ? ` ×${stack.count}` : '') : 'empty'}`;
      cell.innerHTML = `<span class="key">${i + 1}</span>`
        + (stack ? `<span class="ico">${itemIcon(name)}</span>${stack.count > 1 ? `<span class="count">${esc(stack.count)}</span>` : ''}` : '');
      bar.appendChild(cell);
    }
  }
  const gear = $('gear');
  const armor = bot.armor_slots || {};
  const worn = ['head', 'chest', 'legs', 'feet'].map((slot) => armor[slot] && itemName(armor[slot])).filter(Boolean);
  const held = String(bot.held_item || '').replace(/^minecraft:/, '');
  gear.textContent = (slots || held ? `in hand: ${held ? held.replace(/_/g, ' ') : 'nothing'} · ` : '')
    + `armour: ${worn.length ? worn.join(', ').replace(/_/g, ' ') : (bot.armor && bot.armor.length ? bot.armor.join(', ') : 'none')}`
    + ` · off-hand: ${bot.offhand ? String(bot.offhand).replace(/^minecraft:/, '').replace(/_/g, ' ') : 'empty'}`;

  // backpack (slots 9-35) - or every item when the backend only reports a flat list
  const inv = $('inventory');
  inv.innerHTML = '';
  const counts = new Map();
  const source = slots ? slots.slice(9) : (bot.inventory || []);
  for (const stack of source) {
    if (!stack) continue;
    const name = itemName(stack);
    counts.set(name, (counts.get(name) || 0) + Number(stack.count || 1));
  }
  if (!counts.size) {
    inv.innerHTML = `<span class="empty">${slots ? 'backpack empty' : 'empty'}</span>`;
    return;
  }
  for (const [name, count] of counts) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.title = 'in the backpack';
    chip.innerHTML = `${itemIcon(name)} ${esc(name)} <b>×${esc(count)}</b>`;
    inv.appendChild(chip);
  }
}

async function runAction(payload) {
  const result = await api('/api/bot/action', { method: 'POST', body: JSON.stringify(payload) });
  appendLog({ level: result.ok ? 'success' : 'warn', source: 'panel', message: result.detail || 'action run', time: nowTime() });
}

async function refreshFleet() {
  try { renderFleet(await api('/api/fleet/status')); } catch (err) { /* ignore */ }
}

async function refreshModels(quiet = false) {
  try {
    const data = await api('/api/ollama/models');
    const datalist = $('models');
    datalist.innerHTML = '';
    for (const name of data.models || []) {
      const option = document.createElement('option');
      option.value = name;
      datalist.appendChild(option);
    }
    if ((data.models || []).some((name) => /stub/i.test(name))) $('stub-banner').classList.remove('hidden');
    appendLog({ level: 'info', source: 'panel', message: `Ollama models: ${(data.models || []).join(', ') || '(none)'}`, time: nowTime() });
  } catch (err) {
    // at boot an absent Ollama is normal (the pill already says so); only clicks report it loudly
    if (quiet) appendLog({ level: 'debug', source: 'ollama', message: err.message, time: nowTime() });
    else reportError(err);
  }
}

async function pullModel() {
  const model = $('cfg-ollama-model').value.trim();
  if (!model) return;
  try {
    const data = await api('/api/ollama/pull', { method: 'POST', body: JSON.stringify({ model }) });
    appendLog({ level: 'success', source: 'panel', message: `Model ready: ${data.model}`, time: nowTime() });
    await refreshModels();
  } catch (err) { reportError(err); }
}

/* --------------------------------------------------------------------- boot */
async function boot() {
  wire();
  try { await loadConfig(); } catch (err) { reportError(err); }
  try {
    const logs = await api('/api/logs?limit=120');
    for (const entry of logs.events || []) appendLog(entry);
  } catch (err) { reportError(err); }
  try {
    const status = await api('/api/status');
    renderStatus(status);
    renderFleet(status);
    renderAntiAfk(status.antiafk);
  } catch (err) { reportError(err); }
  try { updateOllamaPill(await api('/api/ollama/health')); } catch (err) { /* ignore */ }
  try { updateCortexPill(await api('/api/cortex/health')); } catch (err) { /* ignore */ }
  refreshModels(true);
  await refreshFleet();
  await refreshTrainedModels();
  await refreshTraining();
  connectSocket();
}

document.addEventListener('DOMContentLoaded', boot);
