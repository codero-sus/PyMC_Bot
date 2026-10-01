/* PyMC_Bot control panel -- vanilla JS, no build step. Talks to the FastAPI app
 * on the same origin (relative URLs), so it works behind any proxy/preview host. */
'use strict';

const MAX_LOG_LINES = 500;
const $ = (id) => document.getElementById(id);

let config = null;
let lastStatus = null;
let socket = null;
let socketRetry = 1500;
let logFilter = 'all';
let configPath = '';

/* ------------------------------------------------------------------ helpers */
async function api(path, options = {}) {
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
function renderStatus(status) {
  lastStatus = status;
  const bot = status.bot || {};
  const state = bot.state || 'disconnected';

  const pill = $('status-pill');
  pill.textContent = state;
  pill.className = 'pill ' + (
    state === 'connected' ? 'pill-on' :
    state === 'connecting' ? 'pill-connecting' :
    state === 'error' ? 'pill-error' : 'pill-off'
  );

  const busy = state === 'connected' || state === 'connecting';
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

  const pos = bot.position;
  $('val-pos').textContent = pos ? `${num(pos.x)} ${num(pos.y)} ${num(pos.z)}` : '–';
  const yawDeg = typeof bot.yaw === 'number' ? (bot.yaw * 180 / Math.PI).toFixed(0) : '–';
  const pitchDeg = typeof bot.pitch === 'number' ? (bot.pitch * 180 / Math.PI).toFixed(0) : '–';
  $('val-yaw').textContent = `${yawDeg}° / ${pitchDeg}°`;
  $('val-dim').textContent = fmt(bot.dimension);
  $('val-time').textContent = fmt(bot.time_of_day);
  $('val-uptime').textContent = fmtUptime(bot.uptime);

  const stats = status.stats || {};
  $('val-mined').textContent = fmt(stats.blocks_mined ?? 0);
  $('val-walked').textContent = `${num(stats.distance_walked ?? 0, 0)} m`;
  $('val-chats').textContent = fmt(stats.chats_sent ?? 0);

  // players
  const list = $('player-list');
  list.innerHTML = '';
  const players = bot.players || [];
  if (!players.length) {
    list.innerHTML = '<li class="empty">nobody in range</li>';
  } else {
    for (const player of players) {
      const li = document.createElement('li');
      li.innerHTML = `<span>${player.name}</span><span>${num(player.distance)} m &middot; ${num(player.x, 0)},${num(player.z, 0)}</span>`;
      list.appendChild(li);
    }
  }

  // inventory
  const inv = $('inventory');
  inv.innerHTML = '';
  const items = bot.inventory || [];
  if (!items.length) {
    inv.innerHTML = '<span class="empty">empty</span>';
  } else {
    for (const item of items) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.innerHTML = `${item.name} <b>×${item.count}</b>`;
      inv.appendChild(chip);
    }
  }

  // ai status
  const agent = status.agent || {};
  const last = agent.last_decision;
  $('ai-status').textContent = agent.running
    ? `running · ${agent.decisions} decisions` + (last ? ` · last: ${last.action} ${JSON.stringify(last.params || {})}` : '')
    : `idle · ${agent.decisions} decisions` + (agent.last_error ? ` · last error: ${agent.last_error}` : '');
  $('btn-ai-start').disabled = agent.running || state !== 'connected';
  $('btn-ai-stop').disabled = !agent.running;
  $('btn-ai-step').disabled = state !== 'connected';

  // backend availability note
  const backends = status.backends || {};
  const node = backends.node || {};
  $('backend-note').textContent = node.available
    ? `mineflayer bridge ready (node ${backends.node_version || '?'}).`
    : `${node.reason || 'Node bridge unavailable'} — the simulated world will be used.`;

  renderMinimap(bot);
}

function renderMinimap(bot) {
  const canvas = $('minimap');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 220;
  if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
    canvas.width = width * dpr;
    canvas.height = height * dpr;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const cx = width / 2, cy = height / 2;
  const scale = Math.min(width, height) / 80; // ~80 blocks across

  ctx.strokeStyle = '#16202b';
  ctx.lineWidth = 1;
  for (let g = -40; g <= 40; g += 8) {
    ctx.beginPath(); ctx.moveTo(cx + g * scale, 0); ctx.lineTo(cx + g * scale, height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, cy + g * scale); ctx.lineTo(width, cy + g * scale); ctx.stroke();
  }

  ctx.fillStyle = '#3b4a5c';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText('+X', width - 26, cy - 6);
  ctx.fillText('+Z', cx + 6, 14);

  const pos = bot.position;
  if (!pos) {
    ctx.fillStyle = '#64748b';
    ctx.font = '12px system-ui, sans-serif';
    ctx.fillText('not connected — no world data', 12, 20);
    return;
  }

  const toScreen = (x, z) => [cx + (x - pos.x) * scale, cy - (z - pos.z) * scale];

  for (const player of bot.players || []) {
    const [px, py] = toScreen(player.x, player.z);
    ctx.fillStyle = '#60a5fa';
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#93c5fd';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(player.name, px + 7, py + 4);
  }

  // the bot: triangle pointing along yaw (yaw 0 => +Z, which is up on screen)
  const yaw = typeof bot.yaw === 'number' ? bot.yaw : 0;
  const heading = [-Math.sin(yaw), Math.cos(yaw)];
  const [bx, by] = [cx, cy];
  ctx.save();
  ctx.translate(bx, by);
  ctx.rotate(Math.atan2(heading[0], heading[1]));
  ctx.fillStyle = '#34d399';
  ctx.beginPath();
  ctx.moveTo(0, -9); ctx.lineTo(6, 7); ctx.lineTo(0, 3); ctx.lineTo(-6, 7);
  ctx.closePath(); ctx.fill();
  ctx.restore();
  ctx.fillStyle = '#64748b';
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(`${num(pos.x, 0)}, ${num(pos.z, 0)}`, 10, height - 10);
}

/* ------------------------------------------------------------------- config */
function fillForm(cfg) {
  const mc = cfg.minecraft || {}, ol = cfg.ollama || {}, ag = cfg.agent || {};
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

  check('cfg-ollama-enabled', ol.enabled);
  set('cfg-ollama-url', ol.base_url || 'http://127.0.0.1:11434');
  set('cfg-ollama-model', ol.model || 'llama3.2');
  set('cfg-ollama-interval', ol.decision_interval || 6);
  set('cfg-ollama-temp', ol.temperature ?? 0.4);

  check('cfg-allow-movement', ag.allow_movement);
  check('cfg-allow-chat', ag.allow_chat);
  check('cfg-allow-mining', ag.allow_mining);
  check('cfg-allow-attack', ag.allow_attacking);
  check('cfg-greet', ag.greet_players);

  $('raw-config').value = JSON.stringify(cfg, null, 2);
  $('config-path').textContent = configPath;
}

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
    },
    ollama: {
      enabled: $('cfg-ollama-enabled').checked,
      base_url: $('cfg-ollama-url').value.trim(),
      model: $('cfg-ollama-model').value.trim() || 'llama3.2',
      decision_interval: Number($('cfg-ollama-interval').value) || 6,
      temperature: Number($('cfg-ollama-temp').value ?? 0.4),
    },
    agent: {
      allow_movement: $('cfg-allow-movement').checked,
      allow_chat: $('cfg-allow-chat').checked,
      allow_mining: $('cfg-allow-mining').checked,
      allow_attacking: $('cfg-allow-attack').checked,
      greet_players: $('cfg-greet').checked,
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
function connectSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${proto}//${location.host}/ws`);

  socket.onopen = () => { socketRetry = 1500; };
  socket.onmessage = (message) => {
    let frame;
    try { frame = JSON.parse(message.data); } catch (err) { return; }
    if (frame.type === 'event') appendLog(frame.data);
    else if (frame.type === 'status') {
      renderStatus(frame.data);
      if (frame.data.ollama) updateOllamaPill(frame.data.ollama);
    }
  };
  socket.onclose = () => {
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

  $('btn-ai-start').addEventListener('click', async () => {
    try { await saveConfig(true); await api('/api/ai/start', { method: 'POST' }); }
    catch (err) { reportError(err); }
  });
  $('btn-ai-stop').addEventListener('click', () => api('/api/ai/stop', { method: 'POST' }).catch(reportError));
  $('btn-ai-step').addEventListener('click', () => api('/api/ai/step', { method: 'POST' }).catch(reportError));
  $('btn-refresh-models').addEventListener('click', refreshModels);
  $('btn-pull').addEventListener('click', pullModel);

  $('btn-chat').addEventListener('click', sendChat);
  $('chat-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') sendChat(); });
  $('btn-command').addEventListener('click', runCommand);
  $('command-input').addEventListener('keydown', (event) => { if (event.key === 'Enter') runCommand(); });

  for (const button of document.querySelectorAll('#quick-actions button')) {
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
  });

  $('btn-raw-save').addEventListener('click', async () => {
    try {
      const parsed = JSON.parse($('raw-config').value);
      const data = await api('/api/config', { method: 'PUT', body: JSON.stringify(parsed) });
      config = data.config;
      fillForm(config);
      $('raw-config-status').textContent = 'Saved.';
    } catch (err) {
      $('raw-config-status').textContent = `Not saved: ${err.message}`;
    }
  });
  $('btn-raw-reload').addEventListener('click', () => loadConfig().catch(reportError));
  $('btn-reset').addEventListener('click', async () => {
    try {
      const data = await api('/api/config/reset', { method: 'POST' });
      config = data.config;
      fillForm(config);
      $('raw-config-status').textContent = 'Reset to defaults.';
    } catch (err) { reportError(err); }
  });
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

async function runAction(payload) {
  const result = await api('/api/bot/action', { method: 'POST', body: JSON.stringify(payload) });
  appendLog({ level: result.ok ? 'success' : 'warn', source: 'panel', message: result.detail || 'action run', time: nowTime() });
}

async function refreshModels() {
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
  } catch (err) { reportError(err); }
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
  try { renderStatus(await api('/api/status')); } catch (err) { reportError(err); }
  try { updateOllamaPill(await api('/api/ollama/health')); } catch (err) { /* ignore */ }
  refreshModels();
  connectSocket();
}

document.addEventListener('DOMContentLoaded', boot);
