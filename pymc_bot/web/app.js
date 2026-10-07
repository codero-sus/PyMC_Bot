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
let lastEventTs = 0;
let pollTimer = null;
let fleetStatus = null;

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

  set('cfg-agent-mode', ag.mode || 'auto');
  check('cfg-allow-movement', ag.allow_movement);
  check('cfg-allow-chat', ag.allow_chat);
  check('cfg-allow-mining', ag.allow_mining);
  check('cfg-allow-attack', ag.allow_attacking);
  check('cfg-greet', ag.greet_players);

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

  const tr = cfg.training || {};
  set('train-dataset', tr.dataset || 'pymc-playtime/dataset.jsonl');
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
    },
    ollama: {
      enabled: $('cfg-ollama-enabled').checked,
      base_url: $('cfg-ollama-url').value.trim(),
      model: $('cfg-ollama-model').value.trim() || 'llama3.2',
      decision_interval: Number($('cfg-ollama-interval').value) || 6,
      temperature: Number($('cfg-ollama-temp').value ?? 0.4),
    },
    agent: {
      mode: $('cfg-agent-mode').value,
      allow_movement: $('cfg-allow-movement').checked,
      allow_chat: $('cfg-allow-chat').checked,
      allow_mining: $('cfg-allow-mining').checked,
      allow_attacking: $('cfg-allow-attack').checked,
      greet_players: $('cfg-greet').checked,
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
    },
    training: {
      dataset: $('train-dataset').value.trim() || 'pymc-playtime/dataset.jsonl',
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

/* -------------------------------------------------------------------- fleet */
const STATE_CLASS = {
  connected: 'pill-on',
  connecting: 'pill-connecting',
  error: 'pill-error',
  disconnected: 'pill-off',
  stopped: 'pill-off',
  queued: 'pill-unknown',
};

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

  const bits = [];
  if (dataset.exists) {
    bits.push(`${dataset.samples ?? 0} samples from ${dataset.episodes ?? 0} episode(s)`);
    const top = Object.entries(dataset.actions || {}).slice(0, 4).map(([name, count]) => `${name} ${count}`);
    if (top.length) bits.push(`mostly ${top.join(', ')}`);
  } else {
    bits.push(`no dataset at ${dataset.dataset || training.settings?.dataset || 'pymc-playtime/dataset.jsonl'}`);
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
    pieces.push('dataset has no entity/item data yet — record with /pymc advanced on');
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
    $('policy-preview').textContent =
      `model '${data.run}' would do: ${prediction.action} (${top}) — turn ${prediction.motion?.dyaw ?? 0}°, step ${prediction.motion?.forward ?? 0} blocks`;
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
      <td><span class="who">${bot.username}</span>${bot.sponsor === 'primary' ? ' <span class="mini">main</span>' : ''}
        ${bot.error ? `<div class="mini error-text">${bot.error}</div>` : ''}</td>
      <td>${premium ? '<span class="mini premium">premium</span>' : '<span class="mini">offline</span>'}</td>
      <td><span class="pill ${STATE_CLASS[bot.state] || 'pill-unknown'}">${bot.state}</span></td>
      <td>${bot.ai}${bot.ai_running ? ' ●' : ''}</td>
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
    body.innerHTML = '<tr><td colspan="6" class="empty">no players yet</td></tr>';
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

  $('btn-ai-start').addEventListener('click', async () => {
    try { await saveConfig(true); await api('/api/ai/start', { method: 'POST' }); }
    catch (err) { reportError(err); }
  });
  $('btn-ai-stop').addEventListener('click', () => api('/api/ai/stop', { method: 'POST' }).catch(reportError));
  $('btn-ai-step').addEventListener('click', () => api('/api/ai/step', { method: 'POST' }).catch(reportError));
  $('btn-refresh-models').addEventListener('click', refreshModels);
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
  $('btn-fleet-remove-all').addEventListener('click', () => fleetCall('/api/fleet/stop', { remove: true }).catch(reportError));
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

async function refreshFleet() {
  try { renderFleet(await api('/api/fleet/status')); } catch (err) { /* ignore */ }
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
  try {
    const status = await api('/api/status');
    renderStatus(status);
    renderFleet(status);
    renderAntiAfk(status.antiafk);
  } catch (err) { reportError(err); }
  try { updateOllamaPill(await api('/api/ollama/health')); } catch (err) { /* ignore */ }
  refreshModels();
  await refreshFleet();
  await refreshTrainedModels();
  await refreshTraining();
  connectSocket();
}

document.addEventListener('DOMContentLoaded', boot);
