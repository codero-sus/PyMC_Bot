/* PyMC_Bot control panel - effects, themes, palette, toasts.
 * Made with ❤️ by codero-sus · https://github.com/codero-sus · https://the-codero.web.app/
 *
 * Self-contained and optional: app.js talks to it through window.FX and falls
 * back to no-ops when it is missing. Every effect honours "motion off" (the ✨
 * button / Alt+M, or the OS "reduce motion" setting).
 */
(function () {
  'use strict';

  const root = document.documentElement;
  const byId = (id) => document.getElementById(id);
  const THEMES = ['midnight', 'daylight', 'nether', 'end'];
  const THEME_COLOR = { midnight: '#070b10', daylight: '#eef3f7', nether: '#0f0606', end: '#07060f' };
  const store = {
    get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch (err) { return fallback; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch (err) { /* storage blocked */ } },
  };
  const reduceQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

  function motionOn() {
    const pref = root.dataset.motion;
    if (pref === 'off') return false;
    if (pref === 'on') return true;
    return !reduceQuery.matches;
  }

  function cssVar(name, fallback = '') {
    const value = getComputedStyle(root).getPropertyValue(name).trim();
    return value || fallback;
  }

  const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ================================================================ toasts */
  const TOAST_ICON = { success: '✅', error: '⛔', warn: '⚠️', info: '💬' };
  const TOAST_TITLE = { success: 'Done', error: 'Error', warn: 'Heads up', info: 'Info' };
  let lastToast = { text: '', at: 0 };

  function toast(message, level = 'info', options = {}) {
    const host = byId('toasts');
    if (!host || !message) return null;
    const text = String(message);
    const now = Date.now();
    if (text === lastToast.text && now - lastToast.at < 1500) return null; // no duplicate spam
    lastToast = { text, at: now };
    const timeout = options.timeout ?? (level === 'error' ? 6500 : 3800);
    const el = document.createElement('div');
    el.className = `toast ${level}`;
    el.setAttribute('role', level === 'error' ? 'alert' : 'status');
    el.innerHTML = `<span class="ico">${options.icon || TOAST_ICON[level] || '💬'}</span>
      <span class="msg"><b>${escapeHtml(options.title || TOAST_TITLE[level] || 'Info')}</b>${escapeHtml(text)}</span>
      <span class="life" style="animation-duration:${timeout}ms"></span>`;
    const close = () => {
      if (el.classList.contains('out')) return;
      el.classList.add('out');
      setTimeout(() => el.remove(), motionOn() ? 300 : 0);
    };
    el.addEventListener('click', close);
    let timer = setTimeout(close, timeout);
    el.addEventListener('mouseenter', () => { clearTimeout(timer); el.querySelector('.life').style.animationPlayState = 'paused'; });
    el.addEventListener('mouseleave', () => { timer = setTimeout(close, 1500); });
    host.appendChild(el);
    while (host.childElementCount > 5) host.firstElementChild.remove();
    return el;
  }

  /* ============================================================ value fx */
  function flashText(el, text) {
    if (typeof el === 'string') el = byId(el);
    if (!el) return;
    const value = String(text);
    if (el.textContent === value) return;
    const hadValue = el.textContent !== '' && el.textContent !== '–' && el.textContent !== '—';
    el.textContent = value;
    if (hadValue && motionOn()) {
      el.classList.remove('flash');
      void el.offsetWidth; // restart the animation
      el.classList.add('flash');
    }
  }

  const counters = new WeakMap();
  function countTo(el, target, options = {}) {
    if (typeof el === 'string') el = byId(el);
    if (!el || typeof target !== 'number' || !Number.isFinite(target)) return;
    const digits = options.digits ?? 0;
    const suffix = options.suffix ?? '';
    const render = (v) => `${v.toFixed(digits)}${suffix}`;
    const prev = counters.get(el);
    const from = prev ? prev.value : 0;
    if (prev && prev.raf) cancelAnimationFrame(prev.raf);
    if (!motionOn() || Math.abs(target - from) < Math.pow(10, -digits)) {
      counters.set(el, { value: target, raf: 0 });
      el.textContent = render(target);
      return;
    }
    const start = performance.now();
    const duration = 700;
    const state = { value: from, raf: 0 };
    counters.set(el, state);
    const step = (now) => {
      const k = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - k, 3);
      state.value = from + (target - from) * eased;
      el.textContent = render(state.value);
      if (k < 1) state.raf = requestAnimationFrame(step);
      else { state.value = target; state.raf = 0; }
    };
    state.raf = requestAnimationFrame(step);
  }

  function retrigger(el, cls) {
    if (typeof el === 'string') el = byId(el);
    if (!el || !motionOn()) return;
    el.classList.remove(cls);
    void el.offsetWidth;
    el.classList.add(cls);
    setTimeout(() => el.classList.remove(cls), 900);
  }

  /** Show a spinner on a button until the promise settles. */
  async function busy(button, work) {
    if (typeof button === 'string') button = byId(button);
    if (!button) return work;
    button.classList.add('busy');
    button.setAttribute('aria-busy', 'true');
    try { return await work; }
    finally { button.classList.remove('busy'); button.removeAttribute('aria-busy'); }
  }

  /* ============================================================ confetti */
  const confetti = (() => {
    let canvas, ctx, pieces = [], raf = 0;
    function ensure() {
      canvas = canvas || byId('confetti-canvas');
      if (!canvas) return false;
      ctx = ctx || canvas.getContext('2d');
      const dpr = window.devicePixelRatio || 1;
      const w = window.innerWidth, h = window.innerHeight;
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return true;
    }
    function frame() {
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      pieces = pieces.filter((p) => p.life > 0 && p.y < window.innerHeight + 40);
      for (const p of pieces) {
        p.vy += 0.22; p.vx *= 0.99; p.x += p.vx; p.y += p.vy; p.rot += p.vr; p.life -= 1;
        ctx.save();
        ctx.globalAlpha = Math.max(0, Math.min(1, p.life / 40));
        ctx.translate(p.x, p.y); ctx.rotate(p.rot);
        if (p.glyph) { ctx.font = `${p.size * 2}px system-ui, sans-serif`; ctx.fillText(p.glyph, -p.size, p.size); }
        else { ctx.fillStyle = p.color; ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * (p.flat ? 0.5 : 1)); }
        ctx.restore();
      }
      if (pieces.length) raf = requestAnimationFrame(frame);
      else { raf = 0; ctx.clearRect(0, 0, window.innerWidth, window.innerHeight); }
    }
    return function burst(x, y, options = {}) {
      if (!motionOn() || !ensure()) return;
      if (x && typeof x === 'object' && x.getBoundingClientRect) {
        const r = x.getBoundingClientRect();
        y = r.top + r.height / 2; x = r.left + r.width / 2;
      }
      if (typeof x !== 'number') { x = window.innerWidth / 2; y = window.innerHeight / 3; }
      const colors = options.colors || [cssVar('--accent', '#34d399'), cssVar('--accent-3', '#0ea5e9'), cssVar('--ai', '#a78bfa'), '#facc15', '#f87171'];
      const count = options.count || 110;
      for (let i = 0; i < count; i += 1) {
        const angle = Math.random() * Math.PI * 2;
        const speed = 3 + Math.random() * (options.power || 9);
        pieces.push({
          x, y, vx: Math.cos(angle) * speed, vy: Math.sin(angle) * speed - 4,
          rot: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.3,
          size: 5 + Math.random() * 6, color: colors[i % colors.length], flat: Math.random() < 0.4,
          glyph: options.glyphs ? options.glyphs[i % options.glyphs.length] : null,
          life: 90 + Math.random() * 60,
        });
      }
      if (!raf) raf = requestAnimationFrame(frame);
    };
  })();

  /* ===================================================== floating blocks */
  const particles = (() => {
    let canvas, ctx, blocks = [], raf = 0, last = 0;
    const mouse = { x: 0.5, y: 0.5 };
    function size() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const target = Math.min(46, Math.round((window.innerWidth * window.innerHeight) / 32000));
      while (blocks.length < target) blocks.push(spawn(true));
      blocks.length = target;
    }
    function spawn(anywhere) {
      return {
        x: Math.random() * window.innerWidth,
        y: anywhere ? Math.random() * window.innerHeight : window.innerHeight + 20,
        s: 4 + Math.random() * 12, z: 0.3 + Math.random() * 0.9,
        vy: 0.08 + Math.random() * 0.28, rot: Math.random() * Math.PI, vr: (Math.random() - 0.5) * 0.004,
        hue: Math.floor(Math.random() * 3), a: 0.05 + Math.random() * 0.12,
      };
    }
    function frame(now) {
      raf = requestAnimationFrame(frame);
      if (now - last < 33) return; // ~30 fps is plenty for ambience
      last = now;
      const colors = [cssVar('--accent', '#34d399'), cssVar('--accent-3', '#0ea5e9'), cssVar('--ai', '#a78bfa')];
      ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
      const px = (mouse.x - 0.5) * 30, py = (mouse.y - 0.5) * 20;
      for (const b of blocks) {
        b.y -= b.vy * b.z; b.rot += b.vr;
        if (b.y < -30) Object.assign(b, spawn(false));
        const x = b.x + px * b.z, y = b.y + py * b.z;
        ctx.save();
        ctx.translate(x, y); ctx.rotate(b.rot);
        ctx.globalAlpha = b.a;
        ctx.fillStyle = colors[b.hue];
        ctx.fillRect(-b.s / 2, -b.s / 2, b.s, b.s);
        ctx.globalAlpha = b.a * 0.6;
        ctx.strokeStyle = colors[b.hue];
        ctx.strokeRect(-b.s / 2 - 2, -b.s / 2 - 2, b.s + 4, b.s + 4);
        ctx.restore();
      }
    }
    function start() {
      canvas = canvas || byId('fx-canvas');
      if (!canvas || raf || !motionOn() || document.hidden) return;
      ctx = ctx || canvas.getContext('2d');
      size();
      raf = requestAnimationFrame(frame);
    }
    function stop() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      if (ctx) ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    }
    window.addEventListener('resize', () => { if (raf) size(); });
    window.addEventListener('pointermove', (e) => { mouse.x = e.clientX / window.innerWidth; mouse.y = e.clientY / window.innerHeight; }, { passive: true });
    document.addEventListener('visibilitychange', () => (document.hidden ? stop() : start()));
    return { start, stop };
  })();

  /* ======================================================= theme + motion */
  function setTheme(name, announce = true) {
    if (!THEMES.includes(name)) name = 'midnight';
    root.dataset.theme = name;
    store.set('pymc.theme', name);
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', THEME_COLOR[name]);
    window.dispatchEvent(new CustomEvent('fx:theme', { detail: name }));
    if (announce) toast(`Theme: ${name}`, 'info', { icon: '🎨', title: 'Appearance', timeout: 1800 });
  }
  function cycleTheme() {
    const i = THEMES.indexOf(root.dataset.theme || 'midnight');
    setTheme(THEMES[(i + 1) % THEMES.length]);
  }
  function setMotion(on, announce = true) {
    root.dataset.motion = on ? 'on' : 'off';
    store.set('pymc.motion', root.dataset.motion);
    const btn = byId('btn-motion');
    if (btn) btn.style.opacity = on ? '1' : '0.5';
    if (on) { particles.start(); document.querySelectorAll('.reveal').forEach((el) => el.classList.add('in')); }
    else particles.stop();
    window.dispatchEvent(new CustomEvent('fx:motion', { detail: on }));
    if (announce) toast(on ? 'Animations on' : 'Animations off', 'info', { icon: '✨', title: 'Motion', timeout: 1800 });
  }

  /* ================================================== cards: collapse etc */
  function enhanceCards() {
    const collapsed = new Set(JSON.parse(store.get('pymc.collapsed', '[]') || '[]'));
    const cards = [...document.querySelectorAll('.grid > .card')];
    cards.forEach((card, index) => {
      const header = card.querySelector(':scope > header');
      if (!header || card.querySelector(':scope > .card-body')) return;
      const body = document.createElement('div');
      body.className = 'card-body';
      const inner = document.createElement('div');
      inner.className = 'card-inner';
      while (header.nextSibling) inner.appendChild(header.nextSibling);
      body.appendChild(inner);
      card.appendChild(body);

      const btn = document.createElement('button');
      btn.className = 'collapse-btn';
      btn.type = 'button';
      btn.innerHTML = '<svg class="i" viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>';
      btn.title = 'Collapse / expand';
      btn.setAttribute('aria-expanded', 'true');
      const tools = document.createElement('span');
      tools.className = 'card-tools';
      tools.appendChild(btn);
      header.appendChild(tools);
      const apply = (isCollapsed) => {
        card.classList.toggle('collapsed', isCollapsed);
        btn.setAttribute('aria-expanded', String(!isCollapsed));
        if (isCollapsed) collapsed.add(card.id); else collapsed.delete(card.id);
        store.set('pymc.collapsed', JSON.stringify([...collapsed]));
      };
      btn.addEventListener('click', () => apply(!card.classList.contains('collapsed')));
      header.addEventListener('dblclick', (event) => {
        if (event.target.closest('button, input, select, a, label')) return;
        apply(!card.classList.contains('collapsed'));
      });
      if (collapsed.has(card.id)) apply(true);

      card.style.setProperty('--i', String(index % 6));
      if (motionOn()) card.classList.add('reveal');
    });

    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver((entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) { entry.target.classList.add('in'); io.unobserve(entry.target); }
        }
      }, { threshold: 0.06 });
      cards.forEach((card) => io.observe(card));
    } else {
      cards.forEach((card) => card.classList.add('in'));
    }
  }

  function setAllCollapsed(value) {
    document.querySelectorAll('.grid > .card').forEach((card) => {
      if (card.classList.contains('collapsed') !== value) card.querySelector('.collapse-btn')?.click();
    });
  }

  /* cursor spotlight + glowing border */
  function spotlight() {
    let latest = null;
    let scheduled = false;
    document.addEventListener('pointermove', (event) => {
      latest = event;
      if (scheduled) return;
      scheduled = true;
      requestAnimationFrame(() => {
        scheduled = false;
        const e = latest;
        const card = e && e.target && e.target.closest ? e.target.closest('.card') : null;
        if (!card) return;
        const r = card.getBoundingClientRect();
        card.style.setProperty('--mx', `${e.clientX - r.left}px`);
        card.style.setProperty('--my', `${e.clientY - r.top}px`);
      });
    }, { passive: true });
  }

  /* ripple on every button */
  function ripples() {
    document.addEventListener('pointerdown', (event) => {
      const btn = event.target.closest && event.target.closest('.btn');
      if (!btn || btn.disabled || !motionOn()) return;
      const r = btn.getBoundingClientRect();
      const size = Math.max(r.width, r.height);
      const dot = document.createElement('span');
      dot.className = 'ripple';
      dot.style.width = dot.style.height = `${size}px`;
      dot.style.left = `${event.clientX - r.left - size / 2}px`;
      dot.style.top = `${event.clientY - r.top - size / 2}px`;
      btn.appendChild(dot);
      dot.addEventListener('animationend', () => dot.remove());
    });
  }

  /* ============================================== scroll, nav, scrollspy */
  function scrolling() {
    const bar = byId('scroll-progress');
    const top = byId('btn-to-top');
    let ticking = false;
    const update = () => {
      ticking = false;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const k = max > 0 ? window.scrollY / max : 0;
      if (bar) bar.style.transform = `scaleX(${k})`;
      if (top) top.classList.toggle('show', window.scrollY > 500);
    };
    window.addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, { passive: true });
    update();
    if (top) top.addEventListener('click', () => window.scrollTo({ top: 0, behavior: motionOn() ? 'smooth' : 'auto' }));

    const nav = byId('section-nav');
    if (!nav) return;
    const links = [...nav.querySelectorAll('a[data-target]')];
    nav.addEventListener('click', (event) => {
      const link = event.target.closest('a[data-target]');
      if (!link) return;
      event.preventDefault();
      goTo(link.dataset.target);
    });
    // scrollspy: the active section is the last card whose top has passed the
    // sticky bars (cards in the same row resolve left-to-right, deterministically)
    const targets = links.map((link) => ({ link, el: byId(link.dataset.target) })).filter((x) => x.el);
    let current = null;
    const spy = () => {
      const line = 150;
      let pick = targets[0];
      for (const target of targets) if (target.el.getBoundingClientRect().top <= line) pick = target;
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) pick = targets[targets.length - 1];
      if (!pick || pick === current) return;
      current = pick;
      for (const { link } of targets) link.classList.toggle('active', link === pick.link);
      nav.scrollTo({ left: pick.link.offsetLeft - nav.clientWidth / 2 + pick.link.clientWidth / 2, behavior: motionOn() ? 'smooth' : 'auto' });
    };
    let queued = false;
    window.addEventListener('scroll', () => { if (!queued) { queued = true; requestAnimationFrame(() => { queued = false; spy(); }); } }, { passive: true });
    window.addEventListener('resize', spy);
    spy();
  }

  function goTo(id) {
    const card = byId(id);
    if (!card) return;
    if (card.classList.contains('collapsed')) card.querySelector('.collapse-btn')?.click();
    card.scrollIntoView({ behavior: motionOn() ? 'smooth' : 'auto', block: 'start' });
    card.classList.add('in');
    retrigger(card, 'flash-target');
    history.replaceState(null, '', `#${id}`);
  }

  /* ================================================ tagline rotator */
  function rotator() {
    const el = byId('tagline-rotator');
    if (!el) return;
    const words = ['Minecraft player bot', 'cracked + premium', 'fleets of players', 'human-like anti-AFK',
      'Ollama & Cortex brains', 'learns your playtime', 'Fabric recorder mod'];
    let i = 0;
    setInterval(() => {
      if (!motionOn() || document.hidden) return;
      el.classList.add('out');
      setTimeout(() => { i = (i + 1) % words.length; el.textContent = words[i]; el.classList.remove('out'); }, 350);
    }, 3200);
  }

  /* ===================================================== dialogs */
  function openDialog(id) {
    const dialog = byId(id);
    if (!dialog) return;
    if (typeof dialog.showModal === 'function') { if (!dialog.open) dialog.showModal(); }
    else dialog.setAttribute('open', '');
  }
  function closeDialog(dialog) {
    if (!dialog) return;
    if (typeof dialog.close === 'function') dialog.close(); else dialog.removeAttribute('open');
  }
  function dialogs() {
    for (const dialog of document.querySelectorAll('dialog.modal')) {
      dialog.addEventListener('click', (event) => {
        if (event.target === dialog || event.target.closest('[data-close]')) closeDialog(dialog);
      });
    }
    const about = () => openDialog('about-dialog');
    byId('btn-about')?.addEventListener('click', about);
    byId('btn-about-footer')?.addEventListener('click', about);
  }

  /* ===================================================== command palette */
  function paletteCommands() {
    const click = (id) => () => { const b = byId(id); if (b && !b.disabled) b.click(); else toast('Not available right now', 'warn'); };
    const quick = (action) => () => {
      const btn = [...document.querySelectorAll('#quick-actions [data-action]')].find((b) => b.dataset.action.includes(`"${action}"`));
      if (btn) btn.click();
    };
    const open = (url) => () => window.open(url, '_blank', 'noopener');
    const cmds = [];
    for (const link of document.querySelectorAll('#section-nav a[data-target]')) {
      cmds.push({ icon: '↪', label: `Go to ${link.textContent.replace(/^\S+\s/, '')}`, group: 'section', run: () => goTo(link.dataset.target) });
    }
    cmds.push(
      { icon: '🔌', label: 'Connect the bot', group: 'bot', run: click('btn-connect') },
      { icon: '⏏', label: 'Disconnect the bot', group: 'bot', run: click('btn-disconnect') },
      { icon: '🔄', label: 'Reconnect', group: 'bot', run: click('btn-reconnect') },
      { icon: '💾', label: 'Save settings', group: 'bot', run: click('btn-save') },
      { icon: '🧭', label: 'Wander', group: 'action', run: quick('wander') },
      { icon: '🦘', label: 'Jump', group: 'action', run: quick('jump') },
      { icon: '🍞', label: 'Eat', group: 'action', run: quick('eat') },
      { icon: '✋', label: 'Stop moving', group: 'action', run: quick('stop') },
      { icon: '🧠', label: 'Start the AI', group: 'ai', run: click('btn-ai-start') },
      { icon: '⏸', label: 'Stop the AI', group: 'ai', run: click('btn-ai-stop') },
      { icon: '👣', label: 'AI: step once', group: 'ai', run: click('btn-ai-step') },
      { icon: '🔍', label: 'Check Cortex LLMHoster', group: 'ai', run: click('btn-cortex-check') },
      { icon: '🎓', label: 'Train on playtime', group: 'model', run: click('btn-train-start') },
      { icon: '🧪', label: 'Demo: simulate 5 min and train', group: 'model', run: click('btn-train-simulate') },
      { icon: '⏹', label: 'Stop training', group: 'model', run: click('btn-train-stop') },
      { icon: '🔮', label: 'Preview the next action', group: 'model', run: click('btn-policy-preview') },
      { icon: '👥', label: 'Add players to the fleet', group: 'fleet', run: click('btn-populate') },
      { icon: '👋', label: 'Anti-AFK: poke all now', group: 'fleet', run: click('btn-antiafk-poke') },
      { icon: '🧹', label: 'Clear the event log', group: 'panel', run: click('btn-clear-logs') },
      { icon: '🎨', label: 'Next theme', group: 'look', run: cycleTheme },
      ...THEMES.map((name) => ({ icon: '🎨', label: `Theme: ${name}`, group: 'look', run: () => setTheme(name) })),
      { icon: '✨', label: 'Toggle animations', group: 'look', run: () => setMotion(!motionOn()) },
      { icon: '🎉', label: 'Celebrate', group: 'look', run: () => confetti() },
      { icon: '▸', label: 'Collapse all cards', group: 'look', run: () => setAllCollapsed(true) },
      { icon: '▾', label: 'Expand all cards', group: 'look', run: () => setAllCollapsed(false) },
      { icon: '↑', label: 'Back to top', group: 'look', run: () => window.scrollTo({ top: 0, behavior: 'smooth' }) },
      { icon: 'ⓘ', label: 'About & credits', group: 'help', run: () => openDialog('about-dialog') },
      { icon: '🐙', label: 'Open codero-sus on GitHub', group: 'link', run: open('https://github.com/codero-sus') },
      { icon: '🌍', label: 'Open the-codero.web.app', group: 'link', run: open('https://the-codero.web.app/') },
      { icon: '📦', label: 'Open the PyMC_Bot repository', group: 'link', run: open('https://github.com/codero-sus/PyMC_Bot') },
      { icon: '🧠', label: 'Open Cortex LLMHoster', group: 'link', run: open('https://github.com/codero-sus/Cortex_LLMHoster') },
      { icon: '📘', label: 'Open the API docs (/docs)', group: 'link', run: open('/docs') },
    );
    return cmds;
  }

  function score(label, query) {
    if (!query) return 1;
    const l = label.toLowerCase(), q = query.toLowerCase().trim();
    if (!q) return 1;
    if (l.includes(q)) return 100 - l.indexOf(q);
    let li = 0, hits = 0;
    for (const ch of q) {
      const found = l.indexOf(ch, li);
      if (found < 0) return 0;
      hits += found === li ? 2 : 1;
      li = found + 1;
    }
    return hits;
  }

  function palette() {
    const dialog = byId('palette');
    const input = byId('palette-input');
    const list = byId('palette-list');
    if (!dialog || !input || !list) return { open() {} };
    let cmds = [], shown = [], active = 0;
    const render = () => {
      const query = input.value.trim().toLowerCase();
      let scored = cmds.map((c) => ({ c, s: score(c.label, query) })).filter((x) => x.s > 0);
      // real substring hits win; loose subsequence matches only fill in when there are none
      const exact = scored.filter((x) => x.c.label.toLowerCase().includes(query));
      if (query && exact.length) scored = exact;
      shown = scored.sort((a, b) => b.s - a.s).slice(0, 40).map((x) => x.c);
      active = Math.min(active, Math.max(0, shown.length - 1));
      list.innerHTML = '';
      if (!shown.length) { list.innerHTML = '<li class="empty">no matching command</li>'; return; }
      shown.forEach((cmd, i) => {
        const li = document.createElement('li');
        li.setAttribute('role', 'option');
        if (i === active) { li.className = 'active'; li.setAttribute('aria-selected', 'true'); }
        li.innerHTML = `<span class="ico">${cmd.icon}</span><span>${escapeHtml(cmd.label)}</span><span class="grp">${cmd.group}</span>`;
        li.addEventListener('mousemove', () => { if (active !== i) { active = i; render(); } });
        li.addEventListener('click', () => run(i));
        list.appendChild(li);
      });
      list.children[active]?.scrollIntoView({ block: 'nearest' });
    };
    const run = (i) => {
      const cmd = shown[i];
      if (!cmd) return;
      closeDialog(dialog);
      setTimeout(() => cmd.run(), 60);
    };
    input.addEventListener('input', () => { active = 0; render(); });
    input.addEventListener('keydown', (event) => {
      if (event.key === 'ArrowDown') { active = (active + 1) % Math.max(1, shown.length); render(); event.preventDefault(); }
      else if (event.key === 'ArrowUp') { active = (active - 1 + shown.length) % Math.max(1, shown.length); render(); event.preventDefault(); }
      else if (event.key === 'Enter') { run(active); event.preventDefault(); }
    });
    return {
      open() {
        cmds = paletteCommands();
        input.value = '';
        active = 0;
        render();
        openDialog('palette');
        setTimeout(() => input.focus(), 20);
      },
    };
  }

  /* ===================================================== keyboard */
  function typingIn(target) {
    return target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
  }
  function keyboard(pal) {
    const konami = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
    let k = 0;
    document.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); pal.open(); return; }
      if (event.altKey && event.key.toLowerCase() === 't') { event.preventDefault(); cycleTheme(); return; }
      if (event.altKey && event.key.toLowerCase() === 'm') { event.preventDefault(); setMotion(!motionOn()); return; }
      if (typingIn(event.target)) return;
      if (event.key === '/' || event.key === '?') { event.preventDefault(); pal.open(); return; }
      k = event.key === konami[k] ? k + 1 : (event.key === konami[0] ? 1 : 0);
      if (k === konami.length) {
        k = 0;
        confetti(undefined, undefined, { count: 160, glyphs: ['🟩', '💥', '⛏️', '💎'] });
        toast('Sssssss… (the panel is fine, just a creeper saying hi)', 'warn', { icon: '💥', title: 'Creeper?' });
      }
    });
  }

  /* ===================================================== easter eggs */
  function eggs() {
    byId('heart')?.addEventListener('click', (event) => {
      confetti(event.currentTarget, undefined, { count: 40, power: 6, glyphs: ['❤️', '💚', '💜'] });
      toast('Thanks for using PyMC_Bot — codero-sus', 'success', { icon: '❤️', title: 'Made with love' });
    });
    byId('brand-logo')?.addEventListener('click', (event) => confetti(event.currentTarget, undefined, { count: 60, power: 7 }));
  }

  /* ===================================================== boot */
  function init() {
    setTheme(root.dataset.theme || 'midnight', false);
    const btn = byId('btn-motion');
    if (btn) btn.style.opacity = motionOn() ? '1' : '0.5';
    enhanceCards();
    spotlight();
    ripples();
    scrolling();
    rotator();
    dialogs();
    const pal = palette();
    byId('btn-palette')?.addEventListener('click', () => pal.open());
    byId('btn-theme')?.addEventListener('click', cycleTheme);
    byId('btn-motion')?.addEventListener('click', () => setMotion(!motionOn()));
    keyboard(pal);
    eggs();
    particles.start();
    if (location.hash && byId(location.hash.slice(1))) setTimeout(() => goTo(location.hash.slice(1)), 300);
    console.info('%cPyMC_Bot%c made with ❤️ by codero-sus · https://github.com/codero-sus · https://the-codero.web.app/',
      'background:#22c55e;color:#04231a;padding:2px 6px;border-radius:4px;font-weight:700', 'color:inherit');
  }

  window.FX = {
    toast, confetti, flashText, countTo, busy, motionOn, cssVar, setTheme, cycleTheme, setMotion, goTo,
    hit: (id) => retrigger(id, 'hit'),
    pulse: (id) => retrigger(id, 'flash'),
    escape: escapeHtml,
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
