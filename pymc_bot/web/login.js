/* PyMC_Bot login page - made with ❤️ by codero-sus (https://github.com/codero-sus) */
(() => {
  'use strict';

  const form = document.getElementById('login-form');
  const status = document.getElementById('login-status');
  const submit = document.getElementById('login-submit');
  const setup = form.dataset.mode === 'setup';
  const value = (id) => (document.getElementById(id) || {}).value || '';

  function target() {
    const next = new URLSearchParams(location.search).get('next') || '/';
    return next.startsWith('/') && !next.startsWith('//') ? next : '/';  // never leave the panel
  }

  function show(message, state) {
    status.textContent = message;
    status.dataset.state = state || '';
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const body = { username: value('login-username').trim(), password: value('login-password') };
    if (!body.username || !body.password) return show('Enter your username and password.', 'error');
    if (setup) {
      body.code = value('login-code').trim();
      if (!body.code) return show('Enter the setup code from the console.', 'error');
      if (body.password.length < 8) return show('The password needs at least 8 characters.', 'error');
      if (body.password !== value('login-password2')) return show('The two passwords are different.', 'error');
    } else {
      body.remember = Boolean(document.getElementById('login-remember')?.checked);
    }
    submit.disabled = true;
    show(setup ? 'Creating the account…' : 'Logging in…');
    try {
      const response = await fetch(setup ? '/api/auth/setup' : '/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'same-origin',
        body: JSON.stringify(body),
      });
      let data = {};
      try { data = await response.json(); } catch (err) { /* empty body */ }
      if (!response.ok) {
        const detail = typeof data.detail === 'string' ? data.detail : response.statusText;
        if (response.status === 409 && setup) { location.reload(); return; }  // someone already created it
        throw new Error(detail || `HTTP ${response.status}`);
      }
      show('Welcome!', 'online');
      location.replace(target());
    } catch (err) {
      show(err.message || String(err), 'error');
      document.getElementById('login-password').select();
    } finally {
      submit.disabled = false;
    }
  });
})();
