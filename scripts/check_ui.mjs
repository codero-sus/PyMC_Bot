#!/usr/bin/env node
/**
 * check_ui.mjs -- static sanity checks for the web panel.
 *
 * Catches the bugs that are otherwise only visible in a browser:
 *   1. app.js referencing an element id that index.html does not define,
 *   2. browser-side code calling localhost/127.0.0.1 directly (the panel must
 *      talk to its own origin; the *server* is what talks to Ollama),
 *   3. CSS/JS assets referenced by the page but missing on disk.
 *
 * Usage: node scripts/check_ui.mjs
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const webDir = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'pymc_bot', 'web');
const html = readFileSync(resolve(webDir, 'index.html'), 'utf8');
const js = readFileSync(resolve(webDir, 'app.js'), 'utf8');

const problems = [];

// 1. ids --------------------------------------------------------------------
const htmlIds = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));
const jsIds = new Set([
  ...[...js.matchAll(/\$\('([^']+)'\)/g)].map((m) => m[1]),
  ...[...js.matchAll(/getElementById\('([^']+)'\)/g)].map((m) => m[1]),
  ...[...js.matchAll(/querySelector(?:All)?\('#([a-zA-Z0-9_-]+)/g)].map((m) => m[1]),
]);
for (const id of jsIds) {
  if (!htmlIds.has(id)) problems.push(`app.js uses #${id} but index.html has no such element`);
}

// 2. no direct localhost calls from the browser ------------------------------
const withoutPlaceholders = html.replace(/placeholder="[^"]*"/g, '');
for (const [label, source] of [['index.html', withoutPlaceholders], ['app.js', js.replace(/\/\*[\s\S]*?\*\//g, '')]]) {
  const match = source.match(/(fetch|WebSocket|src|href)\s*[=(]?\s*["'`]https?:\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)/);
  if (match) problems.push(`${label}: browser code calls ${match[2]} directly (use relative URLs)`);
}

// 3. referenced assets exist ------------------------------------------------
for (const asset of [...html.matchAll(/(?:src|href)="\/static\/([^"]+)"/g)].map((m) => m[1])) {
  if (!existsSync(resolve(webDir, asset))) problems.push(`index.html references /static/${asset} which does not exist`);
}

if (problems.length) {
  console.error('UI check failed:');
  for (const problem of problems) console.error(`  - ${problem}`);
  process.exit(1);
}
console.log(`UI check ok (${htmlIds.size} elements, ${jsIds.size} bound ids)`);
