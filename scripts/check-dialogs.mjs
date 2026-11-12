// Lint guard: questions and notices go through src/dialogs.ts (`confirmAction`, `notify`,
// `promptText`). `window.confirm` answers "yes" at once under Tauri, and the dialog plugin's
// `ask` / `message` / `confirm` do nothing in the browser build, so no other file may use them.
// The file pickers (`open`, `save`) from the same plugin are fine.
//
//   node scripts/check-dialogs.mjs        (npm run lint)
import { readdir, readFile } from 'node:fs/promises';
import { join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = fileURLToPath(new URL('..', import.meta.url));
const ALLOWED = 'src/dialogs.ts';
const BANNED_IMPORTS = new Set(['ask', 'message', 'confirm']);
const PLUGIN = String.raw`["']@tauri-apps\/plugin-dialog["']`;

// Blank out comments, keeping every newline so line numbers still match.
const blank = text => text.replace(/\/\*[\s\S]*?\*\/|(^|[^:"'`\\])\/\/.*$/gm, (m, lead) => (lead ?? '') + m.slice((lead ?? '').length).replace(/[^\n]/g, ' '));

const lineOf = (text, index) => text.slice(0, index).split('\n').length;

/** Every offence in one file's source, as `{ line, message }`. */
export function findViolations(source) {
  const text = blank(source);
  const found = [];
  const add = (index, message) => found.push({ line: lineOf(text, index), message });
  const banned = list => list.split(',').map(n => n.trim().replace(/^type\s+/, '').split(/\s+as\s+/)[0]).filter(n => BANNED_IMPORTS.has(n));

  for (const m of text.matchAll(new RegExp(String.raw`(?:import|export)\s+(?:type\s+)?\{([^}]*)\}\s*from\s*${PLUGIN}`, 'g'))) {
    const names = banned(m[1]);
    if (names.length) add(m.index, `${names.join(', ')} from @tauri-apps/plugin-dialog: use confirmAction / notify from ./dialogs`);
  }
  for (const m of text.matchAll(new RegExp(String.raw`\*\s*as\s+\w+\s*from\s*${PLUGIN}`, 'g'))) {
    add(m.index, 'namespace import of @tauri-apps/plugin-dialog: import open / save by name');
  }
  for (const m of text.matchAll(new RegExp(String.raw`\{([^}]*)\}\s*=\s*await\s+import\(\s*${PLUGIN}\s*\)`, 'g'))) {
    const names = banned(m[1]);
    if (names.length) add(m.index, `${names.join(', ')} from @tauri-apps/plugin-dialog: use confirmAction / notify from ./dialogs`);
  }
  for (const m of text.matchAll(/\b(?:window|globalThis|self)\s*\.\s*(confirm|alert|prompt)\b|(?<![\w$.])(confirm|alert|prompt)\s*\(/g)) {
    add(m.index, `${m[1] ?? m[2]}(): use confirmAction / notify / promptText from ./dialogs`);
  }
  return found;
}

async function* sources(dir) {
  for (const entry of await readdir(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* sources(path);
    else if (/\.(ts|tsx)$/.test(entry.name) && !/\.d\.ts$/.test(entry.name)) yield path;
  }
}

async function main() {
  let problems = 0;
  for await (const path of sources(join(ROOT, 'src'))) {
    const rel = relative(ROOT, path).split(sep).join('/');
    if (rel === ALLOWED) continue;
    for (const { line, message } of findViolations(await readFile(path, 'utf8'))) {
      console.error(`${rel}:${line}: ${message}`);
      problems++;
    }
  }
  if (problems) {
    console.error(`\n${problems} dialog problem(s): ask / message / confirm / alert / prompt belong in ${ALLOWED} only.`);
    process.exit(1);
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) await main();
