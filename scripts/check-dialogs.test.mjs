import assert from 'node:assert/strict';
import test from 'node:test';
import { findViolations } from './check-dialogs.mjs';

const lines = source => findViolations(source).map(v => v.line);

test('flags the plugin dialogs that only work on the desktop', () => {
  assert.deepEqual(lines('import { ask, open } from "@tauri-apps/plugin-dialog";'), [1]);
  assert.deepEqual(lines('import {\n  open,\n  message as msg,\n} from "@tauri-apps/plugin-dialog";'), [1]);
  assert.deepEqual(lines('const { confirm } = await import("@tauri-apps/plugin-dialog");'), [1]);
  assert.deepEqual(lines('import * as dialog from "@tauri-apps/plugin-dialog";'), [1]);
});

test('flags the browser dialogs that do nothing under Tauri', () => {
  assert.deepEqual(lines('if (!confirm("sure?")) return;'), [1]);
  assert.deepEqual(lines('\nwindow.alert(text);'), [2]);
  assert.deepEqual(lines('const v = window.prompt("path");'), [1]);
  assert.deepEqual(lines('globalThis.confirm("x")'), [1]);
});

test('leaves file pickers, other calls and comments alone', () => {
  assert.deepEqual(lines('import { open, save } from "@tauri-apps/plugin-dialog";'), []);
  assert.deepEqual(lines('const ok = await confirmAction("sure?");\nnotify(x);'), []);
  assert.deepEqual(lines('onConfirm(x); dialog.confirm(y); const alertCount = 1;'), []);
  assert.deepEqual(lines('// window.confirm(x)\n/* ask(y)\n confirm(z) */'), []);
  assert.deepEqual(lines('const url = "http://x"; confirm("a")'), [1]);
});
