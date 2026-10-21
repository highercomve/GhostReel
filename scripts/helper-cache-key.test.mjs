import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, copyFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';

const exec = promisify(execFile);
test('helper cache survives workspace releases and invalidates on native build inputs', async t => {
  const root = await mkdtemp(join(tmpdir(), 'ghostreel-cache-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await copyFile(new URL('helper-cache-key.mjs', import.meta.url), join(root, 'key.mjs'));
  const save = async (version, dependency = '1.0.0', debug = false) => {
    await writeFile(join(root, 'Cargo.toml'), `[workspace.package]\nversion = "${version}"\nedition = "2024"\n[profile.release]\ndebug = ${debug}\n`);
    await writeFile(join(root, 'Cargo.lock'), `version = 4\n[[package]]\nname = "ghostreel-asr"\nversion = "${version}"\n[[package]]\nname = "ghostreel-llm"\nversion = "${version}"\n[[package]]\nname = "transitive-build-dependency"\nversion = "${dependency}"\nchecksum = "example"\n`);
  };
  const key = async (image = 'ubuntu22') => (await exec(process.execPath, ['key.mjs'], {
    cwd: root, env: { ...process.env, ImageOS: image },
  })).stdout.trim();
  await save('0.12.0');
  const initial = await key();
  assert.match(initial, /^[a-f0-9]{32}$/);
  await save('0.13.0');
  assert.equal(await key(), initial);
  await save('0.13.0', '1.0.1');
  assert.notEqual(await key(), initial);
  await save('0.13.0', '1.0.0', true);
  assert.notEqual(await key(), initial);
  await save('0.12.0');
  assert.notEqual(await key('ubuntu24'), initial);
});
