import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, mkdir, copyFile, readFile, writeFile, rm, stat, chmod } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';

const exec = promisify(execFile);
const hash = value => createHash('sha256').update(value).digest('hex');

async function fixture(t, target) {
  const root = await mkdtemp(join(tmpdir(), 'ghostreel-packaging-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'scripts'));
  for (const script of ['fetch-sidecars.mjs', 'stage-helpers.mjs']) {
    await copyFile(new URL(script, import.meta.url), join(root, 'scripts', script));
  }
  const files = { ffmpeg: 'mock ffmpeg', ffprobe: 'mock ffprobe', 'ffmpeg-LICENSE.txt': 'license' };
  let requests = 0;
  const server = createServer((req, res) => {
    requests++;
    res.end(files[req.url.slice(1)]);
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => new Promise(resolve => server.close(resolve)));
  const config = {
    version: 'fixture',
    source: 'https://example.com/build',
    sourceCode: 'https://example.com/source',
    files: Object.fromEntries(Object.entries(files).map(([name, content]) => [name, {
      url: `http://127.0.0.1:${server.address().port}/${name}`, sha256: hash(content),
    }])),
  };
  const manifest = { ffmpeg: { targets: { [target]: config } } };
  const save = () => writeFile(join(root, 'scripts/sidecars.json'), JSON.stringify(manifest));
  await save();
  const run = script => exec(process.execPath, [join(root, 'scripts', script), '--target', target], { cwd: root });
  return { root, files, config, save, run, requests: () => requests };
}

for (const target of ['aarch64-apple-darwin', 'x86_64-apple-darwin']) {
  test(`stage verified standalone sidecars and resources for ${target}`, async t => {
    const f = await fixture(t, target);
    await f.run('fetch-sidecars.mjs');
    for (const name of ['ffmpeg', 'ffprobe']) {
      const binary = join(f.root, 'src-tauri/binaries', `${name}-${target}`);
      assert.equal(await readFile(binary, 'utf8'), f.files[name]);
      if (process.platform !== 'win32') assert.equal((await stat(binary)).mode & 0o777, 0o755);
    }
    assert.equal(f.requests(), 3);
    // A second staging pass verifies and reuses the cached downloads.
    await f.run('fetch-sidecars.mjs');
    assert.equal(f.requests(), 3);
    // Corrupted cache entries must be downloaded again, never silently reused.
    await writeFile(join(f.root, 'target/sidecar-cache', `ffmpeg-${target}-${hash(f.files.ffmpeg)}`), 'corrupt');
    await f.run('fetch-sidecars.mjs');
    assert.equal(f.requests(), 4);
    await mkdir(join(f.root, 'target/release'), { recursive: true });
    for (const helper of ['ghostreel-asr', 'ghostreel-llm']) {
      await writeFile(join(f.root, 'target/release', helper), 'helper');
    }
    await f.run('stage-helpers.mjs');
    const overlay = JSON.parse(await readFile(join(f.root, 'src-tauri/tauri.bundle.json'), 'utf8'));
    assert.deepEqual(overlay.bundle.externalBin, [
      'binaries/ffmpeg', 'binaries/ffprobe', 'binaries/ghostreel-asr', 'binaries/ghostreel-llm',
    ]);
    assert.equal(overlay.bundle.resources[`binaries/ffmpeg-LICENSE-${target}.txt`], 'ffmpeg-LICENSE.txt');
    assert.equal(overlay.bundle.resources[`binaries/ffmpeg-NOTICE-${target}.txt`], 'ffmpeg-NOTICE.txt');
  });
}

test('reject a standalone download with the wrong checksum before staging it', async t => {
  const target = 'aarch64-apple-darwin';
  const f = await fixture(t, target);
  f.config.files.ffmpeg.sha256 = '0'.repeat(64);
  await f.save();
  await assert.rejects(f.run('fetch-sidecars.mjs'), /SHA-256 mismatch/);
  await assert.rejects(stat(join(f.root, 'src-tauri/binaries', `ffmpeg-${target}`)), { code: 'ENOENT' });
  await assert.rejects(stat(join(f.root, 'target/sidecar-cache', `ffmpeg-${target}-${'0'.repeat(64)}.part`)), { code: 'ENOENT' });
});

test('extract a checksum-verified executable ZIP without losing its target suffix', async t => {
  const target = 'x86_64-apple-darwin';
  const f = await fixture(t, target);
  const content = f.files.ffmpeg;
  await writeFile(join(f.root, 'ffmpeg'), content);
  await exec('zip', ['-q', 'ffmpeg.zip', 'ffmpeg'], { cwd: f.root });
  const archive = await readFile(join(f.root, 'ffmpeg.zip'));
  f.files.ffmpeg = archive;
  f.config.files.ffmpeg.sha256 = hash(archive);
  f.config.files.ffmpeg.archive = 'zip';
  f.config.files.ffmpeg.bin = 'ffmpeg';
  await f.save();
  await f.run('fetch-sidecars.mjs');
  assert.equal(await readFile(join(f.root, 'src-tauri/binaries', `ffmpeg-${target}`), 'utf8'), content);
  await assert.rejects(stat(join(f.root, 'target/sidecar-cache', `extract-ffmpeg-${target}`)), { code: 'ENOENT' });
});

test('Metal helper builds enable the backend for both packages and portable CPU fallback', async t => {
  const root = await mkdtemp(join(tmpdir(), 'ghostreel-metal-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'scripts'));
  await mkdir(join(root, 'bin'));
  await copyFile(new URL('build-helpers.sh', import.meta.url), join(root, 'scripts/build-helpers.sh'));
  const cargo = join(root, 'bin/cargo');
  await writeFile(cargo, '#!/bin/sh\nprintf "%s|%s\\n" "$GGML_NATIVE" "$*" >> "$BUILD_LOG"\n');
  await chmod(cargo, 0o755);
  const log = join(root, 'build.log');
  await exec('bash', [join(root, 'scripts/build-helpers.sh')], {
    env: { ...process.env, PATH: `${join(root, 'bin')}:${process.env.PATH}`, GHOSTREEL_GPU: 'metal', BUILD_LOG: log },
  });
  const lines = (await readFile(log, 'utf8')).trim().split('\n');
  assert.equal(lines.length, 2);
  assert.match(lines[0], /^OFF\|.*-p ghostreel-asr --features metal$/);
  assert.match(lines[1], /^OFF\|.*-p ghostreel-llm --features metal$/);
});
