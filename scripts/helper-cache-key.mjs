#!/usr/bin/env node
// Fingerprint helper build inputs without invalidating on workspace version bumps.
// Keep every external lockfile entry: transitive native/build dependencies matter too.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { execFileSync, spawnSync } from 'node:child_process';

const lock = readFileSync('Cargo.lock', 'utf8').split('[[package]]')
  .filter(block => !/^\s*name = "ghostreel(?:-[^"]+)?"$/m.test(block)).join('[[package]]');
const workspace = readFileSync('Cargo.toml', 'utf8').replace(/^version = "[^"]+"$/m, 'version = "workspace"');
const compiler = process.platform === 'win32' ? spawnSync('cl', [], { encoding: 'utf8' })
  : spawnSync('c++', ['--version'], { encoding: 'utf8' });
if (compiler.error) throw compiler.error;
const hash = createHash('sha256');
for (const input of [lock, workspace, execFileSync('rustc', ['-vV']),
  execFileSync('cmake', ['--version']), compiler.stdout, compiler.stderr, process.env.ImageOS || process.platform]) {
  hash.update(input || '').update('\0');
}
console.log(hash.digest('hex').slice(0, 32));
