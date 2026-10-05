import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { chmod, mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import test from 'node:test';

const exec = promisify(execFile);
const script = new URL('macos-signing.sh', import.meta.url).pathname;

for (const hang of ['', 'remove-trusted-cert', 'delete-keychain']) {
  test(`signing cleanup completes${hang ? ` when ${hang} hangs` : ' normally'}`, async t => {
    const root = await mkdtemp(join(tmpdir(), 'ghostreel-signing-'));
    t.after(() => rm(root, { recursive: true, force: true }));
    const bin = join(root, 'bin');
    const signing = join(root, 'ghostreel-signing');
    const log = join(root, 'commands.log');
    await mkdir(bin);
    await mkdir(signing);
    for (const name of ['cert.pem', 'build.keychain-db']) await writeFile(join(signing, name), 'fixture');
    await writeFile(join(signing, 'keychains.txt'), '    "/tmp/original-login.keychain-db"\n');
    // sudo must never prompt during cleanup; execute the mock security command as-is.
    await writeFile(join(bin, 'sudo'), '#!/bin/sh\n[ "$1" = -n ] || exit 99\nshift\nexec "$@"\n');
    await writeFile(join(bin, 'security'), '#!/bin/sh\nprintf "%s\\n" "$*" >> "$SIGNING_TEST_LOG"\nif [ "$1" = "$SIGNING_TEST_HANG" ]; then sleep 60; fi\n');
    for (const name of ['sudo', 'security']) await chmod(join(bin, name), 0o755);
    const { stdout } = await exec('bash', [script, 'cleanup'], {
      env: { ...process.env, PATH: `${bin}:${process.env.PATH}`, RUNNER_TEMP: root,
        SIGNING_TEST_LOG: log, SIGNING_TEST_HANG: hang, GHOSTREEL_CLEANUP_TIMEOUT_SECONDS: '0.1' },
      timeout: 5000,
    });
    const commands = await readFile(log, 'utf8');
    assert.match(commands, /remove-trusted-cert -d/);
    assert.match(commands, /list-keychains -d user -s \/tmp\/original-login.keychain-db/);
    assert.match(commands, /delete-keychain/);
    if (hang) assert.match(stdout, /timed out; continuing/);
    await assert.rejects(stat(signing), { code: 'ENOENT' });
    // Running cleanup a second time after partial or successful removal is harmless.
    await exec('bash', [script, 'cleanup'], { env: { ...process.env, RUNNER_TEMP: root }, timeout: 5000 });
  });
}
