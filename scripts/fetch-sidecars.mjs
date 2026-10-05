#!/usr/bin/env node
// Fetch and stage pinned ffmpeg sidecar binaries for Tauri externalBin.
// Uses node: builtins only (no external dependencies).

import { createHash } from 'node:crypto';
import {
  createReadStream,
  createWriteStream,
  existsSync,
  readFileSync,
  writeFileSync,
  mkdirSync,
  renameSync,
  unlinkSync,
  rmSync,
  readdirSync,
  statSync,
  copyFileSync,
  chmodSync,
} from 'node:fs';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, join, basename } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function printUsage() {
  console.log(`Usage: node scripts/fetch-sidecars.mjs [--target <rust triple>] [--force]

Options:
  --target <triple>   Rust target triple (default: host triple)
  --force             Force re-download even if already up to date
  --help, -h          Print usage information
`);
}

function getHostTriple() {
  try {
    const res = spawnSync('rustc', ['-vV'], { encoding: 'utf8' });
    if (res.status === 0 && res.stdout) {
      for (const line of res.stdout.split('\n')) {
        if (line.startsWith('host: ')) {
          const host = line.slice(6).trim();
          if (host) return host;
        }
      }
    }
  } catch {
    // rustc missing or failed
  }
  if (process.platform === 'linux' && process.arch === 'x64') {
    return 'x86_64-unknown-linux-gnu';
  }
  if (process.platform === 'win32' && process.arch === 'x64') {
    return 'x86_64-pc-windows-msvc';
  }
  if (process.platform === 'darwin') {
    return process.arch === 'arm64' ? 'aarch64-apple-darwin' : 'x86_64-apple-darwin';
  }
  return null;
}

async function computeSha256(filePath) {
  const hash = createHash('sha256');
  const stream = createReadStream(filePath);
  for await (const chunk of stream) {
    hash.update(chunk);
  }
  return hash.digest('hex');
}

async function main() {
  let target = null;
  let force = false;

  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else if (arg === '--force') {
      force = true;
    } else if (arg === '--target') {
      i++;
      if (i >= args.length) {
        console.error('Error: --target requires an argument.');
        process.exit(1);
      }
      target = args[i];
    } else if (arg.startsWith('--target=')) {
      target = arg.slice('--target='.length);
    } else {
      console.error(`Error: Unknown argument "${arg}".`);
      printUsage();
      process.exit(1);
    }
  }

  if (!target) {
    target = getHostTriple();
    if (!target) {
      console.error('Error: Could not determine host target triple. Please pass --target <triple>.');
      process.exit(1);
    }
  }

  const manifestPath = join(__dirname, 'sidecars.json');
  if (!existsSync(manifestPath)) {
    console.error(`Error: Manifest file not found at ${manifestPath}`);
    process.exit(1);
  }

  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  const targetConfig = manifest.ffmpeg?.targets?.[target];
  if (!targetConfig) {
    const supported = Object.keys(manifest.ffmpeg?.targets || {}).join(', ');
    console.error(`Error: Unknown or unsupported target triple "${target}". Supported targets: ${supported}`);
    process.exit(1);
  }

  const isWindows = target.includes('windows') || target.includes('win32');
  const exeExt = isWindows ? '.exe' : '';
  const binariesDir = join(repoRoot, 'src-tauri', 'binaries');
  // Some macOS providers publish standalone executables instead of an archive.
  if (targetConfig.files) {
    const cacheDir = join(repoRoot, 'target', 'sidecar-cache');
    mkdirSync(cacheDir, { recursive: true });
    mkdirSync(binariesDir, { recursive: true });
    for (const [name, file] of Object.entries(targetConfig.files)) {
      const cached = join(cacheDir, `${name}-${target}-${file.sha256}`);
      if (force || !existsSync(cached) || await computeSha256(cached) !== file.sha256) {
        const part = `${cached}.part`;
        for (let attempt = 1; attempt <= 3; attempt++) {
          try {
            console.log(`Downloading ${file.url} (attempt ${attempt})...`);
            const response = await fetch(file.url, { signal: AbortSignal.timeout(120_000) });
            if (!response.ok) throw new Error(`HTTP ${response.status}: ${file.url}`);
            await pipeline(Readable.fromWeb(response.body), createWriteStream(part));
            if (await computeSha256(part) !== file.sha256) throw new Error(`SHA-256 mismatch: ${name}`);
            renameSync(part, cached);
            break;
          } catch (error) {
            // A checksum mismatch is a rejected payload, not a transient network failure.
            if (attempt === 3 || error.message.startsWith('SHA-256 mismatch')) throw error;
            console.warn(`Download failed: ${error.message}; retrying...`);
            await new Promise(resolve => setTimeout(resolve, 1000));
          } finally {
            rmSync(part, { force: true });
          }
        }
      }
      const isBinary = name === 'ffmpeg' || name === 'ffprobe';
      const destination = join(binariesDir, isBinary ? `${name}-${target}` : `${name.slice(0, -4)}-${target}.txt`);
      if (file.archive) {
        // macOS providers may distribute each executable in its own ZIP.
        const extractDir = join(cacheDir, `extract-${name}-${target}`);
        rmSync(extractDir, { recursive: true, force: true });
        mkdirSync(extractDir, { recursive: true });
        try {
          const result = spawnSync('unzip', ['-q', cached, '-d', extractDir], { stdio: 'inherit' });
          if (result.status !== 0) throw new Error(`Extraction failed: ${name}`);
          copyFileSync(join(extractDir, file.bin), destination);
        } finally {
          rmSync(extractDir, { recursive: true, force: true });
        }
      } else {
        copyFileSync(cached, destination);
      }
      chmodSync(destination, isBinary ? 0o755 : 0o644);
      console.log(`Staged ${destination}`);
    }
    writeFileSync(join(binariesDir, `ffmpeg-NOTICE-${target}.txt`),
      `FFmpeg is bundled as a separate executable. macOS builds: ${targetConfig.version}.\n` +
      `Binaries and build/source information: ${targetConfig.source}\n` +
      `FFmpeg source: ${targetConfig.sourceCode}\n` +
      'See ffmpeg-LICENSE.txt for licensing terms.\n');
    return;
  }
  const stampFile = join(binariesDir, `.ffmpeg-${target}.sha256`);

  // Check if stamp matches and outputs exist
  let upToDate = !force && existsSync(stampFile);
  if (upToDate) {
    try {
      const stampContent = readFileSync(stampFile, 'utf8').trim();
      if (stampContent !== targetConfig.sha256) {
        upToDate = false;
      } else {
        for (const binName of Object.keys(targetConfig.bins)) {
          const outPath = join(binariesDir, `${binName}-${target}${exeExt}`);
          if (!existsSync(outPath)) {
            upToDate = false;
            break;
          }
        }
      }
    } catch {
      upToDate = false;
    }
  }

  if (upToDate) {
    console.log(`ffmpeg sidecars for ${target} are up to date.`);
    process.exit(0);
  }

  // Cache directory
  const cacheDir = join(repoRoot, 'target', 'sidecar-cache');
  mkdirSync(cacheDir, { recursive: true });

  const urlObj = new URL(targetConfig.url);
  const archiveFilename = basename(urlObj.pathname);
  const cachedFilePath = join(cacheDir, archiveFilename);
  const partFilePath = join(cacheDir, `${archiveFilename}.part`);

  let cacheValid = false;
  if (existsSync(cachedFilePath)) {
    console.log(`Verifying cached archive: ${cachedFilePath}...`);
    const hash = await computeSha256(cachedFilePath);
    if (hash === targetConfig.sha256) {
      console.log(`Cached archive SHA-256 verified (${hash}).`);
      cacheValid = true;
    } else {
      console.warn(`Cached archive SHA-256 mismatch (expected ${targetConfig.sha256}, got ${hash}). Re-downloading...`);
      unlinkSync(cachedFilePath);
    }
  }

  if (!cacheValid) {
    if (existsSync(partFilePath)) {
      unlinkSync(partFilePath);
    }
    console.log(`Downloading ${targetConfig.url}...`);
    const res = await fetch(targetConfig.url);
    if (!res.ok) {
      console.error(`Error: Failed to download ${targetConfig.url}: HTTP ${res.status} ${res.statusText}`);
      process.exit(1);
    }

    const contentLength = res.headers.get('content-length');
    const totalBytes = contentLength ? parseInt(contentLength, 10) : 0;
    let receivedBytes = 0;
    let lastLoggedPercent = 0;

    const bodyStream = Readable.fromWeb(res.body);
    bodyStream.on('data', (chunk) => {
      receivedBytes += chunk.length;
      if (totalBytes > 0) {
        const percent = Math.floor((receivedBytes / totalBytes) * 100);
        if (percent >= lastLoggedPercent + 10 || percent === 100) {
          lastLoggedPercent = Math.floor(percent / 10) * 10;
          const mb = (receivedBytes / (1024 * 1024)).toFixed(1);
          const totalMb = (totalBytes / (1024 * 1024)).toFixed(1);
          console.log(`Download progress: ${percent}% (${mb} MB / ${totalMb} MB)`);
        }
      }
    });

    const fileStream = createWriteStream(partFilePath);
    await pipeline(bodyStream, fileStream);

    renameSync(partFilePath, cachedFilePath);

    console.log(`Verifying downloaded file SHA-256...`);
    const downloadedHash = await computeSha256(cachedFilePath);
    if (downloadedHash !== targetConfig.sha256) {
      unlinkSync(cachedFilePath);
      console.error(`Error: SHA-256 mismatch for downloaded file (expected ${targetConfig.sha256}, got ${downloadedHash}). Deleted.`);
      process.exit(1);
    }
    console.log(`Download verified: ${archiveFilename}`);
  }

  // Extract archive
  const extractDir = join(cacheDir, `extract-${target}`);
  rmSync(extractDir, { recursive: true, force: true });
  mkdirSync(extractDir, { recursive: true });

  console.log(`Extracting ${archiveFilename} into ${extractDir}...`);
  const isTarXz = targetConfig.archive === 'tar.xz' || archiveFilename.endsWith('.tar.xz');
  const tarFlag = isTarXz ? '-xJf' : '-xf';
  const tarRes = spawnSync('tar', [tarFlag, cachedFilePath, '-C', extractDir], { stdio: 'inherit' });
  if (tarRes.status !== 0) {
    console.error(`Error: tar extraction failed with exit code ${tarRes.status}`);
    rmSync(extractDir, { recursive: true, force: true });
    process.exit(1);
  }

  // Find the single top-level directory
  const dirEntries = readdirSync(extractDir, { withFileTypes: true }).filter((e) => e.isDirectory());
  if (dirEntries.length === 0) {
    console.error(`Error: No directory found inside extracted archive at ${extractDir}`);
    rmSync(extractDir, { recursive: true, force: true });
    process.exit(1);
  }
  const topDir = join(extractDir, dirEntries[0].name);

  // Copy binaries
  mkdirSync(binariesDir, { recursive: true });
  const staged = [];

  for (const [binName, relBinPath] of Object.entries(targetConfig.bins)) {
    const srcBin = join(topDir, relBinPath);
    if (!existsSync(srcBin)) {
      console.error(`Error: Binary not found in archive: ${srcBin}`);
      rmSync(extractDir, { recursive: true, force: true });
      process.exit(1);
    }
    const destName = `${binName}-${target}${exeExt}`;
    const destBin = join(binariesDir, destName);
    copyFileSync(srcBin, destBin);
    if (!isWindows) {
      chmodSync(destBin, 0o755);
    }
    staged.push(destBin);
  }

  // Copy LICENSE.txt if present at top level
  const licenseFile = join(topDir, 'LICENSE.txt');
  if (existsSync(licenseFile)) {
    const destLicense = join(binariesDir, 'ffmpeg-LICENSE.txt');
    copyFileSync(licenseFile, destLicense);
    staged.push(destLicense);
  }

  // Write stamp file
  writeFileSync(stampFile, `${targetConfig.sha256}\n`, 'utf8');

  // Clean up extract dir
  rmSync(extractDir, { recursive: true, force: true });

  // Print final paths and sizes
  console.log('Staged ffmpeg sidecars:');
  for (const p of staged) {
    const sizeMb = (statSync(p).size / (1024 * 1024)).toFixed(2);
    console.log(`  ${p} (${sizeMb} MB)`);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
