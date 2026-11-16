#!/usr/bin/env node
// Stage native helper binaries and bundled CUDA runtime libraries for Tauri and CLI distribution.
//
// Helpers are linked with RUNPATH `$ORIGIN/lib:$ORIGIN/../lib/GhostReel/lib` (set in the helper
// crates' build.rs); `$ORIGIN/lib` serves the CLI tarball layout, `$ORIGIN/../lib/GhostReel/lib`
// serves deb/AppImage where Tauri puts externalBin in usr/bin and resources in usr/lib/GhostReel/.

import {
  existsSync,
  mkdirSync,
  readdirSync,
  rmSync,
  copyFileSync,
  chmodSync,
  statSync,
  realpathSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve, join, relative } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..');

function printUsage() {
  console.log(`Usage: node scripts/stage-helpers.mjs [--target <triple>] [--target-dir <dir>] [--profile <profile>] [--cuda]

Options:
  --target <triple>       Target rust triple (default: host triple)
  --target-dir <dir>      Cargo target directory (default: CARGO_TARGET_DIR or target/)
  --profile <profile>     Cargo build profile (default: release)
  --cuda                  Require CUDA runtime libraries to be staged (errors if none found)
  --help, -h              Print usage information
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

const SIDECAR_NAMES = ['ffmpeg', 'ffprobe', 'ghostreel-asr', 'ghostreel-llm', 'ghostreel-mlx'];
const ONNX_DIR = 'onnxruntime-x86_64-apple-darwin';

// tauri-build copies every externalBin into the cargo profile dir as <name> (no triple), and first
// deletes what is there with `fs::remove_file(..).unwrap()`. On the macOS runners that delete
// failed with EPERM ("Operation not permitted") for target/release/ghostreel-asr, the binary cargo
// itself had just built there, and panicked the app build. The helpers are staged under
// src-tauri/binaries by now, so the profile-dir copies are no longer needed: remove them here,
// clearing any user-immutable flag first, and fail with a listing of the file if even that fails.
function clearSidecarCopies(profileDir) {
  for (const name of SIDECAR_NAMES) {
    const path = join(profileDir, name);
    if (!existsSync(path)) continue;
    const listing = spawnSync('ls', ['-ldO@', path], { encoding: 'utf8' });
    const described = `${listing.stdout || ''}${listing.stderr || ''}`.trim();
    spawnSync('chflags', ['-R', 'nouchg', path], { stdio: 'ignore' });
    try {
      rmSync(path, { recursive: true, force: true });
      console.log(`Removed ${relative(repoRoot, path)} before tauri-build replaces it (${described})`);
    } catch (error) {
      console.error(`Error: cannot remove ${path}, which tauri-build must replace: ${error.message}`);
      console.error(described);
      process.exit(1);
    }
  }
}

// Intel Macs load ONNX Runtime at run time (crates/ghostreel-core/Cargo.toml). Bundle the dylib
// that scripts/fetch-onnxruntime.sh staged into Contents/Frameworks, where ghostreel-core/src/onnx.rs looks;
// Tauri signs it with the app. The app runs with the hardened runtime and a self-signed identity
// without a Team ID, so library validation would refuse to dlopen it: the app gets
// disable-library-validation (Intel only, written next to the dylib).
function onnxRuntimeBundle(binariesDir) {
  const dylib = join(binariesDir, ONNX_DIR, 'libonnxruntime.dylib');
  if (!existsSync(dylib)) {
    const message = `${relative(repoRoot, dylib)} not staged; run scripts/fetch-onnxruntime.sh (face and speaker matching would be unavailable)`;
    if (process.env.GHOSTREEL_REQUIRE_ONNXRUNTIME === '1') {
      console.error(`Error: ${message}`);
      process.exit(1);
    }
    console.warn(`Warning: ${message}`);
    return {};
  }
  const entitlements = join(binariesDir, ONNX_DIR, 'GhostReel.entitlements');
  writeFileSync(entitlements, `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>com.apple.security.cs.disable-library-validation</key>
  <true/>
</dict>
</plist>
`);
  return { frameworks: [dylib], entitlements };
}

function main() {
  let target = null;
  let targetDirArg = null;
  let profile = 'release';
  let cuda = false;

  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--help' || arg === '-h') {
      printUsage();
      process.exit(0);
    } else if (arg === '--cuda') {
      cuda = true;
    } else if (arg === '--profile') {
      i++;
      if (i >= args.length) {
        console.error('Error: --profile requires an argument.');
        process.exit(1);
      }
      profile = args[i];
    } else if (arg.startsWith('--profile=')) {
      profile = arg.slice('--profile='.length);
    } else if (arg === '--target') {
      i++;
      if (i >= args.length) {
        console.error('Error: --target requires an argument.');
        process.exit(1);
      }
      target = args[i];
    } else if (arg.startsWith('--target=')) {
      target = arg.slice('--target='.length);
    } else if (arg === '--target-dir') {
      i++;
      if (i >= args.length) {
        console.error('Error: --target-dir requires an argument.');
        process.exit(1);
      }
      targetDirArg = args[i];
    } else if (arg.startsWith('--target-dir=')) {
      targetDirArg = arg.slice('--target-dir='.length);
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

  const targetDir = targetDirArg || process.env.CARGO_TARGET_DIR || resolve(repoRoot, 'target');
  const isWindows = target.includes('windows') || target.includes('win32');
  const isLinux = target.includes('linux');
  const isMlx = target === 'aarch64-apple-darwin';
  const exeExt = isWindows ? '.exe' : '';

  const targetProfileDir = join(targetDir, target, profile);
  const defaultProfileDir = join(targetDir, profile);
  const binDir = existsSync(targetProfileDir) ? targetProfileDir : defaultProfileDir;

  // Verify required helpers exist BEFORE mutating any destination paths in src-tauri/
  const asrPath = join(binDir, `ghostreel-asr${exeExt}`);
  const llmPath = join(binDir, `ghostreel-llm${exeExt}`);
  const mlxPath = join(binDir, 'ghostreel-mlx');

  // Destination directories
  const binariesDir = join(repoRoot, 'src-tauri', 'binaries');
  // On macOS a previous run removed the profile-dir copies (clearSidecarCopies): until cargo
  // rebuilds them, the helpers staged then are still the latest build.
  const isDarwin = target.includes('apple-darwin');
  const stagedPath = name => join(binariesDir, `${name}-${target}${exeExt}`);
  const available = (src, name) => existsSync(src) || (isDarwin && existsSync(stagedPath(name)));

  const missingHelpers = [];
  if (!available(asrPath, 'ghostreel-asr')) missingHelpers.push(`ghostreel-asr${exeExt}`);
  if (!available(llmPath, 'ghostreel-llm')) missingHelpers.push(`ghostreel-llm${exeExt}`);
  if (isMlx && !available(mlxPath, 'ghostreel-mlx')) missingHelpers.push('ghostreel-mlx (run scripts/build-mlx.sh)');

  if (missingHelpers.length > 0) {
    console.error(`Error: Missing required helper(s) in ${binDir}: ${missingHelpers.join(', ')}`);
    console.error(`Please run scripts/build-helpers.sh first.`);
    process.exit(1);
  }

  mkdirSync(binariesDir, { recursive: true });

  const helpersToStage = [
    { name: 'ghostreel-asr', src: asrPath },
    { name: 'ghostreel-llm', src: llmPath },
    ...(isMlx ? [{ name: 'ghostreel-mlx', src: mlxPath }] : []),
  ];
  // Copy helpers
  const stagedHelpers = [];
  for (const helper of helpersToStage) {
    const destPath = stagedPath(helper.name);
    if (existsSync(helper.src)) {
      copyFileSync(helper.src, destPath);
    } else {
      console.log(`Keeping ${relative(repoRoot, destPath)}: ${relative(repoRoot, helper.src)} was cleared by an earlier run`);
    }
    if (!isWindows) {
      chmodSync(destPath, 0o755);
    }
    stagedHelpers.push({ name: helper.name, path: destPath });
  }

  if (isDarwin) {
    clearSidecarCopies(binDir);
  }

  // Prepare src-tauri/lib/ (empty first, ensure .keep is always present)
  const libDir = join(repoRoot, 'src-tauri', 'lib');
  if (existsSync(libDir)) {
    for (const entry of readdirSync(libDir)) {
      rmSync(join(libDir, entry), { recursive: true, force: true });
    }
  } else {
    mkdirSync(libDir, { recursive: true });
  }
  writeFileSync(join(libDir, '.keep'), '');

  // CUDA libs on Linux
  if (isLinux) {
    const stagedCudaLibs = new Map();

    for (const helper of stagedHelpers) {
      const lddRes = spawnSync('ldd', [helper.path], { encoding: 'utf8' });
      if (lddRes.status === 0 && lddRes.stdout) {
        for (const line of lddRes.stdout.split('\n')) {
          const trimmed = line.trim();
          const parts = trimmed.split('=>');
          if (parts.length === 2) {
            const soname = parts[0].trim();
            const targetPart = parts[1].trim();
            const resolvedPath = targetPart.split(/\s+/)[0];
            if (
              /^lib(cudart|cublas|cublasLt)\.so/.test(soname) &&
              !soname.startsWith('libcuda.so') &&
              !soname.startsWith('libnvidia-')
            ) {
              if (resolvedPath && existsSync(resolvedPath)) {
                try {
                  const realPath = realpathSync(resolvedPath);
                  stagedCudaLibs.set(soname, realPath);
                } catch {
                  // ignore unresolvable symlinks
                }
              }
            }
          }
        }
      }
    }

    if (cuda && stagedCudaLibs.size === 0) {
      console.error(
        'Error: --cuda specified, but no CUDA runtime libraries (cudart, cublas, cublasLt) were found in staged helpers via ldd.'
      );
      process.exit(1);
    } else if (stagedCudaLibs.size === 0) {
      console.log('No CUDA runtime libraries needed for this backend');
    } else {
      for (const [soname, realPath] of stagedCudaLibs.entries()) {
        const dest = join(libDir, soname);
        copyFileSync(realPath, dest);
        chmodSync(dest, 0o755);
      }
    }

    // RPATH handling on Linux
    const patchelfCheck = spawnSync('which', ['patchelf'], { encoding: 'utf8' });
    const hasPatchelf = patchelfCheck.status === 0;

    for (const helper of stagedHelpers) {
      const readelfRes = spawnSync('readelf', ['-d', helper.path], { encoding: 'utf8' });
      let hasOriginLib = false;
      if (readelfRes.status === 0 && readelfRes.stdout) {
        for (const line of readelfRes.stdout.split('\n')) {
          if ((line.includes('(RUNPATH)') || line.includes('(RPATH)')) && line.includes('$ORIGIN/lib')) {
            hasOriginLib = true;
            break;
          }
        }
      }

      if (!hasOriginLib) {
        if (hasPatchelf) {
          console.log(`Setting RPATH on ${helper.path}...`);
          const patchRes = spawnSync(
            'patchelf',
            ['--set-rpath', '$ORIGIN/lib:$ORIGIN/../lib/GhostReel/lib', helper.path],
            { stdio: 'inherit' }
          );
          if (patchRes.status !== 0) {
            console.warn(`Warning: patchelf failed on ${helper.path}`);
          }
        } else {
          console.warn(
            `Warning: patchelf not found on PATH, and ${helper.path} does not contain $ORIGIN/lib in RUNPATH/RPATH.`
          );
        }
      }
    }
  }

  // CUDA libs on Windows
  if (isWindows) {
    const cudaWanted = cuda || process.env.GHOSTREEL_GPU === 'cuda';
    if (cudaWanted) {
      const cudaPath = process.env.CUDA_PATH;
      if (!cudaPath) {
        console.error('Error: --cuda specified (or GHOSTREEL_GPU=cuda), but CUDA_PATH environment variable is not set.');
        process.exit(1);
      }
      const searchDirs = [join(cudaPath, 'bin'), join(cudaPath, 'bin', 'x64')];
      const foundDlls = new Map();
      for (const dir of searchDirs) {
        if (existsSync(dir)) {
          for (const file of readdirSync(dir)) {
            if (/^(cudart64_|cublas64_|cublasLt64_).*\.dll$/i.test(file)) {
              if (!foundDlls.has(file)) {
                foundDlls.set(file, join(dir, file));
              }
            }
          }
        }
      }
      if (foundDlls.size === 0) {
        console.error(
          `Error: No CUDA DLLs matching /^(cudart64_|cublas64_|cublasLt64_).*\\.dll$/i found in ${searchDirs.join(', ')}`
        );
        process.exit(1);
      }
      for (const [file, filePath] of foundDlls.entries()) {
        copyFileSync(filePath, join(libDir, file));
      }
    } else {
      console.log('No CUDA runtime libraries needed for this backend');
    }
  }

  // Tauri config overlay for `tauri build --config src-tauri/tauri.bundle.json`. externalBin and
  // resources are kept out of tauri.conf.json because tauri-build fails when a listed sidecar is
  // missing, which would break `tauri dev` and clippy.
  const externalBin = [];
  for (const name of ['ffmpeg', 'ffprobe', 'ghostreel-asr', 'ghostreel-llm', ...(isMlx ? ['ghostreel-mlx'] : [])]) {
    if (existsSync(join(binariesDir, `${name}-${target}${exeExt}`))) {
      externalBin.push(`binaries/${name}`);
    } else if (name === 'ffmpeg' || name === 'ffprobe') {
      console.warn(`Warning: ${name} not staged; run node scripts/fetch-sidecars.mjs --target ${target}`);
    }
  }
  const macOS = {
    ...(isMlx ? { minimumSystemVersion: '14.0' } : {}),
    ...(target === 'x86_64-apple-darwin' ? onnxRuntimeBundle(binariesDir) : {}),
  };
  const onnxResources = target === 'x86_64-apple-darwin'
    ? Object.fromEntries([['LICENSE', 'onnxruntime-LICENSE.txt'], ['ThirdPartyNotices.txt', 'onnxruntime-ThirdPartyNotices.txt']]
      .filter(([source]) => existsSync(join(binariesDir, ONNX_DIR, source)))
      .map(([source, name]) => [`binaries/${ONNX_DIR}/${source}`, name]))
    : {};
  const overlay = {
    bundle: {
      externalBin,
      ...(Object.keys(macOS).length ? { macOS } : {}),
      // Linux: usr/lib/GhostReel/lib (helper RUNPATH $ORIGIN/../lib/GhostReel/lib).
      // Windows: next to the exe, where the DLL loader looks.
      resources: {
        'lib/': isWindows ? './' : 'lib/',
        ...Object.fromEntries(['ffmpeg-LICENSE.txt', 'ffmpeg-NOTICE.txt']
          .map(name => ({ name, source: target.includes('apple-darwin')
            ? `${name.slice(0, -4)}-${target}.txt` : name }))
          .filter(({ source }) => existsSync(join(binariesDir, source)))
          .map(({ name, source }) => [`binaries/${source}`, name])),
        ...onnxResources,
      },
    },
  };
  const overlayPath = join(repoRoot, 'src-tauri', 'tauri.bundle.json');
  writeFileSync(overlayPath, `${JSON.stringify(overlay, null, 2)}\n`);
  console.log(`Wrote ${relative(repoRoot, overlayPath)} (externalBin: ${externalBin.join(', ')})`);

  // Print summary
  console.log('\nStaging summary:');
  const summaryItems = [];
  for (const helper of stagedHelpers) {
    summaryItems.push(helper.path);
  }
  if (existsSync(libDir)) {
    for (const entry of readdirSync(libDir)) {
      if (entry === '.keep') continue;
      summaryItems.push(join(libDir, entry));
    }
  }
  for (const filePath of summaryItems) {
    const stat = statSync(filePath);
    const sizeMb = (stat.size / (1024 * 1024)).toFixed(2);
    const relPath = relative(repoRoot, filePath);
    console.log(`  ${relPath}: ${sizeMb} MB`);
  }
}

main();
