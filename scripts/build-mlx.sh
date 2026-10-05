#!/usr/bin/env bash
# Freeze Python + MLX/Metal into one signed executable, only on native Apple Silicon.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo 'MLX packaging requires an Apple Silicon macOS runner' >&2
  exit 1
fi
python="${GHOSTREEL_MLX_PYTHON:-python3}"
root="${CARGO_TARGET_DIR:-target}"
venv="$root/mlx-venv"
"$python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "MLX packaging requires Python 3.12 (GHOSTREEL_MLX_PYTHON)"'
"$python" -m venv "$venv"
# A macOS 15 runner otherwise selects MLX's macOS 15 wheel even though our ARM app supports 14.
mkdir -p "$root/mlx-wheels"
"$venv/bin/python" -m pip download --only-binary=:all: --require-hashes \
  --platform macosx_14_0_arm64 --python-version 3.12 --implementation cp --abi cp312 \
  --dest "$root/mlx-wheels" -r scripts/mlx-requirements.lock
"$venv/bin/python" -m pip install --disable-pip-version-check --no-index \
  --find-links "$root/mlx-wheels" --require-hashes -r scripts/mlx-requirements.lock
# Collect package data (Metal shaders) and dynamically imported model/processor implementations.
# PyInstaller ad-hoc signs extracted native libraries on arm64; Tauri signs the outer executable.
"$venv/bin/python" -m PyInstaller --noconfirm --clean --onefile \
  --name ghostreel-mlx --target-arch arm64 \
  --distpath "$root/release" --workpath "$root/mlx-build" --specpath "$root/mlx-build" \
  --collect-all mlx --collect-all mlx_vlm \
  --collect-all transformers --collect-all tokenizers --collect-all llguidance \
  --copy-metadata mlx-vlm --copy-metadata mlx --copy-metadata mlx-metal \
  --copy-metadata transformers --copy-metadata llguidance \
  --hidden-import sentencepiece --hidden-import PIL.Image --hidden-import llguidance.hf \
  --exclude-module torch --exclude-module torchvision --exclude-module torchaudio \
  --exclude-module tensorflow --exclude-module gradio \
  scripts/ghostreel_mlx.py
"$root/release/ghostreel-mlx" --self-test
