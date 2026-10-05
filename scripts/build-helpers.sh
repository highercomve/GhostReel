#!/usr/bin/env bash
# Build GhostReel's native helpers: ghostreel-asr (whisper.cpp) and ghostreel-llm (llama.cpp).
# Picks Metal on macOS, otherwise CUDA when the toolkit is available.
#   scripts/build-helpers.sh [asr|llm]... (default: both)
#   GHOSTREEL_GPU=cuda|vulkan|metal|cpu   force a backend
#   PROFILE=release|dev             (default release)
set -euo pipefail
cd "$(dirname "$0")/.."
want="${GHOSTREEL_GPU:-${GHOSTREEL_ASR_GPU:-auto}}"
nvcc="$(command -v nvcc || true)"
[ -z "$nvcc" ] && [ -x /opt/cuda/bin/nvcc ] && nvcc=/opt/cuda/bin/nvcc
[ -z "$nvcc" ] && [ -x /usr/local/cuda/bin/nvcc ] && nvcc=/usr/local/cuda/bin/nvcc
features=()
if [ "$want" = cuda ] || { [ "$want" = auto ] && [ -n "$nvcc" ] && command -v nvidia-smi >/dev/null; }; then
  root="$(dirname "$(dirname "$nvcc")")"
  export CUDA_PATH="${CUDA_PATH:-$root}" CUDAToolkit_ROOT="${CUDAToolkit_ROOT:-$root}" CUDACXX="${CUDACXX:-$nvcc}"
  export PATH="$root/bin:$PATH"
  export CMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-86;89}"
  features=(--features cuda)
elif [ "$want" = metal ] || { [ "$want" = auto ] && [ "$(uname -s)" = Darwin ]; }; then
  features=(--features metal)
  # Distributable CPU fallback must not inherit the CI runner's CPU instruction set.
  export GGML_NATIVE=OFF
elif [ "$want" = vulkan ]; then
  features=(--features vulkan)
fi
# Full-parallel CUDA builds of whisper.cpp exhaust RAM (see plan §11).
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-4}"
profile="${PROFILE:-release}"
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=(asr llm)
# One crate at a time: both vendor a large CUDA ggml build.
for t in "${targets[@]}"; do
  echo "building ghostreel-$t ($profile) ${features[*]:-cpu}"
  cargo build -j "${JOBS:-4}" --profile "$profile" -p "ghostreel-$t" "${features[@]}"
done
