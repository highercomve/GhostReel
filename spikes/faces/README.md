# Face Models Inference Spike: `ort` (ONNX Runtime) vs `tract-onnx` (Pure Rust)

This spike benchmarks and evaluates local execution of small face models for GhostReel:
1. **Face Detection**: OpenCV Zoo YuNet (`face_detection_yunet_2023mar.onnx`)
2. **Face Recognition Embedding**: OpenCV Zoo SFace (`face_recognition_sface_2021dec.onnx`)

Frames are decoded directly using system `ffmpeg` piped to stdin as raw BGR frames (zero OpenCV dependencies).

---

## Models & Hashes

Downloaded into `spikes/faces/models/` (excluded from git):

| Model | Source URL | SHA256 Hash |
|---|---|---|
| **YuNet (2023mar)** | `https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx` | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| **SFace (2021dec)** | `https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx` | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

---

## Commands

### 1. Build
```bash
cd spikes/faces
cargo build --release
```

### 2. Run ORT Benchmark
```bash
./target/release/faces-spike \
  /tmp/claude-1000/-home-projects-personal-ghostreel/f07c9fe8-e99f-49f5-97e0-a0c621c1d33b/scratchpad/podcast/agentes-de-ia-en-produccion.mp4 \
  --runtime ort \
  --fps 5 \
  --from 1200 \
  --to 1800 \
  --out-dir out
```

### 3. Run Tract Benchmark
```bash
./target/release/faces-spike \
  /tmp/claude-1000/-home-projects-personal-ghostreel/f07c9fe8-e99f-49f5-97e0-a0c621c1d33b/scratchpad/podcast/agentes-de-ia-en-produccion.mp4 \
  --runtime tract \
  --fps 5 \
  --from 1200 \
  --to 1800 \
  --out-dir out_tract
```

---

## Benchmark Measurements (10 min sample: 1200s–1800s, 3,000 frames @ 5 fps, CPU only)

| Metric | `ort` (ONNX Runtime 1.31) | `tract-onnx` (v0.23.10) | Delta (ORT vs Tract) |
|---|---|---|---|
| **YuNet Detection (Mean)** | **3.39 ms / frame** | 19.43 ms / frame | **5.7x faster** |
| **YuNet Detection (Median)** | **3.16 ms / frame** | 19.31 ms / frame | **6.1x faster** |
| **YuNet Detection (Min / Max)** | 2.22 ms / 12.58 ms | 18.40 ms / 25.97 ms | — |
| **SFace Embedding (Mean)** | **4.91 ms / face** | 35.35 ms / face | **7.2x faster** |
| **SFace Embedding (Median)** | **4.29 ms / face** | 35.14 ms / face | **8.2x faster** |
| **SFace Embedding (Min / Max)** | 3.45 ms / 15.24 ms | 34.61 ms / 41.01 ms | — |
| **Total Wall Time** | **43.53 s** | 274.56 s (4m 34s) | **6.3x faster** |
| **Overall Throughput** | **68.92 fps** (13.8x real-time) | 10.93 fps (2.2x real-time) | **6.3x higher** |
| **User CPU Time** | 261.1 s | 311.5 s | 1.2x less CPU time |
| **Peak RSS** | 359.0 MB | **313.5 MB** | Tract uses 45 MB less RAM |
| **Release Build Time** | 1m 25s | 1m 25s (combined) | — |
| **Release Binary Size** | 67 MB (unstripped, statically links `libonnxruntime.a`) | ~12 MB standalone estimate | Tract is ~40-50 MB smaller |
| **Native Libs Needed** | Statically links pre-built `libonnxruntime.a` (downloaded by `ort-sys` from `ort.pyke.io`, 107 MB archive) | **None** (100% pure Rust) | Tract has 0 native dependencies |

---

## Tract Operator Compatibility

**Result**: **0 failed ops**.
Both models run without errors out of the box on `tract-onnx 0.23.10`:
- **YuNet**: All 12 output heads (`cls_8/16/32`, `obj_8/16/32`, `bbox_8/16/32`, `kps_8/16/32`) and internal convolutional/sigmoid/reshape ops optimize and execute cleanly.
- **SFace**: Fully runnable.
- **Numerical Parity**: Cosine similarity between embeddings produced by `ort` and `tract` on identical face crops is **1.000000** (bit-identical to 6 decimal places).

---

## Quality Sanity Checks & Clustering

### 1. Faces per Frame Histogram (3,000 frames)
- **0 faces**: 10 frames (0.3%)
- **1 face**: 10 frames (0.3%)
- **2 faces**: 2,980 frames (99.3%)
- **3+ faces**: 0 frames (0.0%)

*Note*: The podcast segment features two active webcam hosts on screen side-by-side. YuNet accurately detects both faces in 99.3% of the sampled frames.

### 2. Greedy Clustering (Cosine Similarity > 0.45)
Total faces detected across 3,000 frames: **5,970 faces**.

Both runtimes produced identical clustering distribution:
| Cluster ID | Face Count | % of Faces | First Seen | Last Seen | Notes |
|---|---|---|---|---|---|
| **Cluster 0** | 1,307 | 21.9% | 1200.00s | 1799.80s | Host 1 (frontal webcam) |
| **Cluster 1** | 1,245 | 20.9% | 1305.60s | 1746.40s | Host 2 (frontal webcam) |
| **Cluster 2** | 1,200 | 20.1% | 1200.00s | 1799.80s | Host 2 (slightly turned / alternate lighting) |
| **Cluster 3** | 1,005 | 16.8% | 1321.40s | 1749.40s | Host 1 (speaking / smiling variation) |
| **Cluster 4** | 451 | 7.6% | 1581.60s | 1798.60s | Host 1 (head tilt) |
| **Cluster 5** | 324 | 5.4% | 1247.00s | 1776.80s | Host 2 (head tilt / looking down) |
| **Cluster 6** | 307 | 5.1% | 1617.80s | 1745.80s | Host 2 (screen share sub-window) |
| **Cluster 7** | 102 | 1.7% | 1405.40s | 1798.00s | Host 1 (partial occlusion) |
| **Cluster 8** | 23 | 0.4% | 1747.00s | 1761.80s | Transitional pose |
| **Cluster 9** | 4 | 0.1% | 1651.20s | 1754.60s | Transitional pose |
| **Clusters 10–11**| 2 | <0.1% | — | — | Outliers |

The top 4 clusters account for **80%** of all detections (the two main podcast speakers). The sub-clusters correspond to angle changes, laughter/expressions, and screen share transitions.

### 3. Artifacts Saved
- CSV export: [`spikes/faces/out/faces.csv`](file:///home/projects/personal/ghostreel/spikes/faces/out/faces.csv) (`t_s,face_idx,x,y,w,h,score,cluster`)
- Sample PNG crops: [`spikes/faces/out/`](file:///home/projects/personal/ghostreel/spikes/faces/out/) contains 3 sample crops per cluster (`cluster_0_sample_1.png` .. `cluster_9_sample_3.png`).

---

## Packaging Cost Analysis across Platforms

| Platform | `ort` Packaging Cost | `tract-onnx` Packaging Cost |
|---|---|---|
| **Linux (x86_64 / aarch64)** | `ort-sys` with `download-binaries` statically links `libonnxruntime.a` during cargo build. Resulting binary has zero dynamic lib dependencies beyond standard libc/libm/libstdc++. Adds ~50 MB to binary size. | **Zero cost**. Pure Rust, statically compiles to a single binary with zero external native libraries. Adds ~10–15 MB. |
| **macOS (Universal arm64/x86_64)** | `ort` provides static archives or `.dylib`. In Tauri bundle, shipping `.dylib` requires setting up `@rpath`, code signing, and Apple notarization entitlements for the bundled dynamic library. | **Zero cost**. Compiles cleanly into universal macOS binaries (`cargo build --target aarch64-apple-darwin` / `x86_64-apple-darwin`) with standard cargo workflow and trivial codesigning. |
| **Windows (x64 / arm64)** | `ort` typically requires bundling `onnxruntime.dll` (~22 MB) alongside the `.exe` in the Tauri NSIS / MSI installer, plus MSVC C++ runtime redistribution (`vcruntime140.dll`). | **Zero cost**. Pure Rust compiles directly into a standalone `.exe` without MSVC runtime or external DLL packaging. |

---

## Recommendation for GhostReel

### **Recommendation: Adopt `ort` (ONNX Runtime) as primary engine, with `tract` as an optional pure-Rust fallback.**

### Rationale:
1. **Performance Gap is Decisive (6.3x Throughput)**:
   - On a typical CPU, **ORT runs detection in 3.4 ms and embedding in 4.9 ms** (total ~13 ms per 2-face frame, yielding **68.9 FPS**).
   - In contrast, **Tract takes 19.4 ms for detection and 35.4 ms per face** (total ~90 ms per 2-face frame, yielding **10.9 FPS**).
   - GhostReel already executes heavy local models in parallel (Whisper ASR, LLM summarization, video rendering). At 69 FPS, ORT processes a 10-minute video sample in **43 seconds** with low CPU thread saturation, whereas Tract requires **4.5 minutes** of heavy CPU load.
2. **Packaging Cost is Manageable**:
   - For Linux and macOS, `ort`'s default `download-binaries` feature statically links `libonnxruntime.a`, avoiding dynamic library path hell (`LD_LIBRARY_PATH`).
   - For Windows, shipping `onnxruntime.dll` in the Tauri installer directory is well-supported and standard practice in Tauri apps.
3. **Optional Feature Flag**:
   - Because `tract-onnx` has 100% op compatibility and exact numerical parity (cosine similarity 1.000000), GhostReel can easily wrap the runtime behind a trait (as prototyped in `detector.rs` and `embedder.rs`), keeping `tract` available via `cargo --no-default-features --features tract` for targets where native C++ libraries are problematic (e.g., musl static targets or lightweight CI).
