# GhostReel binding extension

Upstream: utilityai/llama-cpp-rs, crates.io llama-cpp-2 0.1.156 (MIT OR Apache-2.0).
The native llama-cpp-sys-2 dependency remains pinned and unmodified.

Local additions: an owned resident CPU threadpool on LlamaContext, freed after the context,
and a Pin-safe mmap setter on LlamaModelParams. These support CPU expert offload without
per-graph pool allocation or paging mmap expert weights under memory pressure.
Remove this patch when equivalent upstream APIs are available.
