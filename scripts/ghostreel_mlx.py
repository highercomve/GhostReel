#!/usr/bin/env python3
"""Self-contained Apple Silicon vision/chat helper, using GhostReel's JSON-lines protocol.

Models are downloaded by Rust, not by the runtime. stdout is reserved for protocol replies.
The frozen executable includes Python, MLX/Metal and tokenizer/processor dependencies.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time


def emit(value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


class MlxEngine:
    def __init__(self, model_dir, ctx, kv_type):
        # A runtime must never fetch untracked files or execute model repository code.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        import mlx.core as mx
        from mlx_vlm import load, generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.generate.dispatch import _prepare_generation_inputs
        from mlx_vlm.structured import build_json_schema_logits_processor

        self.mx = mx
        self.generate = generate
        self.template = apply_chat_template
        self.prepare = _prepare_generation_inputs
        self.schema_processor = build_json_schema_logits_processor
        self.ctx = ctx
        self.kv_bits = {'q4_0': 4, 'q8_0': 8, 'f16': None}[kv_type]
        self.model, self.processor = load(str(model_dir), trust_remote_code=False, use_fast=False, local_files_only=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)

    def complete(self, request):
        prompt = request.get('prompt') or 'Describe this image.'
        image = request.get('image')
        schema = request.get('schema')
        # Grammar-constrained JSON cannot contain an unconstrained reasoning prefix.
        think = bool(request.get('think', False)) and not schema
        formatted = self.template(
            self.processor, self.model.config, prompt, num_images=1 if image else 0,
            enable_thinking=think,
        )
        max_tokens = int(request.get('max_tokens', 1200))
        if max_tokens < 1 or max_tokens >= self.ctx:
            raise ValueError('output token budget must be positive and smaller than the context window')
        # Reserve output space before allocating a cache; image tokens are checked by the runtime.
        text_tokens = len(self.tokenizer.encode(formatted))
        if text_tokens + max_tokens > self.ctx:
            raise ValueError('prompt and output budget exceed the configured context window')
        if image and not Path(image).is_file():
            raise ValueError('image file does not exist')
        options = {
            'max_tokens': max_tokens,
            'temperature': float(request.get('temperature', 0.1)),
            'max_kv_size': self.ctx,
            'enable_thinking': think,
            'kv_bits': self.kv_bits,
        }
        if schema:
            options['logits_processors'] = [self.schema_processor(self.tokenizer, schema)]
        inputs, pixels, mask, extra = self.prepare(
            self.model, self.processor, formatted, [image] if image else None, None, None, {},
        )
        if inputs.size + max_tokens > self.ctx:
            raise ValueError('image, prompt and output budget exceed the configured context window')
        options.update(extra)
        options.update(input_ids=inputs, pixel_values=pixels, mask=mask)
        self.mx.random.seed(int(request.get('seed', 42)))
        started = time.monotonic()
        try:
            result = self.generate(
                self.model, self.processor, formatted, image=[image] if image else None,
                verbose=False, **options,
            )
            if result.finish_reason == 'length':
                raise ValueError('model response truncated: increase the output budget or context window')
            if result.prompt_tokens + result.generation_tokens > self.ctx:
                raise ValueError('image, prompt and response exceed the configured context window')
            content = result.text
            if schema:
                json.loads(content)  # Never return malformed JSON as a successful structured reply.
            return {
                'ok': True, 'content': content, 'prompt_tokens': result.prompt_tokens,
                'gen_tokens': result.generation_tokens, 'secs': time.monotonic() - started,
            }
        finally:
            self.mx.clear_cache()


def dispatch(engine, request):
    command = request.get('cmd')
    if command in ('describe', 'complete'):
        return engine.complete(request)
    if command == 'batch_describe':
        # Process a bounded queue serially to avoid multiplying unified-memory usage.
        # Each item has its own grammar state and may fail without losing the other results.
        results = []
        for item in request.get('items', []):
            try:
                results.append(engine.complete(item))
            except Exception as error:
                results.append({'ok': False, 'error': str(error)})
        return {'ok': True, 'results': results}
    raise ValueError(f'unsupported MLX command: {command}')


def serve(engine, source=sys.stdin):
    for line in source:
        request = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError('request must be a JSON object')
            # Libraries occasionally print warnings; only emit() may write to protocol stdout.
            with contextlib.redirect_stdout(sys.stderr):
                reply = dispatch(engine, request)
            reply['id'] = request.get('id')
        except Exception as error:
            reply = {'id': request.get('id') if isinstance(request, dict) else None,
                     'ok': False, 'error': str(error)}
        emit(reply)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--ctx', type=int, default=8192)
    parser.add_argument('--kv-type', choices=['f16', 'q8_0', 'q4_0'], default='q4_0')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        # Run the frozen dependencies and Metal kernels without a multi-GB model download.
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            import mlx_vlm.models.qwen2_5_vl
            import mlx_vlm.models.qwen3_5
            import mlx_vlm.models.gemma3
            from transformers import AutoProcessor
            from mlx_vlm.structured import build_json_schema_logits_processor
            import llguidance.hf
            mx.set_default_device(mx.gpu)
            result = mx.sum(mx.ones((16, 16)))
            mx.eval(result)
            assert result.item() == 256
        emit({'ok': True, 'backend': 'mlx', 'device': 'metal'})
        return
    if args.model_dir is None or not (args.model_dir / 'config.json').is_file():
        parser.error('--model-dir must contain a complete local MLX model')
    if args.ctx < 2048:
        parser.error('--ctx must be at least 2048')
    try:
        with contextlib.redirect_stdout(sys.stderr):
            engine = MlxEngine(args.model_dir, args.ctx, args.kv_type)
    except Exception as error:
        print(f'ghostreel-mlx: {error}', file=sys.stderr, flush=True)
        raise SystemExit(1)
    emit({'ready': True, 'backend': 'mlx', 'embed_dim': None, 'ctx': args.ctx})
    serve(engine)


if __name__ == '__main__':
    main()
