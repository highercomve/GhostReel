#!/usr/bin/env python3
"""Compare decode speed on the exact same installed GGUF and expert split.

No downloads. Warm up before measuring; report prompt and generation time separately.
"""
import argparse
import json
import selectors
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helper', 'model', 'mmproj', 'output-dir'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--context', type=int, default=32768)
    parser.add_argument('--cpu-moe', type=int, default=26)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = []
    prompt = ('<|im_start|>system\nYou help documentary filmmakers.<|im_end|>\n'
              '<|im_start|>user\nWrite 50 numbered practical filming tips. Each tip must have '
              '10 to 15 words. Continue until all 50 tips are written.<|im_end|>\n')
    for mode in ('legacy-runtime', 'optimized'):
        command = [str(args.helper.resolve()), '--model', str(args.model), '--mmproj', str(args.mmproj),
                   '--ctx', str(args.context), '--cpu-moe', str(args.cpu_moe), '--concurrency', '1',
                   '--kv-type', 'q4_0', '--flash-attn', 'auto']
        if mode == 'legacy-runtime':
            command += ['--threads', '4', '--mmap', 'on', '--no-threadpool']
        print(f'{mode}: loading', flush=True)
        with (args.output_dir / f'{mode}.log').open('w') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, bufsize=1)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)

            def read():
                if not selector.select(300):
                    raise RuntimeError(f'{mode}: timed out')
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f'{mode}: helper exited; see log')
                return json.loads(line)

            try:
                ready = read()
                assert ready.get('ready') and ready['ngl'] > 0 and ready.get('expert_overrides'), ready
                for number, budget in enumerate((32, 256), 1):
                    process.stdin.write(json.dumps(dict(id=number, cmd='complete', prompt=prompt,
                        max_tokens=budget, think=False, seed=42, temperature=0)) + '\n')
                    process.stdin.flush()
                    response = read()
                    assert response.get('ok'), response
                seconds = response['generation_seconds']
                result = dict(mode=mode, ready=ready, prompt_tokens=response['prompt_tokens'],
                              generated_tokens=response['gen_tokens'], prompt_seconds=response['prompt_seconds'],
                              generation_seconds=seconds, request_seconds=response['secs'],
                              tokens_per_second=response['gen_tokens'] / seconds)
                report.append(result)
                print(json.dumps(result), flush=True)
            finally:
                selector.close()
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdin.close()
                process.stdout.close()
                (args.output_dir / 'speed-report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
