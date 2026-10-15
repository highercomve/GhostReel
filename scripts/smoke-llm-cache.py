#!/usr/bin/env python3
"""Opt-in real-model regression / timing check for text prefix and recurrent-state reuse.

Run with --helper, --model, --mmproj and optionally --cpu. Downloads nothing.
Checks a repeated prompt and a growing tool conversation against fresh inference.
"""
import argparse
import json
import selectors
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helper', 'model', 'mmproj'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--notes', type=int, default=60)
    args = parser.parse_args()
    if not 2048 <= args.context <= 131072 or args.notes < 1:
        parser.error('context must be 2048–131072 and notes must be positive')
    command = [str(args.helper.resolve()), '--model', str(args.model), '--mmproj', str(args.mmproj),
               '--ctx', str(args.context), '--concurrency', '1', '--kv-type', 'f16' if args.cpu else 'q4_0']
    if args.cpu:
        command += ['--cpu', '--flash-attn', 'off']

    def conversation(prompts):
        with tempfile.TemporaryFile(mode='w+') as diagnostics:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=diagnostics, text=True, bufsize=1)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)

            def read():
                if not selector.select(120):
                    raise RuntimeError('local model did not respond within 120 seconds')
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError('local model exited while processing a request')
                return json.loads(line)

            try:
                ready = read()
                assert ready['ready'] and ready['ctx'] <= args.context, ready
                responses = []
                for index, prompt in enumerate(prompts, 1):
                    request = dict(id=index, cmd='complete', prompt=prompt, max_tokens=1,
                                   seed=42, temperature=0)
                    process.stdin.write(json.dumps(request) + '\n')
                    process.stdin.flush()
                    response = read()
                    assert response['id'] == index and response['ok'], response
                    responses.append(response)
            finally:
                selector.close()
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            diagnostics.seek(0)
            return responses, diagnostics.read()

    prompt = ('<|im_start|>system\nYou help write documentary scripts.\n'
              + 'Remember the project footage. ' * args.notes
              + '<|im_end|>\n<|im_start|>user\nSuggest a title.<|im_end|>\n')
    extended = (prompt + '<|im_start|>assistant\nA morning in the city.<|im_end|>\n'
                '<|im_start|>user\nTool result for get_video: a sunrise over the city. '
                'Suggest another title.<|im_end|>\n')
    warm, log = conversation([prompt, prompt, extended])
    cold, _ = conversation([extended])
    assert warm[0]['content'] == warm[1]['content'], (warm[0], warm[1])
    assert warm[2]['content'] == cold[0]['content'], (warm[2], cold[0])
    reused = [line for line in log.splitlines() if 'reused ' in line]
    print(json.dumps({'first_s': warm[0]['secs'], 'repeat_s': warm[1]['secs'],
                      'growing_warm_s': warm[2]['secs'], 'growing_cold_s': cold[0]['secs'],
                      'prompt_tokens': [warm[0]['prompt_tokens'], warm[2]['prompt_tokens']],
                      'cache': reused, 'outputs_match': True}, indent=2))
    assert len(reused) >= 2, f'expected prefix reuse for both requests: {reused}'


if __name__ == '__main__':
    main()
