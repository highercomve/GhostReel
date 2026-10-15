#!/usr/bin/env python3
"""Opt-in NVIDIA hardware validation. Uses installed GGUFs; downloads nothing.

Checks CUDA placement, flash attention, a 32k context, structured output, and
optional automatic / explicit MoE expert offload. Keeps logs and JSON evidence.
"""
import argparse
import json
import re
import selectors
import subprocess
import time
from pathlib import Path


def runtime_log(diagnostics):
    # Auto-fit profiles several hypothetical layouts with unloaded/zero-size buffers.
    # Validate the final real load, not expert overrides from one of those probes.
    start = diagnostics.rfind('ghostreel-llm: auto-fit selected')
    return diagnostics[start:] if start >= 0 else diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('helper', 'model', 'mmproj'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--moe-model', type=Path)
    parser.add_argument('--moe-mmproj', type=Path)
    parser.add_argument('--cpu-moe', type=int, default=32)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cpu_moe <= 1024:
        parser.error('--cpu-moe must be between 1 and 1024')
    if bool(args.moe_model) != bool(args.moe_mmproj):
        parser.error('--moe-model and --moe-mmproj go together')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.check_output([
        'nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free',
        '--format=csv,noheader',
    ], text=True).strip()
    report = {'gpu': gpu, 'helper': str(args.helper.resolve()), 'cases': []}

    schema = {
        'type': 'object', 'additionalProperties': False, 'required': ['title', 'clips'],
        'properties': {
            'title': {'type': 'string'},
            'clips': {'type': 'array', 'minItems': 2, 'maxItems': 3, 'items': {
                'type': 'object', 'additionalProperties': False,
                'required': ['video_id', 'in_s', 'out_s'],
                'properties': {'video_id': {'type': 'integer', 'enum': [1, 2, 3]},
                               'in_s': {'type': 'integer', 'minimum': 0, 'maximum': 2},
                               'out_s': {'type': 'integer', 'minimum': 3, 'maximum': 8}},
            }},
        },
    }
    brief = ('Write a short documentary script as JSON, using two or three clips. '
             'Available footage: video 1, sunrise, 8 seconds; video 2, person walking '
             'to work, 8 seconds; video 3, city at night, 8 seconds. '
             'Use a short title and only the listed video IDs. No narration field.')

    def run_case(name, model, projector, context, flash, cpu_moe=0, experts=False, long=False):
        print(f'{name}: loading', flush=True)
        command = [str(args.helper.resolve()), '--model', str(model), '--mmproj', str(projector),
                   '--ctx', str(context), '--concurrency', '1', '--kv-type', 'q4_0',
                   '--flash-attn', flash]
        if cpu_moe:
            command += ['--cpu-moe', str(cpu_moe)]
        log_path = args.output_dir / f'{name}.log'
        case = {'name': name, 'command': command, 'log': str(log_path), 'ok': False}
        report['cases'].append(case)
        with log_path.open('w+') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=log, text=True, bufsize=1)
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)

            def read():
                if not selector.select(300):
                    raise RuntimeError('helper did not respond within 300 seconds')
                line = process.stdout.readline()
                if not line:
                    raise RuntimeError(f'helper exited ({process.poll()}); see {log_path}')
                return json.loads(line)

            try:
                ready = read()
                case['ready'] = ready
                assert ready.get('ready') and ready['ngl'] > 0, ready
                assert ready['ctx'] == context, ready
                assert ready['kv_type'] == ('f16' if flash == 'off' else 'q4_0'), ready
                if experts:
                    assert ready.get('expert_overrides'), 'no expert offload was selected'
                memory = subprocess.check_output([
                    'nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits',
                ], text=True)
                used = [int(row.split(',')[1].strip()) for row in memory.splitlines()
                        if row.split(',')[0].strip() == str(process.pid)]
                assert used and used[0] > 0, f'no CUDA memory for helper PID {process.pid}'
                case['gpu_memory_mib'] = used[0]
                text = ('Background catalog note: city footage is available.\n' * 1100 if long else '') + brief
                prompt = ('<|im_start|>system\nYou are a film editor. Return the requested JSON.'
                          '<|im_end|>\n<|im_start|>user\n' + text + '<|im_end|>\n')
                request = dict(id=1, cmd='complete', prompt=prompt, schema=schema,
                               max_tokens=256, think=False, temperature=0, seed=42)
                process.stdin.write(json.dumps(request) + '\n')
                process.stdin.flush()
                response = read()
                case['response'] = response
                assert response.get('ok') and not response.get('truncated'), response
                script = json.loads(response['content'])
                assert script['title'] and 2 <= len(script['clips']) <= 3, script
                for clip in script['clips']:
                    assert clip['video_id'] in (1, 2, 3) and 0 <= clip['in_s'] <= 2, clip
                    assert 3 <= clip['out_s'] <= 8, clip
                if long:
                    assert response['prompt_tokens'] > 8192, response['prompt_tokens']
                log.flush()
                diagnostics = runtime_log(log_path.read_text())
                assert 'CUDA0' in diagnostics, 'runtime did not report a CUDA device'
                assert not re.search(r'flash_attn.*not supported|CUDA error|out of memory', diagnostics, re.I), \
                    'runtime reported unsupported attention or an allocation failure'
                if flash != 'off':
                    assert re.search(r'flash_attn\s*=\s*enabled|flash_attn enabled', diagnostics), \
                        'flash attention was not confirmed enabled'
                if experts:
                    # CUDA_Host is pinned system RAM, not GPU device memory.
                    evidence = re.findall(r'tensor blk\..*exps.*buffer type overridden to (?:CPU\S*|CUDA_Host)', diagnostics)
                    assert evidence, 'expert tensors were not confirmed in system RAM'
                    case['expert_tensor_evidence'] = evidence[:3]
                case['placement_evidence'] = [line for line in diagnostics.splitlines()
                    if re.search(r'auto-fit selected|offloaded \d+/|flash_attn\s*=|flash_attn enabled|'
                                 r'(CPU_Mapped|CUDA0).*model buffer size', line)]
                case['ok'] = True
                print(f"{name}: PASS, {used[0]} MiB GPU, {response['prompt_tokens']} prompt tokens, "
                      f"{response['gen_tokens']} generated in {response['secs']:.2f}s", flush=True)
            except Exception as error:
                case['error'] = str(error)
                raise
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
                (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
                # Give the driver a moment to release the context before fitting the next model.
                time.sleep(1)

    run_case('dense-flash-auto-32k', args.model, args.mmproj, 32768, 'auto', long=True)
    run_case('dense-flash-off', args.model, args.mmproj, 8192, 'off')
    if args.moe_model:
        run_case('moe-auto-32k', args.moe_model, args.moe_mmproj, 32768, 'auto', experts=True)
        run_case('moe-explicit-32k', args.moe_model, args.moe_mmproj, 32768, 'on',
                 cpu_moe=args.cpu_moe, experts=True)


if __name__ == '__main__':
    main()
