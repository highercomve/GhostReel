#!/usr/bin/env python3
"""Native CI integration: pinned model download + frozen helper vision/chat/structured JSON."""
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request


def main():
    root = Path(__file__).resolve().parent.parent
    entry = next(e for e in json.loads((root / 'crates/ghostreel-core/src/mlx-models.json').read_text())
                 if e['id'] == 'qwen3.5-9b')
    with tempfile.TemporaryDirectory(prefix='ghostreel-mlx-smoke-') as tmp:
        model = Path(tmp) / 'model'
        model.mkdir()
        for file in entry['files']:
            url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{file['name']}"
            dest = model / file['name']
            print(f"Downloading {file['name']}", flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, dest.open('wb') as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if dest.stat().st_size != file['size_bytes']:
                raise RuntimeError(f"incomplete fixture model file: {dest}")
        # PPM avoids needing a Python imaging dependency in the CI driver.
        image = Path(tmp) / 'red.ppm'
        image.write_bytes(b'P6\n128 128\n255\n' + bytes([255, 0, 0]) * 128 * 128)
        schema = {'type': 'object', 'properties': {'description': {'type': 'string', 'enum': ['red', 'green', 'blue']}},
                  'required': ['description'], 'additionalProperties': False}
        requests = [
            {'id': 1, 'cmd': 'describe', 'image': str(image), 'prompt': 'Return JSON naming the square color: red, green, or blue.',
             'schema': schema, 'max_tokens': 128},
            {'id': 2, 'cmd': 'complete', 'prompt': 'Reply with the word ready.', 'max_tokens': 16},
            {'id': 3, 'cmd': 'complete', 'prompt': 'Return JSON describing a red square.',
             'schema': schema, 'max_tokens': 128},
            {'id': 4, 'cmd': 'batch_describe', 'items': [
                {'image': str(image), 'prompt': 'Return JSON naming the square color: red, green, or blue.', 'schema': schema, 'max_tokens': 128},
                {'image': str(Path(tmp) / 'missing.ppm'), 'max_tokens': 128}]},
        ]
        helper = root / 'src-tauri/binaries/ghostreel-mlx-aarch64-apple-darwin'
        # Standard hosted Apple Silicon runners have 7 GB RAM. Keep the smoke workload
        # small while exercising the same Qwen 9B model distributed by the app.
        try:
            result = subprocess.run([str(helper), '--model-dir', str(model), '--ctx', '2048', '--kv-type', 'q4_0'],
                                    input='\n'.join(json.dumps(r) for r in requests) + '\n',
                                    text=True, capture_output=True, timeout=900)
        except subprocess.TimeoutExpired as error:
            for name, output in [('stdout', error.stdout), ('stderr', error.stderr)]:
                if isinstance(output, bytes):
                    output = output.decode('utf-8', errors='replace')
                print(f"Frozen helper {name} before timeout:\n{output or '(empty)'}", flush=True)
            raise
        if result.returncode:
            raise RuntimeError(f"MLX helper failed ({result.returncode}): {result.stderr}")
        replies = [json.loads(line) for line in result.stdout.splitlines()]
        for reply in replies:
            print('Frozen helper reply: ' + json.dumps(reply), flush=True)
        assert replies[0]['ready'], result.stdout
        for reply, request in zip(replies[1:], requests, strict=True):
            assert reply['id'] == request['id'] and reply['ok'], reply
            if request['cmd'] == 'batch_describe':
                assert reply['results'][0]['ok'] and not reply['results'][1]['ok'], reply
                assert json.loads(reply['results'][0]['content'])['description'] == 'red', reply
            elif request.get('schema'):
                assert json.loads(reply['content'])['description'] == 'red', reply
            else:
                assert reply['content'].strip(), reply
        print('Frozen MLX helper: image description, chat, structured JSON and batch errors passed', flush=True)


if __name__ == '__main__':
    main()
