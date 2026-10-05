"""Protocol tests run without Apple hardware; real Metal packaging is checked in macOS CI."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('ghostreel_mlx', Path(__file__).with_name('ghostreel_mlx.py'))
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class Engine:
    def complete(self, request):
        print('library diagnostic')
        if request.get('image') == 'bad':
            raise ValueError('bad image')
        return {'ok': True, 'content': '{"description":"A person"}'}


class ProtocolTests(unittest.TestCase):
    def run_requests(self, requests):
        output, diagnostics = io.StringIO(), io.StringIO()
        source = io.StringIO('\n'.join(requests))
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
            helper.serve(Engine(), source)
        return [json.loads(line) for line in output.getvalue().splitlines()], diagnostics.getvalue()

    def test_describe_chat_and_batch_share_protocol(self):
        replies, diagnostics = self.run_requests([
            json.dumps({'id': 1, 'cmd': 'describe', 'image': 'good'}),
            json.dumps({'id': 2, 'cmd': 'complete', 'prompt': 'Write a script'}),
            json.dumps({'id': 3, 'cmd': 'batch_describe', 'items': [{'image': 'good'}, {'image': 'bad'}]}),
        ])
        self.assertEqual([r['id'] for r in replies], [1, 2, 3])
        self.assertTrue(replies[0]['ok'])
        self.assertTrue(replies[1]['ok'])
        self.assertTrue(replies[2]['results'][0]['ok'])
        self.assertEqual(replies[2]['results'][1], {'ok': False, 'error': 'bad image'})
        self.assertIn('library diagnostic', diagnostics)

    def test_invalid_requests_do_not_kill_helper(self):
        replies, _ = self.run_requests([
            'bad json', '[]', json.dumps({'id': 5, 'cmd': 'embed'}),
            json.dumps({'id': 6, 'cmd': 'complete'}),
        ])
        self.assertEqual([r['ok'] for r in replies], [False, False, False, True])
        self.assertEqual(replies[2]['id'], 5)
        self.assertEqual(replies[3]['id'], 6)


class GenerationTests(unittest.TestCase):
    def engine(self, result):
        engine = helper.MlxEngine.__new__(helper.MlxEngine)
        engine.ctx, engine.kv_bits = 8192, 4
        engine.model = SimpleNamespace(config={})
        engine.processor = object()
        engine.prepare = lambda *args: (SimpleNamespace(size=12), None, None, {})
        engine.tokenizer = SimpleNamespace(encode=lambda text: [1] * len(text))
        engine.template = lambda *args, **kwargs: args[2]
        engine.schema_processor = lambda tokenizer, schema: "grammar"
        engine.mx = SimpleNamespace(random=SimpleNamespace(seed=lambda seed: None), clear_cache=lambda: None)
        engine.options = None
        def generate(*args, **kwargs):
            engine.options = kwargs
            return result
        engine.generate = generate
        return engine

    def test_structured_generation_uses_grammar_and_preserves_metrics(self):
        engine = self.engine(SimpleNamespace(text='{"description":"red"}', finish_reason='stop',
                                           prompt_tokens=12, generation_tokens=5))
        reply = engine.complete({'prompt': 'Describe', 'schema': {'type': 'object'}, 'think': True})
        self.assertTrue(reply['ok'])
        self.assertEqual(reply['gen_tokens'], 5)
        self.assertEqual(engine.options['logits_processors'], ['grammar'])
        self.assertFalse(engine.options['enable_thinking'])
        self.assertEqual(engine.options['kv_bits'], 4)

    def test_truncation_malformed_json_and_context_overflow_are_errors(self):
        engine = self.engine(SimpleNamespace(text='{"description":', finish_reason='length',
                                           prompt_tokens=12, generation_tokens=5))
        with self.assertRaisesRegex(ValueError, 'truncated'):
            engine.complete({'prompt': 'Describe'})
        engine = self.engine(SimpleNamespace(text='invalid', finish_reason='stop',
                                           prompt_tokens=12, generation_tokens=5))
        with self.assertRaises(ValueError):
            engine.complete({'schema': {'type': 'object'}})
        with self.assertRaisesRegex(ValueError, 'exceed'):
            engine.complete({'prompt': 'x' * 8192})



if __name__ == '__main__':
    unittest.main()
