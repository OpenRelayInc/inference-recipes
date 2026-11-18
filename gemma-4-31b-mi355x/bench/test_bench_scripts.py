import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BENCH = Path(__file__).resolve().parent
SCRIPTS = ("bench.sh", "bench_text.sh")
RESULT = {
    "completed": 1, "failed": 0, "median_ttft_ms": 100,
    "p95_ttft_ms": 200, "median_tpot_ms": 3, "p95_tpot_ms": 4,
    "p95_e2el_ms": 500, "total_input_tokens": 100, "duration": 2,
    "output_throughput": 50,
}
DOCKER = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
out = next(args[i+1][:-5] for i, a in enumerate(args[:-1])
           if a == "-v" and args[i+1].endswith(":/out"))
name = args[args.index("--result-filename")+1]
result = pathlib.Path(out, name)
with open(os.environ["CALLS"], "a") as calls:
    calls.write(name + "\\n")
failed = name == os.environ.get("FAIL_RESULT")
if not failed or os.environ.get("PARTIAL"):
    result.write_text(json.dumps(json.loads(os.environ["RESULT"])))
print("fixture docker failure" if failed else "Failed requests: 0")
sys.exit(23 if failed else 0)
'''


class BenchmarkScriptTests(unittest.TestCase):
    def run_script(self, script, root, rates, fail=None, partial=False):
        work = root / "work with spaces"
        data = work / "data"
        data.mkdir(parents=True, exist_ok=True)
        (data / "text_bench.jsonl").write_text('{"prompt": "fixture"}\n')
        bindir = root / "bin"
        bindir.mkdir(exist_ok=True)
        docker = bindir / "docker"
        docker.write_text(DOCKER)
        docker.chmod(0o755)
        calls = root / "calls"
        env = dict(os.environ, WORK=str(work), PATH=str(bindir) + os.pathsep + os.environ['PATH'],
                   CALLS=str(calls), RESULT=json.dumps(RESULT), DUR="1", SEED="7",
                   DATA=str(data / 'text_bench.jsonl'))
        env.pop('FAIL_RESULT', None)
        env.pop('PARTIAL', None)
        if fail is not None:
            env['FAIL_RESULT'] = f'rate-{fail}.json'
        if partial:
            env['PARTIAL'] = '1'
        result = subprocess.run(['bash', str(BENCH / script), 'fixture', '8000', *rates],
                                env=env, text=True, capture_output=True, timeout=10)
        return result, calls.read_text().splitlines()

    def outdir(self, root, script):
        tag = 'k-fixture-s7' if script == 'bench.sh' else 'text-fixture'
        return root / 'work with spaces' / 'results' / tag

    def test_successful_sweep_replaces_previous_result(self):
        for script in SCRIPTS:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = self.outdir(root, script)
                out.mkdir(parents=True)
                (out / 'rate-1.json').write_text('{"stale": true}')
                result, calls = self.run_script(script, root, ['1', '2'])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(calls, ['rate-1.json', 'rate-2.json'])
                self.assertEqual(json.loads((out / 'rate-1.json').read_text()), RESULT)
                self.assertEqual(json.loads((out / 'rate-2.json').read_text()), RESULT)

    def test_failed_rerun_removes_stale_result_and_stops(self):
        for script in SCRIPTS:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = self.outdir(root, script)
                out.mkdir(parents=True)
                (out / 'rate-1.json').write_text(json.dumps(RESULT))
                result, calls = self.run_script(script, root, ['1', '2'], fail='1')
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertEqual(calls, ['rate-1.json'])
                self.assertFalse((out / 'rate-1.json').exists())
                self.assertIn('failed', result.stderr.lower())
                table = subprocess.run(['python3', str(BENCH / 'table.py'), str(out.parent)],
                                       text=True, capture_output=True, check=True)
                self.assertNotIn(out.name, table.stdout)

    def test_failed_run_removes_partial_result(self):
        for script in SCRIPTS:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result, calls = self.run_script(script, root, ['1'], fail='1', partial=True)
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertEqual(calls, ['rate-1.json'])
                self.assertFalse((self.outdir(root, script) / 'rate-1.json').exists())

    def test_later_failure_preserves_earlier_success(self):
        for script in SCRIPTS:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                result, calls = self.run_script(script, root, ['1', '2', '3'], fail='2', partial=True)
                out = self.outdir(root, script)
                self.assertEqual(result.returncode, 23, result.stdout + result.stderr)
                self.assertEqual(calls, ['rate-1.json', 'rate-2.json'])
                self.assertEqual(json.loads((out / 'rate-1.json').read_text()), RESULT)
                self.assertFalse((out / 'rate-2.json').exists())
                self.assertFalse((out / 'rate-3.json').exists())


if __name__ == '__main__':
    unittest.main()
