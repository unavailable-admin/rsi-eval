"""Protocol invariants; actual container checks are recorded in docs/VERIFICATION.md."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import suite


class ContractTests(unittest.TestCase):
    def args(self, root, benchmark='mle', action='validate'):
        return argparse.Namespace(benchmark=benchmark, action=action, task='test',
                                  candidate=root / 'candidate', run_dir=root / 'run',
                                  gpu=None, timeout=5, seed=None)

    def test_validation_never_requests_private_grading(self):
        with tempfile.TemporaryDirectory() as tmp:
            for benchmark, flag in [('mle', '--validation-only'), ('ale', '--public-only'), ('fml', '--validation-only')]:
                args = self.args(Path(tmp), benchmark)
                self.assertIn(flag, suite.command(args))
                args.action = 'finalize'
                self.assertNotIn(flag, suite.command(args))

    def test_backend_paths_are_in_this_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for benchmark in ['mle', 'ale', 'fml']:
                cmd = suite.command(self.args(Path(tmp), benchmark))
                self.assertEqual(Path(cmd[1]).parent, suite.SOURCE)
                self.assertTrue(Path(cmd[1]).is_file())

    def test_private_feedback_is_not_returned_by_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            def backend(*unused, **kwargs):
                args.run_dir.mkdir()
                (args.run_dir / 'report.json').write_text(json.dumps({
                    'state': 'candidate_executed', 'validation': {'score': 1}, 'grading': {'secret': 2}}))
                return subprocess.CompletedProcess([], 0, stdout='verbose backend log', stderr='')
            out = io.StringIO()
            with patch('suite.subprocess.run', side_effect=backend), contextlib.redirect_stdout(out):
                self.assertEqual(suite.evaluate(args), 0)
            result = json.loads(out.getvalue())
            self.assertEqual(result['feedback'], {'score': 1})
            self.assertNotIn('secret', out.getvalue())
            self.assertFalse(result['private_grading_requested'])

    def test_missing_report_is_failure_even_with_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            with patch('suite.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')), contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(suite.evaluate(args), 0)
            result = json.loads((args.run_dir / 'suite-result.json').read_text())
            self.assertEqual(result['state'], 'failed')

    def test_backend_start_error_has_machine_readable_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.args(Path(tmp))
            with patch('suite.subprocess.run', side_effect=FileNotFoundError('missing interpreter')), contextlib.redirect_stdout(io.StringIO()):
                self.assertNotEqual(suite.evaluate(args), 0)
            result = json.loads((args.run_dir / 'suite-result.json').read_text())
            self.assertIn('missing interpreter', result['error_summary'])

    def test_existing_run_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / 'candidate.py'
            candidate.write_text('pass')
            argv = ['suite.py', 'validate', '--benchmark', 'mle', '--task', 'leaf-classification',
                    '--candidate', str(candidate), '--run-dir', str(root)]
            with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exc:
                    suite.main()
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(candidate.read_text(), 'pass')


if __name__ == '__main__':
    unittest.main()
