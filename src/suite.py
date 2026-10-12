"""JSON task descriptions and isolated candidate evaluation for MLE, ALE and FML."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from settings import WORKSPACE, SOURCE


def catalog():
    tasks = []
    for task in (WORKSPACE / 'projects/mle-bench/experiments/splits/low.txt').read_text().split():
        public = WORKSPACE / 'data/mlebench' / task / 'prepared/public'
        tasks.append(dict(benchmark='mle', task_id=task, public_data_dir=str(public),
                          description_path=str(public / 'description.md')))
    for task in (WORKSPACE / 'data/alebench/problem_ids.txt').read_text().split():
        tasks.append(dict(benchmark='ale', task_id=task, description_path=str(
            WORKSPACE / 'runs/prepare-benchmarks/ale/statements' / f'{task}.md')))
    for config in sorted((WORKSPACE / 'projects/fml-bench/ml_tasks').glob('*/config.json')):
        tasks.append(dict(benchmark='fml', task_id=config.parent.name,
                          description_path=str(config.parent / 'prompt.json')))
    return tasks


def describe(task):
    benchmark = task['benchmark']
    spec = dict(schema_version=1, **task, actions=['validate', 'finalize'])
    spec['candidate_format'] = dict(mle='python_adapter', ale='cpp20_source',
                                   fml='target_files_directory')[benchmark]
    spec['timeout_scope'] = dict(mle='each_candidate_execution_or_grading_phase',
                                fml='each_val_or_test_phase',
                                ale='official_per_case_limits_only')[benchmark]
    if benchmark == 'fml':
        path = WORKSPACE / 'projects/fml-bench/ml_tasks' / task['task_id'] / 'config.json'
        cfg = json.loads(path.read_text())
        spec.update(target_files=cfg['target_files'], metric=cfg['metric'],
                    metric_direction=cfg['metric_direction'],
                    baseline_code_dir=str(WORKSPACE / 'projects/fml-bench' / cfg['repo_dir']))
    if benchmark == 'mle':
        spec['adapter_command'] = 'python /agent.py --task /task.json'
        spec['validation_source'] = 'adapter-reported on a split of public training data'
    return spec


def command(args):
    interpreter = WORKSPACE / 'envs' / ('alebench' if args.benchmark == 'ale' else 'mlebench') / 'bin/python'
    cmd = [str(interpreter), str(SOURCE / f'run_{args.benchmark}.py'), args.task,
           '--run-dir', str(args.run_dir)]
    if args.benchmark == 'ale':
        cmd += ['--submission-file', str(args.candidate)]
        if args.action == 'validate':
            cmd += ['--public-only']
    else:
        candidate_flag = '--candidate-dir' if args.benchmark == 'fml' else '--candidate'
        cmd += [candidate_flag, str(args.candidate), '--timeout', str(args.timeout)]
        if args.gpu is not None:
            cmd += ['--gpu', str(args.gpu)]
        if args.action == 'validate':
            cmd += ['--validation-only']
    if args.benchmark == 'mle':
        cmd += ['--seed', str(args.seed or 0)]
    return cmd


def evaluate(args):
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    # Keep verbose backend logs out of stdout: stdout is one machine-readable JSON object.
    try:
        result = subprocess.run(command(args), cwd=WORKSPACE, capture_output=True, text=True)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except OSError as exc:
        code, stdout, stderr = 1, '', str(exc)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / 'runner-stdout.log').write_text(stdout)
    error_path = args.run_dir / 'runner-stderr.log'
    error_path.write_text(stderr)
    path = args.run_dir / 'report.json'
    backend = None
    try:
        if path.exists():
            backend = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        stderr += f'\nCannot read backend report: {exc}'
    expected = {'mle': {'candidate_executed', 'completed'},
                'ale': {'public_verified', 'completed'}, 'fml': {'validated', 'completed'}}
    ok = code == 0 and isinstance(backend, dict) and backend.get('state') in expected[args.benchmark]
    if not ok and code == 0:
        code = 1
    envelope = dict(schema_version=1, benchmark=args.benchmark, task_id=args.task,
                    action=args.action, state='succeeded' if ok else 'failed', exit_code=code,
                    backend_report=str(path) if backend else None,
                    private_grading_requested=args.action == 'finalize', feedback=None,
                    error_log=str(error_path) if stderr else None,
                    error_summary=(stderr[-2000:] or (backend or {}).get('error') or
                                   'Backend did not produce a successful report') if not ok else None)
    if backend:
        public = args.action == 'validate'
        if args.benchmark == 'mle':
            envelope['feedback'] = backend.get('validation' if public else 'grading')
        elif args.benchmark == 'ale':
            envelope['feedback'] = backend.get('splits', {}).get('public' if public else 'private')
            if not public:
                envelope.update(rank=backend.get('rank'), performance=backend.get('performance'))
        else:
            envelope['feedback'] = backend.get('splits', {}).get('val' if public else 'test', {}).get('metrics')
    (args.run_dir / 'suite-result.json').write_text(json.dumps(envelope, indent=2) + '\n')
    print(json.dumps(envelope), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    actions.add_parser('list')
    for action in ['describe', 'validate', 'finalize']:
        sub = actions.add_parser(action)
        sub.add_argument('--benchmark', choices=['mle', 'ale', 'fml'], required=True)
        sub.add_argument('--task', required=True)
        if action != 'describe':
            sub.add_argument('--candidate', type=Path, required=True)
            sub.add_argument('--run-dir', type=Path, required=True)
            sub.add_argument('--gpu', type=int)
            sub.add_argument('--seed', type=int, help='MLE adapter seed only; other benchmarks use their native task configuration')
            sub.add_argument('--timeout', type=int, help='Per execution/grading phase, seconds; default 14400. Not supported for ALE.')
    args = parser.parse_args()
    tasks = catalog()
    if args.action == 'list':
        print(json.dumps(dict(schema_version=1, tasks=tasks), indent=2))
        return 0
    task = next((t for t in tasks if t['benchmark'] == args.benchmark and t['task_id'] == args.task), None)
    if task is None:
        parser.error('Unknown task; use list')
    if args.action == 'describe':
        print(json.dumps(describe(task), indent=2))
        return 0
    if args.benchmark == 'ale' and (args.timeout is not None or args.gpu is not None):
        parser.error('ALE uses official per-case limits and CPU execution; --timeout/--gpu are not supported')
    if args.benchmark != 'mle' and args.seed is not None:
        parser.error('--seed is currently supported only for MLE')
    if args.timeout is not None and args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.gpu is not None and args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    args.timeout = args.timeout or 14400
    args.run_dir, args.candidate = args.run_dir.resolve(), args.candidate.resolve()
    if args.run_dir.exists():
        parser.error('Run directory exists; use a fresh directory')
    if not (args.candidate.is_dir() if args.benchmark == 'fml' else args.candidate.is_file()):
        parser.error('Candidate is missing or has the wrong file/directory type')
    return evaluate(args)


if __name__ == '__main__':
    sys.exit(main())
