import json
import os
from pathlib import Path
import subprocess

import pytest

import run as cli


PROJECT = Path(__file__).resolve().parents[1]


def test_run_cli_without_seconds_passes_unlimited_duration(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run_session(**kwargs):
        seen.update(kwargs)
        return {'mode': 'run', 'frames': 0}

    monkeypatch.setattr('vd.runtime.run_session', fake_run_session)
    monkeypatch.setattr('sys.argv', [str(PROJECT / 'run.py'), 'run',
                                    '--recording', str(tmp_path / 'session')])
    cli.main()

    assert seen['seconds'] is None
    assert seen['physical'] is True
    assert seen['learn_lead'] is True
    assert seen['recording'] is True
    assert seen['fps'] == 60
    assert seen['capture_priority'] == 5
    assert seen['variant'] == 'baseline'
    assert 'Ctrl+C' in capsys.readouterr().out


@pytest.mark.parametrize(('variant', 'fps', 'priority'), [
    ('baseline', 60, 5),
    ('responsive', 60, 0),
    ('quiet', 60, 10),
    ('fps50', 50, 5),
    ('fps45', 45, 5),
    ('fps30', 30, 5),
    ('deep-quiet', 30, 10),
])
def test_run_cli_resolves_capture_profile(monkeypatch, tmp_path, variant, fps, priority):
    seen = {}

    def fake_run_session(**kwargs):
        seen.update(kwargs)
        return {'mode': 'run', 'frames': 0}

    monkeypatch.setattr('vd.runtime.run_session', fake_run_session)
    monkeypatch.setattr('sys.argv', [str(PROJECT / 'run.py'), 'run', '--variant', variant,
                                    '--recording', str(tmp_path / 'session')])
    cli.main()

    assert seen['fps'] == fps
    assert seen['capture_priority'] == priority
    assert seen['variant'] == variant


def test_explicit_capture_settings_override_the_selected_profile(monkeypatch, tmp_path):
    seen = {}

    def fake_run_session(**kwargs):
        seen.update(kwargs)
        return {'mode': 'run', 'frames': 0}

    monkeypatch.setattr('vd.runtime.run_session', fake_run_session)
    monkeypatch.setattr('sys.argv', [str(PROJECT / 'run.py'), 'run', '--variant', 'quiet',
                                    '--fps', '50', '--capture-priority', '2',
                                    '--recording', str(tmp_path / 'session')])
    cli.main()

    assert seen['fps'] == 50
    assert seen['capture_priority'] == 2
    assert seen['variant'] == 'quiet'


def test_run_cli_prints_partial_performance_summary_after_ctrl_c(monkeypatch, tmp_path, capsys):
    recording = tmp_path / 'partial-run'
    recording.mkdir()
    (recording / 'manifest.json').write_text(json.dumps({
        'complete': False,
        'frames_written': 120,
        'frames_dropped': 0,
        'capture_sequences_skipped': 0,
        'performance': {'frame_processing_ms': {'p95': 7.0}},
    }))

    def interrupt(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr('vd.runtime.run_session', interrupt)
    monkeypatch.setattr('sys.argv', [str(PROJECT / 'run.py'), 'run',
                                    '--recording', str(recording)])
    cli.main()

    output = capsys.readouterr().out
    assert 'Stopped.' in output
    assert str(recording) in output
    assert 'frame_processing_ms' in output
    assert '7.0' in output


def invoke_start_solver(tmp_path, *args):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    capture_args = tmp_path / 'python-args.txt'
    fake_python = fake_bin / 'python'
    fake_python.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$SOLVER_TEST_ARGS"\n')
    fake_python.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = str(fake_bin) + os.pathsep + env['PATH']
    env['SOLVER_TEST_ARGS'] = str(capture_args)
    subprocess.run([str(PROJECT / 'start-solver.sh'), *args], env=env, check=True)
    return capture_args.read_text().splitlines()


def test_start_solver_omits_time_limit_by_default(tmp_path):
    args = invoke_start_solver(tmp_path)
    assert args[1] == 'run'
    assert '--seconds' not in args
    assert '--no-recording' not in args


def test_start_solver_keeps_explicit_time_limit_optional(tmp_path):
    args = invoke_start_solver(tmp_path, '--seconds=45', '--lead-ms', '55')
    assert args[1:] == ['run', '--seconds', '45', '--lead-ms', '55']


def test_start_solver_forwards_capture_profile(tmp_path):
    args = invoke_start_solver(tmp_path, '--variant', 'deep-quiet')
    assert args[1:] == ['run', '--variant', 'deep-quiet']


def test_start_solver_accepts_explicit_recording_directory(tmp_path):
    directory = str(tmp_path / 'session')
    args = invoke_start_solver(tmp_path, '--recording', directory)
    assert args[-3:] == ['run', '--recording', directory]
    assert '--no-recording' not in args


def invoke_start_practice(tmp_path, *args):
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    capture_args = tmp_path / 'practice-args.txt'
    fake_xdg_open = fake_bin / 'xdg-open'
    fake_xdg_open.write_text('#!/bin/sh\nprintf \'%s\\n\' "$@" > "$PRACTICE_TEST_ARGS"\n')
    fake_xdg_open.chmod(0o755)
    env = os.environ.copy()
    env['PATH'] = str(fake_bin) + os.pathsep + env['PATH']
    env['PRACTICE_TEST_ARGS'] = str(capture_args)
    subprocess.run([str(PROJECT / 'start-practice.sh'), *args], env=env, check=True)
    return capture_args.read_text().splitlines()


def test_start_practice_opens_the_solver_bench(tmp_path):
    args = invoke_start_practice(tmp_path)
    assert args == [(PROJECT / 'simulator' / 'skillcheck.html').as_uri()]
