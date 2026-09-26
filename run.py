#!/usr/bin/env python3
"""Development entry point for the independent VD rewrite."""
import os
import sys
from pathlib import Path

if __name__=='__main__':
    venv=Path(__file__).resolve().parent/'.venv'
    python=venv/'bin/python'
    if python.exists() and Path(sys.prefix).resolve()!=venv.resolve():
        os.execv(str(python),[str(python),str(Path(__file__).resolve()),*sys.argv[1:]])

import argparse
from datetime import datetime
import json
import statistics
import time

from vd.capture import PortalCapture


def main():
    parser = argparse.ArgumentParser(description='VD rewrite: capture, dry-run, replay')
    parser.add_argument('mode', choices=['capture-probe','dry-run','run','replay','summary','preflight'])
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--seconds', type=float)
    parser.add_argument('--fps', type=int, default=60,
                        help='requested maximum portal frame rate; synthetic source uses fixed FPS')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--recording', type=Path, help='New output directory, or input for replay')
    parser.add_argument('--lead-ms', type=float, default=60)
    parser.add_argument('--lead-uncertainty-ms', type=float, default=15)
    parser.add_argument('--learn-lead', action='store_true',
                        help='Apply session-local CV latency estimates after four consistent observations')
    args = parser.parse_args()
    if args.mode=='summary':
        if args.recording is None:
            parser.error('summary requires --recording')
        from vd.recording import summarize_recording
        print(json.dumps(summarize_recording(args.recording),indent=2,ensure_ascii=False))
        return
    if args.mode=='preflight':
        from vd.preflight import inspect_system
        result=inspect_system()
        print(json.dumps(result,indent=2,ensure_ascii=False))
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        if not result['ok']:
            raise SystemExit(1)
        return
    if args.seconds is None:
        args.seconds=3600 if args.mode=='run' else 5
    if not 0 < args.seconds <= 86400:
        parser.error('seconds must be in (0, 86400]')
    if args.mode=='run' and args.synthetic:
        parser.error('run does not accept synthetic capture')
    if args.learn_lead and args.mode!='run':
        parser.error('--learn-lead requires run')
    if not 1<=args.fps<=240:
        parser.error('fps must be in [1, 240]')
    if not 0<=args.lead_ms<=300 or not 0<=args.lead_uncertainty_ms<=100:
        parser.error('lead must be in [0, 300] ms; uncertainty in [0, 100] ms')
    if args.mode!='capture-probe':
        from vd.runtime import run_session, replay
        if args.mode=='replay':
            if args.recording is None:
                parser.error('replay requires --recording')
            result=replay(args.recording)
        else:
            directory=args.recording or (Path(__file__).resolve().parent/'recordings'/
                         datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
            if args.mode=='run':
                print(f'RUN: Space при удержании LMB; lead={args.lead_ms:g} ms '
                      f'(CV-калибровка: {"включена" if args.learn_lead else "только наблюдение"}).',flush=True)
            result=run_session(seconds=args.seconds,synthetic=args.synthetic,fps=args.fps,
                           directory=directory,lead_seconds=args.lead_ms/1000,
                           lead_uncertainty=args.lead_uncertainty_ms/1000,
                           physical=args.mode=='run',learn_lead=args.learn_lead)
        print(json.dumps({k:v for k,v in result.items() if k not in ('rows','events')},indent=2))
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(result,indent=2)+'\n')
        return
    rows = []
    with PortalCapture(synthetic=args.synthetic, fps=args.fps) as capture:
        # Screen selection can take up to a minute; no input device is opened.
        frame = capture.next(timeout=65)
        if frame is None:
            raise RuntimeError('No first frame')
        until = time.monotonic() + args.seconds
        while time.monotonic() < until:
            rows.append(dict(sequence=frame.sequence, media_time=frame.media_time,
                received_time=frame.received_time, consumed_time=frame.consumed_time,
                timestamp_kind=frame.timestamp_kind, pts_ns=frame.pts_ns,
                negotiated_caps=frame.negotiated_caps))
            frame = capture.next(frame.sequence, timeout=2)
            if frame is None:
                raise RuntimeError('Capture stalled')
    gaps = [(b['received_time']-a['received_time'])*1000 for a,b in zip(rows,rows[1:])]
    ages = [(r['received_time']-r['media_time'])*1000 for r in rows if r['media_time'] is not None]
    result = dict(synthetic=args.synthetic, frames=len(rows),
        requested_fps=args.fps,
        negotiated_caps=sorted({r['negotiated_caps'] for r in rows}),
        delivery_gap_median_ms=statistics.median(gaps) if gaps else None,
        delivery_gap_max_ms=max(gaps) if gaps else None,
        mapped_media_age_median_ms=statistics.median(ages) if ages else None,
        mapped_media_age_max_ms=max(ages) if ages else None,
        game_render_age_measured=False,
        worker_exit_code=capture.proc.returncode,
        reader_stopped=not capture.reader.is_alive(),
        rows=rows)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped.')
    except (RuntimeError,ValueError,OSError) as exc:
        print('VD: '+str(exc),file=sys.stderr)
        raise SystemExit(1)
