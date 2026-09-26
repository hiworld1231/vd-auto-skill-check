"""Final dispatch gate for the single-owner runtime.

Locks publication of capture and mouse state only across the final check and
keydown. CV and recording must never run under these locks. This bounds races
with already published observations; it cannot make OS event delivery atomic.
"""
from contextlib import nullcontext
from dataclasses import asdict
import time


def dispatch(engine, capture, sequence, *, mouse=None, output=None, clock=time.monotonic):
    if output is not None and mouse is None:
        raise ValueError('Physical input requires a mouse monitor')
    with capture.cv:
        now=clock()
        if capture.stopping or capture.error or capture.proc.poll() is not None:
            engine.cancel('CAPTURE_LOST',now)
            return None
        frame=capture.latest
        if frame is None or frame.sequence!=sequence:
            engine.planner.invalidate('NEW_FRAME_PENDING')
            engine.reason='NEW_FRAME_PENDING'
            return None
        if output is not None and frame.synthetic:
            engine.cancel('SYNTHETIC_INPUT_FORBIDDEN',now)
            raise ValueError('Physical input is forbidden for synthetic capture')
        if frame.media_time is None or not -.002<=now-frame.media_time<=engine.planner.max_age:
            engine.planner.invalidate('STALE_FRAME')
            engine.reason='STALE_FRAME'
            return None
        with mouse.lock if mouse is not None else nullcontext():
            state=mouse.snapshot() if mouse is not None else None
            if state is not None and not state.healthy:
                engine.cancel('MOUSE_UNAVAILABLE',clock())
                return None
            held=state.held if state is not None else True
            plan=engine.poll(clock(),held=held,capture_alive=True)
            if plan is None:
                return None
            if output is not None:
                try:
                    keydown=output.pulse()
                except BaseException as exc:
                    engine.emit('INPUT_FAILED',clock(),error=str(exc),plan=asdict(plan))
                    engine.cancel('INPUT_FAILED',clock())
                    raise
                engine.emit('KEYDOWN',keydown.syn_completed_at,
                    requested_at=keydown.requested_at,plan=asdict(plan),
                    frame_at=frame.media_time,
                    prefire_motion=asdict(engine.motion.estimate) if engine.motion.estimate else None,
                    great=asdict(engine.planner.target),
                    good=asdict(engine.good) if engine.good else None,
                    dispatch_lag_ms=(keydown.syn_completed_at-plan.press_at)*1000,
                    outside_target_window=bool(keydown.syn_completed_at>plan.latest_press_at))
            return plan
