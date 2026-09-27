"""Independent capture/CV/engine integration. No physical input in dry-run."""
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

from vd.capture import PortalCapture
from vd.calibration import FreezeObserver, LeadEstimator
from vd.motion import Motion
from vd.engine import Engine
from vd.dispatch import dispatch
from vd.recording import Recorder, read_recording
from vd.vision import Arc, Detector, retained_target


class _NullRecorder:
    """Keep the live decision loop without video encoding or disk writes."""
    def __init__(self):
        self.metadata = {}
        self.dropped = 0

    def event(self, event):
        pass

    def submit(self, *args, **kwargs):
        pass


def _timing_summary(samples):
    ordered = sorted(samples)
    if not ordered:
        return {"samples": 0, "p50": None, "p95": None, "max": None}
    p95_index = min(len(ordered) - 1, int(len(ordered) * .95))
    return {"samples": len(ordered), "p50": round(statistics.median(ordered), 3),
            "p95": round(ordered[p95_index], 3), "max": round(ordered[-1], 3)}


def dry_run(*, seconds, synthetic, fps, directory, lead_seconds, lead_uncertainty,
            capture_priority=5, variant='baseline'):
    return run_session(seconds=seconds,synthetic=synthetic,fps=fps,directory=directory,
                       lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty,
                       physical=False,capture_priority=capture_priority,variant=variant)


def run_session(*, seconds, synthetic, fps, directory, lead_seconds, lead_uncertainty,
                physical=False, learn_lead=False, recording=True,
                capture_priority=5, variant='baseline'):
    if physical and synthetic:
        raise ValueError('Physical input is forbidden for synthetic capture')
    if learn_lead and not physical:
        raise ValueError('Learning requires physical input observations')
    engine=Engine(lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty)
    detector=Detector()
    reasons=Counter()
    nframes=0
    skipped=0
    last=-1
    presses=0
    keydowns=0
    mouse=None
    output=None
    mode='run' if physical else 'dry-run'
    observer=None
    observer_generation=None
    estimator=LeadEstimator(lead_seconds,lead_uncertainty)
    frame_processing_ms=deque(maxlen=600)
    frame_delivery_gap_ms=deque(maxlen=600)
    frame_age_ms=deque(maxlen=600)
    capture_pipe_age_ms=deque(maxlen=600)
    previous_received_time=None
    last_idle_recorded_at=float('-inf')
    started_wall=time.monotonic()
    started_cpu=time.process_time()
    with ExitStack() as stack:
        if recording:
            recorder=stack.enter_context(Recorder(directory,metadata=dict(
                mode=mode,synthetic=synthetic,variant=variant,requested_fps=fps,
                capture_priority=capture_priority,
                lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty,
                learn_lead=learn_lead,
                video_sampling='full_rate_during_check_1fps_while_idle',
                held_source='evdev' if physical else 'assumed_for_dry_run',
                game_outcome_measured=False)))
        else:
            recorder=_NullRecorder()

        def events():
            nonlocal presses,keydowns,observer,observer_generation
            for event in engine.take_events():
                event['physical_input']=(None if event['kind']=='INPUT_FAILED'
                                         else event['kind']=='KEYDOWN')
                recorder.event(event)
                if event['kind']=='PRESS_CLAIM':
                    presses+=1
                if event['kind']=='KEYDOWN':
                    keydowns+=1
                    if event['prefire_motion'] is not None:
                        observer=FreezeObserver(motion=Motion(**event['prefire_motion']),
                            press_at=event['at'],frame_at=event['frame_at'],
                            great=Arc(**event['great']),
                            good=Arc(**event['good']) if event['good'] else None,
                            chain=event['chain'],late_dispatch=event['outside_target_window'])
                        observer_generation=event['generation']
                print(json.dumps(event),flush=True)
                if event['kind']=='END' and observer is not None and observer_generation==event['generation']:
                    landing=observer.finish(event['reason'])
                    updated=estimator.observe(landing,at=event['at'],physical=physical)
                    diagnostic=dict(kind='CV_LANDING',at=event['at'],generation=event['generation'],
                        physical_input=False,game_outcome_measured=False,landing=asdict(landing),
                        calibration_reason=estimator.reason,calibration_applied=updated and learn_lead,
                        estimated_lead=estimator.lead,estimated_uncertainty=estimator.uncertainty)
                    recorder.event(diagnostic)
                    print(json.dumps(diagnostic),flush=True)
                    if updated and learn_lead:
                        engine.planner.lead=estimator.lead
                        engine.planner.lead_uncertainty=estimator.uncertainty
                    observer=None

        try:
            capture=stack.enter_context(PortalCapture(synthetic=synthetic,fps=fps,
                                                     priority=capture_priority))
            frame=capture.next(timeout=65)
            if frame is None:
                raise RuntimeError('No first frame')
            started_wall=time.monotonic()
            started_cpu=time.process_time()
            if physical:
                try:
                    from vd.input import MouseMonitor, SpaceOutput
                except ImportError as exc:
                    raise RuntimeError(
                        'Physical input requires the evdev package; '
                        'dry-run and replay remain available.'
                    ) from exc
                mouse=stack.enter_context(MouseMonitor())
                output=stack.enter_context(SpaceOutput())
            until=None if seconds is None else time.monotonic()+seconds
            while until is None or time.monotonic()<until:
                if output is not None and output.error:
                    raise RuntimeError('Input failed: '+output.error)
                held=mouse.snapshot().held if mouse is not None else True
                if frame is not None:
                    frame_started=time.perf_counter()
                    if previous_received_time is not None:
                        frame_delivery_gap_ms.append(
                            max(0,(frame.received_time-previous_received_time)*1000))
                    previous_received_time=frame.received_time
                    frame_age_ms.append(max(0,(time.monotonic()-frame.received_time)*1000))
                    if frame.media_time is not None:
                        capture_pipe_age_ms.append(max(0,(frame.received_time-frame.media_time)*1000))
                    skipped+=max(0,frame.sequence-last-1) if last>=0 else 0
                    last=frame.sequence
                    if not held:
                        engine.cancel('LMB_RELEASED',time.monotonic())
                    elif frame.media_time is None:
                        engine.cancel('NO_MEDIA_TIMESTAMP',time.monotonic())
                    else:
                        m=detector.measure(frame.image,frame.media_time,center_hint=engine.center)
                        m=retained_target(m,great=engine.target,good=engine.good,center=engine.center)
                        now=time.monotonic()
                        if observer is not None:
                            if not -.002<=now-m.timestamp<=engine.planner.max_age:
                                observer.tainted='STALE_FRAME'
                            observer.feed(m)
                        engine.observe(m,now=now,held=held)
                        dispatch(engine,capture,last,mouse=mouse,output=output,clock=time.monotonic)
                    decision_time=time.monotonic()
                    state=engine.snapshot()
                    if held and frame.media_time is not None:
                        state['measurement']=asdict(m)
                    reasons[state['reason']]+=1
                    events()
                    if (held and engine.active
                            or frame.received_time-last_idle_recorded_at>=1):
                        recorder.submit(frame,decision_time=decision_time,held=held,
                                        state=state,events=[])
                        if not held or not engine.active:
                            last_idle_recorded_at=frame.received_time
                    frame_processing_ms.append((time.perf_counter()-frame_started)*1000)
                    nframes+=1
                else:
                    dispatch(engine,capture,last,mouse=mouse,output=output,clock=time.monotonic)
                    events()
                now=time.monotonic()
                plan=engine.planner.current
                timeout=.020 if until is None else min(.020,max(0,until-now))
                if plan is not None:
                    timeout=min(timeout,max(0,plan.press_at-now),max(0,plan.valid_until-now))
                frame=capture.next(last,timeout=timeout)
        except BaseException as exc:
            recorder.metadata['runtime_error']=str(exc) or type(exc).__name__
            raise
        finally:
            engine.cancel('STOPPED',time.monotonic())
            events()
            recorder.metadata['capture_sequences_skipped']=skipped
            elapsed=max(time.monotonic()-started_wall,1e-9)
            performance=dict(
                frame_processing_ms=_timing_summary(frame_processing_ms),
                frame_delivery_gap_ms=_timing_summary(frame_delivery_gap_ms),
                frame_age_ms=_timing_summary(frame_age_ms),
                capture_pipe_age_ms=_timing_summary(capture_pipe_age_ms),
                elapsed_seconds=round(elapsed,3),
                main_cpu_percent=round(100*(time.process_time()-started_cpu)/elapsed,1),
                profile_window_frames=frame_processing_ms.maxlen)
            recorder.metadata['performance']=performance
            recorder.metadata['final_lead_estimate']=dict(lead=estimator.lead,
                uncertainty=estimator.uncertainty,reason=estimator.reason,
                samples=estimator.samples,applied=learn_lead)
            if not recording:
                print(json.dumps(dict(kind='PERFORMANCE',**recorder.metadata['performance']),
                                 ensure_ascii=False),flush=True)
    return dict(mode=mode,frames=nframes,claims=presses,physical_presses=keydowns,
                capture_sequences_skipped=skipped,reasons=dict(reasons),
                performance=performance,
                recording=str(directory) if recording else None,
                recording_frames_dropped=recorder.dropped)


def replay(directory):
    manifest=json.loads((Path(directory)/'manifest.json').read_text())
    engine=Engine(lead_seconds=manifest['lead_seconds'],
                  lead_uncertainty=manifest['lead_uncertainty'])
    detector=Detector()
    rows=[]
    events=[]
    previous_decision=None
    previous_held=True
    for row,image in read_recording(directory):
        now=row['decision_time']
        if previous_decision is not None and now<previous_decision:
            raise ValueError('Non-monotonic recorded decision time')
        plan=engine.planner.current
        if plan is not None and plan.press_at<now:
            wake=max(previous_decision,plan.press_at)
            engine.poll(wake,held=previous_held)
            events.extend(engine.take_events())
        if manifest.get('learn_lead'):
            settings=row['state']
            lead=float(settings['lead_seconds'])
            uncertainty=float(settings['lead_uncertainty'])
            if not 0<=lead<=.300 or not 0<=uncertainty<=.100:
                raise ValueError('Invalid recorded lead settings')
            engine.planner.lead=lead
            engine.planner.lead_uncertainty=uncertainty
        timestamp=row['media_time']
        if timestamp is None:
            engine.cancel('NO_MEDIA_TIMESTAMP',now)
        else:
            m=detector.measure(image,timestamp,center_hint=engine.center)
            m=retained_target(m,great=engine.target,good=engine.good,center=engine.center)
            engine.observe(m,now=now,held=row['held'])
            engine.poll(now,held=row['held'])
        events.extend(engine.take_events())
        rows.append(dict(sequence=row['sequence'],**engine.snapshot()))
        previous_decision=now
        previous_held=row['held']
    if previous_decision is not None:
        engine.cancel('REPLAY_ENDED',previous_decision)
        events.extend(engine.take_events())
    return dict(mode='replay',game_outcome_measured=False,
                counterfactual_after_claim=True,idealized_timer=True,
                input_timing_mode='frame_samples_only',
                lead_schedule='recorded_frame_settings' if manifest.get('learn_lead') else 'fixed_initial',
                source_recording_complete=(manifest.get('complete',False)
                    and manifest.get('capture_sequences_skipped',0)==0),
                source_capture_sequences_skipped=manifest.get('capture_sequences_skipped',0),
                frames=len(rows),claims=sum(e['kind']=='PRESS_CLAIM' for e in events),
                reasons=dict(Counter(r['reason'] for r in rows)),events=events,rows=rows)
