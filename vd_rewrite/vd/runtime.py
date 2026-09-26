"""Independent capture/CV/engine integration. No physical input in dry-run."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import time

from vd.capture import PortalCapture
from vd.calibration import FreezeObserver, LeadEstimator
from vd.motion import Motion
from vd.engine import Engine
from vd.dispatch import dispatch
from vd.input import MouseMonitor, SpaceOutput
from vd.recording import Recorder, read_recording
from vd.vision import Arc, Detector, retained_target


def dry_run(*, seconds, synthetic, fps, directory, lead_seconds, lead_uncertainty):
    return run_session(seconds=seconds,synthetic=synthetic,fps=fps,directory=directory,
                       lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty,
                       physical=False)


def run_session(*, seconds, synthetic, fps, directory, lead_seconds, lead_uncertainty,
                physical=False, learn_lead=False):
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
    with ExitStack() as stack:
        recorder=stack.enter_context(Recorder(directory,metadata=dict(
            mode=mode,synthetic=synthetic,requested_fps=fps,
            lead_seconds=lead_seconds,lead_uncertainty=lead_uncertainty,
            learn_lead=learn_lead,
            held_source='evdev' if physical else 'assumed_for_dry_run',
            game_outcome_measured=False)))

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
            capture=stack.enter_context(PortalCapture(synthetic=synthetic,fps=fps))
            frame=capture.next(timeout=65)
            if frame is None:
                raise RuntimeError('No first frame')
            if physical:
                mouse=stack.enter_context(MouseMonitor())
                output=stack.enter_context(SpaceOutput())
            until=time.monotonic()+seconds
            while time.monotonic()<until:
                if output is not None and output.error:
                    raise RuntimeError('Input failed: '+output.error)
                held=mouse.snapshot().held if mouse is not None else True
                if frame is not None:
                    skipped+=max(0,frame.sequence-last-1) if last>=0 else 0
                    last=frame.sequence
                    if frame.media_time is None:
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
                    if frame.media_time is not None:
                        state['measurement']=asdict(m)
                    reasons[state['reason']]+=1
                    events()
                    recorder.submit(frame,decision_time=decision_time,held=held,
                                    state=state,events=[])
                    nframes+=1
                else:
                    dispatch(engine,capture,last,mouse=mouse,output=output,clock=time.monotonic)
                    events()
                now=time.monotonic()
                plan=engine.planner.current
                timeout=min(.020,max(0,until-now))
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
            recorder.metadata['final_lead_estimate']=dict(lead=estimator.lead,
                uncertainty=estimator.uncertainty,reason=estimator.reason,
                samples=estimator.samples,applied=learn_lead)
    return dict(mode=mode,frames=nframes,claims=presses,physical_presses=keydowns,
                capture_sequences_skipped=skipped,reasons=dict(reasons),recording=str(directory),
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
