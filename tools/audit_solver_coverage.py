#!/usr/bin/env python3
"""Summarize solver-observed checks; does not infer game outcomes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def audit(directory):
    directory=Path(directory)
    events=[json.loads(line) for line in (directory/'events.jsonl').open()]
    begins=[e for e in events if e.get('kind')=='BEGIN']
    keys=[e for e in events if e.get('kind')=='KEYDOWN']
    endings=[e for e in events if e.get('kind')=='END']
    landings=[e for e in events if e.get('kind')=='CV_LANDING']
    pressed=Counter(e['generation'] for e in keys)
    return dict(recording=str(directory),detected_checks=len(begins),
        checks_with_keydown=sum(pressed[b['generation']]>0 for b in begins),
        keydowns=len(keys),duplicate_keydowns=sum(n-1 for n in pressed.values() if n>1),
        unpressed_generations=[b['generation'] for b in begins if not pressed[b['generation']]],
        end_reasons=dict(Counter(e.get('reason') for e in endings)),
        targets=dict(Counter(e.get('plan',{}).get('target_grade') for e in keys)),
        cv_landings=dict(Counter(e.get('landing',{}).get('label') for e in landings)),
        game_outcomes_measured=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings',nargs='+',type=Path)
    args=parser.parse_args()
    print(json.dumps([audit(p) for p in args.recordings],indent=2))
