#!/usr/bin/env python3
"""Scan legacy program-annotated diagnostic montages, including MP4-less replays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vd.vision import Detector, retained_target


def center_error(a, b):
    return abs((a-b+180) % 360 - 180)


def read_expected(path):
    jp=path.with_name(path.name.removesuffix("_diagnostic.png")+".json")
    if not jp.exists():
        return None, None
    try:
        d=json.loads(jp.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError):
        return None, None
    z=d.get("locked_zones") or {"white":d.get("locked_w"),"black":d.get("locked_b")}
    a=z.get("white") or z.get("w")
    b=z.get("black") or z.get("b")
    if not isinstance(a,dict) or not isinstance(b,dict):
        return d,None
    return d,(a,b)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots",nargs="+",type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    images=sorted({p.resolve() for root in args.roots if root.exists()
                   for p in root.rglob("*_diagnostic.png") if p.is_file()})
    detector=Detector()
    rows=[]
    for n,path in enumerate(images,1):
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        image=cv2.imdecode(__import__('numpy').frombuffer(raw,dtype='uint8'),cv2.IMREAD_COLOR)
        if image is None:
            rows.append({"path":str(path),"sha256":digest,"layout":"UNREADABLE_IMAGE"})
            continue
        if image.shape[0]!=240 or image.shape[1]%320:
            # Several early diagnostic sheets are five labeled 240×224 panels,
            # not horizontal strips of raw 320×240 detector frames.
            rows.append({"path":str(path),"sha256":digest,
                         "layout":"COMPOSITED_DIAGNOSTIC_PANELS",
                         "width":image.shape[1],"height":image.shape[0]})
            continue
        d,expected=read_expected(path)
        frames=image.shape[1]//320
        measured=[]
        center_hint=great_hint=good_hint=None
        for i in range(frames):
            crop=image[:,i*320:(i+1)*320]
            m=detector.measure(crop,i/60,center_hint=center_hint)
            if m.center is not None:center_hint=m.center
            m=retained_target(m,great=great_hint,good=good_hint,center=center_hint)
            if m.great is not None and m.good is not None:
                measured.append((m.great,m.good))
                great_hint,good_hint=m.great,m.good
        row={"path":str(path),"sha256":digest,"frame_tiles":frames,
             "detected_paired_tiles":len(measured),
             "outcome":((d.get("evaluation") or d.get("outcome_info") or {}).get("outcome") if d else None)}
        if expected and measured:
            a,b=expected
            ac=float(a.get("center",(float(a.get("start",0))+float(a.get("end",0)))/2))
            bc=float(b.get("center",(float(b.get("start",0))+float(b.get("end",0)))/2))
            aw=float(a.get("width",float(a.get("end",0))-float(a.get("start",0))))
            bw=float(b.get("width",float(b.get("end",0))-float(b.get("start",0))))
            first=measured[0]
            row["zone_comparison"]={
                "great_center_error_deg":round(center_error(first[0].center,ac),3),
                "great_width_error_deg":round(first[0].width-aw,3),
                "good_center_error_deg":round(center_error(first[1].center,bc),3),
                "good_width_error_deg":round(first[1].width-bw,3),
            }
        rows.append(row)
        if n%100==0:print(f"scanned {n}/{len(images)} diagnostic montages",flush=True)
    compared=[r for r in rows if "zone_comparison" in r]
    non_frame=[r for r in rows if "layout" in r]
    report={"diagnostic_images":len(images),"compared_to_json":len(compared),
            "composited_panels_not_raw_frames":len(non_frame),
            "panel_layout_distribution":{
                f"{w}x{h}":sum(1 for r in non_frame if r.get("width")==w and r.get("height")==h)
                for w,h in sorted({(r.get("width"),r.get("height")) for r in non_frame})},
            "non_frame_layouts":non_frame,"results":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"diagnostic_images":len(images),"compared_to_json":len(compared),
                      "composited_panels_not_raw_frames":len(non_frame),
                      "panel_layout_distribution":report["panel_layout_distribution"]},
                     ensure_ascii=False,indent=2))


if __name__=="__main__":main()
