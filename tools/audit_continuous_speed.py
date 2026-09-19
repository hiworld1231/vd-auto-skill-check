#!/usr/bin/env python3
"""Audit old replay JSONs: old runtime speed vs continuous recent-angle fit before FIRE.
No counterfactual outcome claims are made."""
import argparse,glob,json,math,statistics,os

def unwrap(prev,a):
    if prev is None:return a
    d=(a-(prev%360)+540)%360-180
    if d<-20:d+=360
    return prev+d

def uniq(frames):
    out=[];u=None
    for f in frames:
        if 'needle_angle' not in f:continue
        u2=unwrap(u,float(f['needle_angle'])); t=float(f['time_rel_ms'])/1000
        if out and abs(u2-out[-1][1])<.35:continue
        out.append((t,u2,f));u=u2
    return out

def fit(pts):
    ss=[]
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            dt=pts[j][0]-pts[i][0]
            if dt>=.008:
                s=(pts[j][1]-pts[i][1])/dt
                if 20<s<1500:ss.append(s)
    return statistics.median(ss) if ss else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path'); ap.add_argument('--lead-ms',type=int,default=100)
    a=ap.parse_args(); rows=[]
    files=glob.glob(os.path.join(a.path,'check_*.json'))
    for p in files:
        d=json.load(open(p));fs=d.get('frames',[])
        ti=next((i for i,f in enumerate(fs) if f.get('timing',{}).get('scheduler_diag')),None)
        if ti is None:continue
        tt=fs[ti]['time_rel_ms']/1000; pts=uniq(fs[:ti+1]); ref=fit(pts[-10:]) if len(pts)>=5 else None
        if not ref:continue
        cutoff=tt-a.lead_ms/1000; pp=[x for x in pts if x[0]<=cutoff]
        cont=fit(pp[-10:]) if len(pp)>=3 else None
        raw=[f for f in fs[:ti+1] if 'needle_angle' in f and f['time_rel_ms']/1000<=cutoff]
        old=float(raw[-1]['speed_deg_s']) if raw and raw[-1].get('speed_deg_s') is not None else None
        if cont is not None and old is not None: rows.append((abs(cont-ref),abs(old-ref)))
    print('checks',len(rows),'offset_ms',a.lead_ms)
    if rows:
        print('continuous_median_abs_speed_error',round(statistics.median(x for x,_ in rows),2))
        print('old_runtime_median_abs_speed_error',round(statistics.median(y for _,y in rows),2))
if __name__=='__main__':main()
