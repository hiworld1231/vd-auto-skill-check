#!/usr/bin/env python3
"""Compare embedded event attachments/subtitles with each MKV video timeline.

This is a provenance audit only: it does not guess clock offsets or relabel
game outcomes. A mismatch means the session must not be used as synchronized
ground truth without an independently verified alignment.
"""
import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def run(args):
    return subprocess.run(args, check=True, capture_output=True)


def ass_seconds(value):
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[.](\d{2}|\d{3})", value.strip())
    if not m:
        return None
    h, mi, sec, frac = m.groups()
    scale = 100 if len(frac) == 2 else 1000
    return int(h) * 3600 + int(mi) * 60 + int(sec) + int(frac) / scale


def duration_seconds(value):
    if not value:
        return None
    m = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[.](\d+)", value)
    if not m:
        return None
    h, mi, sec, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(sec) + float("0." + frac)


def inspect(path):
    probe = json.loads(run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)
    ]).stdout)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    subtitles = [s for s in streams if s.get("codec_type") == "subtitle"]
    attachments = [s for s in streams if s.get("codec_type") == "attachment"]
    video_duration = (float(video.get("duration")) if video and video.get("duration") else
                      duration_seconds(video.get("tags", {}).get("DURATION")) if video else None)
    if video_duration is None and video:
        video_duration = float(probe.get("format", {}).get("duration", 0) or 0)
    sub_events = []
    for index, stream in enumerate(subtitles):
        data = run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
            "-map", f"0:{stream['index']}", "-f", "ass", "-"
        ]).stdout.decode("utf-8", "replace")
        for line in data.splitlines():
            if not line.startswith("Dialogue:"):
                continue
            cols = line.split(",", 9)
            if len(cols) < 10:
                continue
            start = ass_seconds(cols[1])
            end = ass_seconds(cols[2])
            text = cols[9].strip()
            event = re.search(r"\b(LMB_DOWN|LMB_UP|SPACE_DOWN|SPACE_UP|LABEL_WHITE|LABEL_BLACK)\b", text)
            if start is not None and event:
                raw_time = re.match(r"([0-9]+(?:\.[0-9]+)?)\s+", text)
                sub_events.append({
                    "stream": index, "start_s": start, "end_s": end,
                    "event": event.group(1),
                    "text_timestamp_s": float(raw_time.group(1)) if raw_time else None,
                })
    attachment_events = []
    for stream in attachments:
        if stream.get("tags", {}).get("filename") != "events.tsv":
            continue
        with tempfile.TemporaryDirectory(prefix="vd-event-audit-") as tmp:
            dest = Path(tmp) / "events.tsv"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                f"-dump_attachment:t:{stream.get('id', 0)}", str(dest),
                "-i", str(path), "-f", "null", "-"
            ], check=True, capture_output=True)
            if dest.exists():
                for line in dest.read_text(encoding="utf-8", errors="replace").splitlines()[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            attachment_events.append({"time_s": float(parts[0]), "event": parts[1]})
                        except ValueError:
                            pass
    sub_space = [e for e in sub_events if e["event"] == "SPACE_DOWN"]
    att_space = [e for e in attachment_events if e["event"] == "SPACE_DOWN"]
    timestamp_text_ok = all(e["text_timestamp_s"] is None or abs(e["text_timestamp_s"] - e["start_s"]) <= 0.02 for e in sub_events)
    event_sequence_ok = [e["event"] for e in sorted(sub_events, key=lambda x: x["start_s"])] == [
        e["event"] for e in sorted(attachment_events, key=lambda x: x["time_s"])
    ]
    pairs = []
    for i in range(max(len(sub_space), len(att_space))):
        s = sub_space[i] if i < len(sub_space) else None
        a = att_space[i] if i < len(att_space) else None
        pairs.append({
            "ordinal": i + 1,
            "subtitle_start_s": s["start_s"] if s else None,
            "subtitle_text_timestamp_s": s["text_timestamp_s"] if s else None,
            "attachment_time_s": a["time_s"] if a else None,
            "subtitle_attachment_delta_s": round(s["start_s"] - a["time_s"], 6) if s and a else None,
        })
    subtitles_with_events = [e for e in sub_events]
    subtitle_duration = max((e["end_s"] for e in subtitles_with_events), default=None)
    in_video = all(-0.02 <= e["start_s"] <= video_duration + 0.02 for e in sub_events) if video_duration is not None else False
    attachment_in_video = all(-0.02 <= e["time_s"] <= video_duration + 0.02 for e in attachment_events) if video_duration is not None else False
    aligned = bool(sub_space and att_space and event_sequence_ok and timestamp_text_ok and in_video and attachment_in_video)
    return {
        "file": str(path), "video_duration_s": video_duration,
        "subtitle_duration_s": subtitle_duration,
        "subtitle_event_count": len(sub_events), "attachment_event_count": len(attachment_events),
        "space_down_counts": {"subtitles": len(sub_space), "attachment": len(att_space)},
        "subtitle_text_timestamps_match_cue_times": timestamp_text_ok,
        "all_event_sequence_matches": event_sequence_ok,
        "all_event_times_inside_video": {"subtitles": in_video, "attachment": attachment_in_video},
        "synchronized_ground_truth_usable": aligned,
        "space_comparison": pairs,
        "attachment_events": attachment_events,
        "subtitle_events": sub_events,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    sessions = []
    for path in sorted(args.root.rglob("*.mkv")):
        try:
            sessions.append(inspect(path))
        except (subprocess.CalledProcessError, OSError, ValueError, KeyError) as exc:
            sessions.append({"file": str(path), "audit_error": f"{type(exc).__name__}: {exc}"})
    report = {
        "method": "Compare decoded ASS cue starts and embedded events.tsv timestamps, event order, and bounds against the video-stream duration. This is a strict timeline consistency gate and does not infer a corrective offset.",
        "sessions": sessions,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sessions": len(sessions), "usable": sum(bool(s.get("synchronized_ground_truth_usable")) for s in sessions), "report": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
