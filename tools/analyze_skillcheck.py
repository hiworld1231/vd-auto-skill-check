#!/usr/bin/env python3
import argparse
import fcntl
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT.parent / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
if VENV_PY.exists() and sys.prefix != str(VENV_DIR.resolve()):
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT_ALL = ROOT / "analysis_all.json"
DATASET_PATH = ROOT / "skillcheck_dataset.jsonl"
MERGE_LOCK = ROOT / ".analysis_merge.lock"
TEMPLATE_PATH = ROOT / "space_template.png"

# Default angular speed in Violent District (deg/s) (calibrated across all 49 sessions)
DEFAULT_SPEED_DEG_S = 270.3
RING_RADIUS = 66.5


def load_template():
    if not TEMPLATE_PATH.exists():
        scratch_template = Path("/home/oae/.gemini/antigravity/brain/7ae406cc-d8eb-49cc-8959-fe1a4cf97b09/scratch/space_template.png")
        if scratch_template.exists():
            import shutil
            shutil.copy(str(scratch_template), str(TEMPLATE_PATH))
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")
    return cv2.imread(str(TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)


SPACE_TEMPLATE = load_template()
TEMPLATE_H, TEMPLATE_W = SPACE_TEMPLATE.shape


def newest_session():
    files = [x for x in os.listdir(ROOT) if re.fullmatch(r"session_\d+\.mkv", x)]
    if not files:
        raise FileNotFoundError("Не найден session_*.mkv")
    return str(ROOT / max(files, key=lambda x: int(re.search(r"\d+", x).group())))


def parse_ass_time(value):
    m = re.match(r"^(\d+):(\d+):(\d+)[.:](\d+)$", value.strip())
    if not m:
        return None
    h, mi, s, fraction = m.groups()
    if len(fraction) == 2:
        return int(h) * 3600 + int(mi) * 60 + int(s) + int(fraction) / 100.0
    elif len(fraction) == 3:
        return int(h) * 3600 + int(mi) * 60 + int(s) + int(fraction) / 1000.0
    return int(h) * 3600 + int(mi) * 60 + float(f"{s}.{fraction}")


def parse_events(path):
    events = []
    # 1. Try attachment events.tsv
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-dump_attachment:t:0", "pipe:1", "-i", str(path), "-y"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if p.returncode == 0 and p.stdout:
            for line in p.stdout.splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    try:
                        t = float(parts[0])
                        ev = parts[1]
                        events.append((t, ev))
                    except ValueError:
                        pass
    except Exception:
        pass

    if events:
        return sorted(events, key=lambda x: x[0])

    # 2. Fallback: Parse dialogue z[9] (uncorrupted float)
    try:
        q = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:s:0", "-f", "ass", "-"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in q.stdout.splitlines():
            if not line.startswith("Dialogue:"):
                continue
            z = line.split(",", 9)
            if len(z) < 10:
                continue
            m = re.match(r"^\s*([0-9.]+)\s+(.*)$", z[9])
            if m:
                try:
                    t = float(m.group(1))
                    ev = m.group(2)
                    events.append((t, ev))
                    continue
                except ValueError:
                    pass
            t = parse_ass_time(z[1])
            if t is not None:
                k = re.search(r"(LMB_DOWN|LMB_UP|SPACE_DOWN|SPACE_UP|LABEL_WHITE|LABEL_BLACK)", z[9])
                if k:
                    events.append((t, k.group(1)))
    except Exception:
        pass

    return sorted(events, key=lambda x: x[0])


def associate_labels(events_list):
    checks = []
    pending = []
    orphans = []
    for t, kind in events_list:
        if kind == "SPACE_DOWN":
            pending.append({"space_time": t})
        elif kind in {"LABEL_WHITE", "LABEL_BLACK"}:
            if pending:
                check = pending.pop(0)
                check["label"] = "WHITE" if kind == "LABEL_WHITE" else "BLACK"
                check["label_time"] = t
                check["label_delay"] = t - check["space_time"]
                checks.append(check)
            else:
                orphans.append({"time": t, "label": "WHITE" if kind == "LABEL_WHITE" else "BLACK"})
    checks_by_time = {x["space_time"]: x for x in checks}
    result = []
    for t, kind in events_list:
        if kind == "SPACE_DOWN":
            result.append(checks_by_time.get(t, {"space_time": t, "label": None, "label_time": None, "label_delay": None}))
    return result, orphans


def probe_video(path):
    q = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames", "-of", "json", str(path)],
        capture_output=True,
        text=True
    )
    try:
        data = json.loads(q.stdout)["streams"][0]
        a, b = data["r_frame_rate"].split("/")
        data["fps"] = float(a) / float(b)
        return data
    except Exception:
        return {"width": 320, "height": 240, "fps": 60.0, "duration": "0.0"}


def detect_frame(frame_gray, frame_bgr, expected_angle=None, search_window=35.0):
    # Fast center ROI match (0.08ms vs 1.18ms)
    roi_y0, roi_y1 = 110, 215
    roi_x0, roi_x1 = 100, 220
    crop = frame_gray[roi_y0:roi_y1, roi_x0:roi_x1]
    res = cv2.matchTemplate(crop, SPACE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
    _, max_v, _, max_l = cv2.minMaxLoc(res)

    if max_v >= 0.80:
        cx = float(roi_x0 + max_l[0] + TEMPLATE_W / 2.0)
        cy = float(roi_y0 + max_l[1] + TEMPLATE_H / 2.0)
    else:
        # Full frame fallback if outside center region
        res_full = cv2.matchTemplate(frame_gray, SPACE_TEMPLATE, cv2.TM_CCOEFF_NORMED)
        _, max_v, _, max_l = cv2.minMaxLoc(res_full)
        if max_v < 0.80:
            return None
        cx = float(max_l[0] + TEMPLATE_W / 2.0)
        cy = float(max_l[1] + TEMPLATE_H / 2.0)

    # Polar unroll: cols=radius (0 to 78), rows=angle (0 to 360)
    max_r = 78
    polar = cv2.warpPolar(frame_bgr, (max_r, 360), (cx, cy), max_r, cv2.WARP_POLAR_LINEAR)

    # Needle detection (radial ray from r=22 to r=64)
    needle_crop = polar[:, 22:64]
    r_ch = needle_crop[:, :, 2].astype(np.float32)
    g_ch = needle_crop[:, :, 1].astype(np.float32)
    b_ch = needle_crop[:, :, 0].astype(np.float32)
    redness = r_ch - np.maximum(g_ch, b_ch)
    red_profile = np.mean(np.maximum(redness, 0), axis=1)

    # Find all local peaks to avoid background noise/sparks hijacking the needle
    peaks = []
    for i in range(360):
        prev_v = red_profile[(i - 1) % 360]
        curr_v = red_profile[i]
        next_v = red_profile[(i + 1) % 360]
        if curr_v >= prev_v and curr_v > next_v and curr_v > 15.0:
            peaks.append((i, curr_v))

    if expected_angle is not None and peaks:
        cand = [p for p in peaks if abs((p[0] - expected_angle + 180) % 360 - 180) <= search_window]
        if cand:
            peak_idx = max(cand, key=lambda x: x[1])[0]
        else:
            peak_idx = int(np.argmax(red_profile))
    else:
        peak_idx = int(np.argmax(red_profile))

    needle_strength = float(red_profile[peak_idx])

    # Sub-degree parabolic peak refinement
    p_prev = float(red_profile[(peak_idx - 1) % 360])
    p_curr = needle_strength
    p_next = float(red_profile[(peak_idx + 1) % 360])
    denom = 2.0 * (2.0 * p_curr - p_prev - p_next)
    if denom > 1e-5:
        delta = (p_next - p_prev) / denom
        needle_angle = float((peak_idx + delta) % 360.0)
    else:
        needle_angle = float(peak_idx)

    # Success zone detection on outer ring (r=64 to r=68 for robust SNR)
    ring_crop = polar[:, 64:68].astype(np.float32)
    r66_b = np.mean(ring_crop[:, :, 0], axis=1)
    r66_g = np.mean(ring_crop[:, :, 1], axis=1)
    r66_r = np.mean(ring_crop[:, :, 2], axis=1)
    r66_val = (r66_b + r66_g + r66_r) / 3.0

    # Great zone: solid glowing neon white segment along the ring
    white_mask = (r66_val > 220) & (r66_r > 200) & (r66_g > 200) & (r66_b > 200)
    # Good zone: solid dark segment along the ring
    black_mask = (r66_val < 35)

    return {
        "confidence": float(max_v),
        "cx": cx,
        "cy": cy,
        "needle_angle": needle_angle,
        "needle_strength": needle_strength,
        "white_mask": white_mask,
        "black_mask": black_mask,
        "r66_val": r66_val,
    }


def find_contiguous_segment(mask, min_len=4):
    if mask is None:
        return None
    doubled = np.concatenate([mask, mask])
    runs = []
    start = None
    for i in range(len(doubled)):
        if doubled[i] and start is None:
            start = i
        elif not doubled[i] and start is not None:
            length = i - start
            if min_len <= length <= 90:
                runs.append((start % 360, (i - 1) % 360, length))
            start = None
    if start is not None:
        length = len(doubled) - start
        if min_len <= length <= 90:
            runs.append((start % 360, (len(doubled) - 1) % 360, length))

    if not runs:
        return None
    runs.sort(key=lambda x: x[2], reverse=True)
    return runs[0]


def extract_zones(det, locked_zones=None):
    """
    Extracts high-precision Great (white) and Good (black) zones.
    Physical facts in Violent District:
    1. Needle spawns at 270° and rotates clockwise.
    2. Skill check zones strictly spawn between 10° and 235°.
    3. The Great (white) zone is glowing neon white, physical width ~8°..11°.
    4. The Good (black) zone is dark, physical width ~40°..44°, immediately following Great zone.
    5. The HUD is stationary during a check: once locked with high confidence,
       zone coordinates remain locked to eliminate noise or needle occlusion.
    """
    if det is None:
        return locked_zones or (None, None)
    if locked_zones and locked_zones[0] is not None:
        return locked_zones

    white_mask = det.get("white_mask")
    black_mask = det.get("black_mask")
    r66_val = det.get("r66_val")
    w_dict = None
    b_dict = None

    # Step 1: Search for white segment (Great zone) inside valid spawn arc [10°, 235°]
    if white_mask is not None:
        doubled = np.concatenate([white_mask, white_mask])
        runs = []
        start = None
        for i in range(len(doubled)):
            if doubled[i] and start is None:
                start = i
            elif not doubled[i] and start is not None:
                length = i - start
                s_deg = start % 360
                e_deg = (i - 1) % 360
                if 5 <= length <= 15 and 10 <= s_deg <= 235:
                    runs.append((s_deg, e_deg, length))
                start = None
        if runs:
            runs.sort(key=lambda x: x[2], reverse=True)
            w_start, w_end, w_len = runs[0]
            w_center = (w_start + w_len / 2.0) % 360
            w_dict = {
                "start": float(w_start),
                "end": float(w_end),
                "width": float(w_len),
                "center": float(w_center),
            }

    # Step 2: Search for Good (black) zone and cross-validate with Great zone
    if black_mask is not None:
        doubled_b = np.concatenate([black_mask, black_mask])
        b_runs = []
        start = None
        for i in range(len(doubled_b)):
            if doubled_b[i] and start is None:
                start = i
            elif not doubled_b[i] and start is not None:
                length = i - start
                s_deg = start % 360
                e_deg = (i - 1) % 360
                if 25 <= length <= 55 and 15 <= s_deg <= 255:
                    b_runs.append((s_deg, e_deg, length))
                start = None

        if w_dict is not None and b_runs:
            # Black zone MUST start within 0..6° of white zone ending!
            valid_b = [r for r in b_runs if abs((r[0] - w_dict["end"]) % 360) <= 6]
            if valid_b:
                b_s, b_e, b_l = valid_b[0]
                b_dict = {
                    "start": float(b_s),
                    "end": float(b_e),
                    "width": float(b_l),
                    "center": float((b_s + b_l / 2.0) % 360),
                }

        # If white is clear but black is fragmented by shadows, physically construct Good zone
        if w_dict is not None and b_dict is None:
            b_s = (w_dict["end"] + 1.0) % 360
            b_dict = {
                "start": float(b_s),
                "end": float((b_s + 42.0) % 360),
                "width": 42.0,
                "center": float((b_s + 21.0) % 360),
            }

        # Fallback if white was occluded on the very first frame:
        if w_dict is None and b_runs:
            b_runs.sort(key=lambda x: x[2], reverse=True)
            for cand in b_runs:
                b_s, b_e, b_l = cand
                if r66_val is not None:
                    pre_vals = [r66_val[(b_s - k) % 360] for k in range(1, 7)]
                    if max(pre_vals) > 160:  # Confirmed white zone precedes this black segment
                        w_s = (b_s - 9.5) % 360
                        w_dict = {
                            "start": float(w_s),
                            "end": float(b_s),
                            "width": 9.5,
                            "center": float((w_s + 4.75) % 360),
                        }
                        b_dict = {
                            "start": float(b_s),
                            "end": float(b_e),
                            "width": float(b_l),
                            "center": float((b_s + b_l / 2.0) % 360),
                        }
                        break

    return w_dict, b_dict


def is_angle_in_arc(angle, start, end, tol_start=2.0, tol_end=2.0):
    if start is None or end is None:
        return False
    adj_start = (start - tol_start) % 360
    adj_end = (end + tol_end) % 360
    diff = (angle - adj_start) % 360
    span = (adj_end - adj_start) % 360
    return diff <= span


def angular_dist_signed(target, current):
    return (target - current) % 360


def analyze_session(video_path):
    path = Path(video_path).resolve()
    print(f"\n===== АНАЛИЗ {path.name} =====")
    meta = probe_video(path)
    fps = meta.get("fps", 60.0)
    print(f"Видео: {meta.get('width')}x{meta.get('height')} @ {fps:.1f} FPS")

    events_list = parse_events(path)
    spaces = [t for t, k in events_list if k == "SPACE_DOWN"]
    labels, orphans = associate_labels(events_list)
    print(f"События ввода: {len(events_list)} всего, {len(spaces)} SPACES, {len(labels)} LABELS")

    cap = cv2.VideoCapture(str(path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    episodes = []
    current_episode = None

    exp_angle = None
    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        det = detect_frame(frame_gray, frame, expected_angle=exp_angle)
        t = f_idx / fps

        if det is not None:
            det["frame"] = f_idx
            det["t"] = t
            if current_episode is None:
                current_episode = {"start_f": f_idx, "start_t": t, "frames": []}
            current_episode["frames"].append(det)
            exp_angle = (det["needle_angle"] + DEFAULT_SPEED_DEG_S / fps) % 360
        else:
            exp_angle = None
            if current_episode is not None:
                if len(current_episode["frames"]) >= 8:
                    current_episode["end_f"] = f_idx - 1
                    current_episode["end_t"] = (f_idx - 1) / fps
                    current_episode["dur"] = current_episode["end_t"] - current_episode["start_t"]
                    episodes.append(current_episode)
                current_episode = None

    if current_episode is not None and len(current_episode["frames"]) >= 8:
        current_episode["end_f"] = total_frames - 1
        current_episode["end_t"] = (total_frames - 1) / fps
        current_episode["dur"] = current_episode["end_t"] - current_episode["start_t"]
        episodes.append(current_episode)

    cap.release()

    print(f"Обнаружено skill-check эпизодов: {len(episodes)}")

    processed_checks = []
    for i, ep in enumerate(episodes, 1):
        frames = ep["frames"]
        valid_needle = [f for f in frames if f["needle_strength"] > 25]
        if len(valid_needle) >= 5:
            ts = np.array([f["t"] for f in valid_needle])
            angs = np.array([f["needle_angle"] for f in valid_needle], dtype=float)
            angs_unwrapped = np.unwrap(angs * np.pi / 180.0) * 180.0 / np.pi
            poly = np.polyfit(ts, angs_unwrapped, 1)
            speed = float(poly[0])
            residual = float(np.sqrt(np.mean((np.polyval(poly, ts) - angs_unwrapped) ** 2)))
        else:
            speed = DEFAULT_SPEED_DEG_S
            residual = 0.0

        w_d, b_d = None, None
        for f in frames[:min(10, len(frames))]:
            w_d, b_d = extract_zones(f, None)
            if w_d is not None:
                break
        if w_d is None:
            w_d, b_d = extract_zones(frames[len(frames) // 2], None)

        if w_d:
            white_start, white_end, white_len = w_d["start"], w_d["end"], w_d["width"]
            great_target_angle = w_d["center"]
        else:
            white_start = white_end = white_len = great_target_angle = None

        if b_d:
            black_start, black_end, black_len = b_d["start"], b_d["end"], b_d["width"]
            good_target_angle = b_d["center"]
        else:
            black_start = black_end = black_len = good_target_angle = None

        white_seg = (white_start, white_end, white_len) if white_start is not None else None
        black_seg = (black_start, black_end, black_len) if black_start is not None else None

        last_det = frames[-1]
        last_needle = last_det["needle_angle"]
        first_needle = frames[0]["needle_angle"]

        in_white = False
        in_black = False
        if white_seg:
            in_white = is_angle_in_arc(last_needle, white_start, white_end, tol_start=2.0, tol_end=2.5)
        if black_seg:
            in_black = is_angle_in_arc(last_needle, black_start, black_end, tol_start=2.0, tol_end=3.5)

        if in_white:
            vis_outcome = "WHITE"
        elif in_black:
            vis_outcome = "BLACK"
        else:
            if black_seg and white_seg:
                d_from_black_end = angular_dist_signed(last_needle, black_end)
                if d_from_black_end < 45:
                    vis_outcome = "MISS"
                else:
                    vis_outcome = "EARLY"
            else:
                vis_outcome = "UNKNOWN"

        best_space = None
        min_dt = float("inf")
        for s in spaces:
            dt = s - ep["end_t"]
            if abs(dt) < min_dt and dt >= -0.10:
                min_dt = dt
                best_space = s

        label_info = None
        if best_space is not None:
            for l in labels:
                if l.get("space_time") == best_space:
                    label_info = l
                    break

        gt_label = label_info.get("label") if label_info else None
        reaction_delay = (best_space - ep["end_t"]) if best_space is not None else None

        check_summary = {
            "index": i,
            "start_t": ep["start_t"],
            "end_t": ep["end_t"],
            "duration": ep["dur"],
            "frames": len(frames),
            "speed_deg_s": speed,
            "fit_residual_deg": residual,
            "first_needle_angle": first_needle,
            "last_needle_angle": last_needle,
            "white_zone": {
                "start": white_start,
                "end": white_end,
                "width": white_len,
                "center": great_target_angle
            },
            "black_zone": {
                "start": black_start,
                "end": black_end,
                "width": black_len,
                "center": good_target_angle
            },
            "visual_outcome": vis_outcome,
            "ground_truth": gt_label,
            "space_time": best_space,
            "reaction_delay_s": reaction_delay,
            "label_delay_s": label_info.get("label_delay") if label_info else None
        }
        processed_checks.append(check_summary)

        match_str = ""
        if gt_label:
            match_str = " [MATCH]" if vis_outcome == gt_label else " [MISMATCH]"
        print(f"  #{i} [{ep['start_t']:.2f}s..{ep['end_t']:.2f}s] Speed={speed:.1f}°/s -> Visual={vis_outcome} | Label={gt_label or '?'}{match_str}")
        if white_seg and black_seg:
            print(f"      Zones: Great=[{white_start}..{white_end}] ({white_len}°), Good=[{black_start}..{black_end}] ({black_len}°), LastNeedle={last_needle:.1f}°")

    labeled_count = sum(1 for c in processed_checks if c["ground_truth"])
    agreed_count = sum(1 for c in processed_checks if c["ground_truth"] and c["visual_outcome"] == c["ground_truth"])
    accuracy = (agreed_count / labeled_count) if labeled_count > 0 else 1.0

    report = {
        "file": path.name,
        "video": meta,
        "events_count": len(events_list),
        "spaces_count": len(spaces),
        "labeled_count": labeled_count,
        "agreed_count": agreed_count,
        "accuracy": accuracy,
        "checks": processed_checks
    }

    report_file = ROOT / f"analysis_session_{path.stem}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    merge_into_all(report)
    update_dataset_rows(report)

    print(f"\nИТОГ {path.name}: Всего checks={len(processed_checks)}, Labeled={labeled_count}, Visual Agreement={agreed_count}/{labeled_count} ({accuracy*100:.1f}%)")
    print(f"===== КОНЕЦ АНАЛИЗА {path.name} =====\n")
    return report


def merge_into_all(session_report):
    lock_fd = os.open(str(MERGE_LOCK), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        data = {"version": 4, "sessions": []}
        if REPORT_ALL.exists():
            try:
                with open(REPORT_ALL, "r", encoding="utf-8") as f:
                    old = json.load(f)
                    if isinstance(old, dict) and "sessions" in old:
                        data["sessions"] = [s for s in old["sessions"] if s.get("file") != session_report["file"]]
            except Exception:
                pass
        data["sessions"].append(session_report)
        data["total_sessions"] = len(data["sessions"])
        data["total_checks"] = sum(len(s.get("checks", [])) for s in data["sessions"])

        with open(REPORT_ALL, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def update_dataset_rows(session_report):
    existing = {}
    if DATASET_PATH.exists():
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    existing[row["id"]] = row
                except Exception:
                    pass

    filename = session_report["file"]
    for c in session_report.get("checks", []):
        row_id = f"{filename}:{c['index']}"
        outcome = c["ground_truth"] or c["visual_outcome"]
        if outcome not in {"WHITE", "BLACK", "MISS"}:
            continue
        row = {
            "id": row_id,
            "session": filename,
            "check_index": c["index"],
            "outcome": outcome,
            "speed_deg_s": c["speed_deg_s"],
            "white_width": c["white_zone"].get("width"),
            "black_width": c["black_zone"].get("width"),
            "white_center": c["white_zone"].get("center"),
            "black_center": c["black_zone"].get("center"),
            "reaction_delay_s": c.get("reaction_delay_s"),
            "duration": c["duration"],
            "frames": c["frames"]
        }
        existing[row_id] = row

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        for k in sorted(existing.keys()):
            f.write(json.dumps(existing[k], ensure_ascii=False) + "\n")


def analyze_all():
    files = sorted(ROOT.glob("session_*.mkv"))
    print(f"Найдено {len(files)} файлов сессий в {ROOT}. Запускаю полный анализ...")
    total_checks = 0
    total_labeled = 0
    total_agreed = 0

    for f in files:
        if f.stat().st_size < 10000:
            print(f"Пропуск пустого/слишком маленького файла: {f.name}")
            continue
        try:
            rep = analyze_session(f)
            total_checks += len(rep["checks"])
            total_labeled += rep["labeled_count"]
            total_agreed += rep["agreed_count"]
        except Exception as e:
            print(f"Ошибка при анализе {f.name}: {e}")

    acc = (total_agreed / total_labeled * 100.0) if total_labeled > 0 else 100.0
    print("\n==========================================")
    print(f"ИТОГ ПО ВСЕМ СЕССИЯМ: {total_checks} skill-checks, {total_labeled} с ручной разметкой, Совпадение визуального анализа={acc:.1f}%")
    print(f"База данных сохранена в {REPORT_ALL}")
    print(f"Датасет сохранен в {DATASET_PATH}")
    print("==========================================\n")


def main():
    ap = argparse.ArgumentParser(description="Violent District Skillcheck Fast Analyzer")
    ap.add_argument("file", nargs="?", help="Файл сессии (по умолчанию самый свежий)")
    ap.add_argument("--all", action="store_true", help="Проанализировать все файлы session_*.mkv")
    args = ap.parse_args()

    if args.all:
        analyze_all()
    else:
        path = args.file or newest_session()
        analyze_session(path)


if __name__ == "__main__":
    main()
