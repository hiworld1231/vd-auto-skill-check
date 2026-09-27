#!/usr/bin/env python3
"""Extract representative game-scene frames without a visible check prompt.

The source MKVs may include legacy program overlays while a check is active.
This utility only samples frames where the Space prompt template is absent;
it makes no claim that all other pixels are free of overlays.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path)
    parser.add_argument("--max-per-session", type=int, default=1)
    args = parser.parse_args()
    if args.max_per_session < 1:
        parser.error("--max-per-session must be positive")

    videos = sorted({p.resolve() for root in args.roots if root.exists()
                     for p in ([root] if root.is_file() else root.rglob("*.mkv"))
                     if p.is_file() and p.suffix.lower() == ".mkv"})
    args.out.mkdir(parents=True, exist_ok=True)
    detector_template = cv2.imread(str(Path(__file__).resolve().parents[1] /
                                        "assets/space_template.png"), cv2.IMREAD_GRAYSCALE)
    if detector_template is None:
        raise RuntimeError("Cannot load the skill-check prompt template")

    seen = set()
    rows = []
    for path in videos:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            rows.append({"source": str(path), "error": "VIDEO_OPEN_FAILED"})
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        frame_number = 0
        selected = []
        step = max(1, round(fps / 4))
        try:
            while len(selected) < args.max_per_session:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_number += 1
                if frame_number < fps * 1.25 or frame_number % step:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, prompt_score, _, _ = cv2.minMaxLoc(
                    cv2.matchTemplate(gray, detector_template, cv2.TM_CCOEFF_NORMED))
                if prompt_score >= .80:
                    continue
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                yellow = cv2.inRange(hsv, (22, 150, 130), (42, 255, 255))
                count, _, stats, centers = cv2.connectedComponentsWithStats(yellow)
                avatar_pixels = max((int(stats[i, cv2.CC_STAT_AREA])
                                      for i in range(1, count)
                                      if centers[i, 1] >= frame.shape[0] * .48), default=0)
                if avatar_pixels < 500:
                    continue
                name = f"{path.stem}_{digest[:8]}_{len(selected)+1:02d}.png"
                destination = args.out / name
                if not cv2.imwrite(str(destination), frame):
                    raise RuntimeError(f"Cannot write {destination}")
                selected.append({"image": str(destination), "frame": frame_number,
                                 "time_s": round(frame_number / fps, 3),
                                 "prompt_template_score": round(float(prompt_score), 4),
                                 "lower_frame_yellow_component_pixels": avatar_pixels,
                                 "width": int(frame.shape[1]), "height": int(frame.shape[0])})
        finally:
            cap.release()
        rows.append({"source": str(path), "sha256": digest, "fps": fps,
                     "selected_scenes": selected,
                     "decoded_frames": frame_number})
        print(f"{path.name}: selected {len(selected)} clean-prompt scene(s)", flush=True)

    report = {"unique_source_sessions": len(seen), "video_paths": len(videos),
              "selected_scene_count": sum(len(r.get("selected_scenes", [])) for r in rows),
              "selection_rule": "Prompt template score below 0.80 and a connected yellow avatar-colored component of at least 500 pixels in the lower half. This heuristic excludes obvious terminal/menu frames but does not prove absence of other overlays.",
              "sessions": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    if args.scene_list is not None:
        scene_list_base = args.scene_list.parent.resolve()
        images = [f"./{p.relative_to(scene_list_base).as_posix()}"
                  for row in rows for scene in row.get("selected_scenes", [])
                  for p in [Path(scene["image"]).resolve()]]
        args.scene_list.parent.mkdir(parents=True, exist_ok=True)
        args.scene_list.write_text("window.VDSceneBackgrounds = " +
                                   json.dumps(images, ensure_ascii=False) + ";\n",
                                   encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("unique_source_sessions", "video_paths",
                                             "selected_scene_count")}, indent=2))


if __name__ == "__main__":
    main()
