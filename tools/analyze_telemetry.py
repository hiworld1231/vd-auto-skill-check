#!/usr/bin/env python3
"""
Violent District Skill Check Telemetry & Replay Analyzer.
Scans all recorded flight recorder episodes in replays/, performs rigorous error distribution analysis,
identifies root causes for misses, computes mathematically optimal hardware latency compensation,
and outputs formatted diagnostics.
"""

import argparse
import datetime
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = ROOT.parent / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
if VENV_PY.exists() and sys.prefix != str(VENV_DIR.resolve()):
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

REPLAYS_DIR = ROOT / "replays"
CONFIG_PATH = ROOT / "config.json"


def load_all_episodes(replays_dir: Path) -> List[Dict[str, Any]]:
    """Loads all check_*.json files from the replays directory."""
    if not replays_dir.exists():
        return []

    episodes = []
    manifest_file = replays_dir / "manifest.jsonl"
    json_files = sorted(replays_dir.glob("check_*_*.json"))

    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            data["_json_path"] = str(jf)
            base = jf.stem
            png_path = replays_dir / f"{base}_diagnostic.png"
            mp4_path = replays_dir / f"{base}.mp4"
            data["_diagnostic_png"] = str(png_path) if png_path.exists() else None
            data["_mp4"] = str(mp4_path) if mp4_path.exists() else None
            episodes.append(data)
        except Exception:
            continue

    return episodes


def analyze_episodes(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not episodes:
        return {"total": 0}

    total = len(episodes)
    greats = []
    goods = []
    aborted = []
    misses = []
    unknowns = []

    valid_errors_ms = []
    valid_errors_deg = []
    latencies = []
    speeds = []

    for ep in episodes:
        eval_data = ep.get("evaluation") or {}
        outcome = eval_data.get("outcome", "UNKNOWN")
        err_deg = eval_data.get("error_deg")
        err_ms = eval_data.get("error_ms")
        lat = ep.get("configured_latency_ms")
        speed = (ep.get("trigger") or {}).get("speed_deg_s") or 270.0

        if outcome == "GREAT":
            greats.append(ep)
        elif outcome == "GOOD":
            goods.append(ep)
        elif outcome in ("ABORTED_LMB", "ABORTED"):
            aborted.append(ep)
        elif outcome == "MISS":
            misses.append(ep)
        else:
            unknowns.append(ep)

        if outcome in ("GREAT", "GOOD") and err_ms is not None and abs(err_ms) <= 80.0:
            valid_errors_ms.append(err_ms)
            if err_deg is not None:
                valid_errors_deg.append(err_deg)
            if lat is not None:
                latencies.append(lat)
            if speed > 0:
                speeds.append(speed)

    # Statistical breakdown
    mean_err_ms = float(np.mean(valid_errors_ms)) if valid_errors_ms else 0.0
    std_err_ms = float(np.std(valid_errors_ms)) if valid_errors_ms else 0.0
    median_err_ms = float(np.median(valid_errors_ms)) if valid_errors_ms else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 100.0
    recommended_latency = round(mean_lat + median_err_ms, 1)

    completed_total = len(greats) + len(goods) + len(misses)
    return {
        "total": total,
        "completed_total": completed_total,
        "great_count": len(greats),
        "good_count": len(goods),
        "aborted_count": len(aborted),
        "miss_count": len(misses),
        "unknown_count": len(unknowns),
        "great_pct": (len(greats) / completed_total * 100.0) if completed_total else 0.0,
        "good_pct": (len(goods) / completed_total * 100.0) if completed_total else 0.0,
        "miss_pct": (len(misses) / completed_total * 100.0) if completed_total else 0.0,
        "mean_error_ms": mean_err_ms,
        "std_error_ms": std_err_ms,
        "median_error_ms": median_err_ms,
        "current_mean_latency_ms": mean_lat,
        "recommended_latency_ms": recommended_latency,
        "misses": misses,
        "episodes": episodes,
    }


def print_analysis_report(stats: Dict[str, Any]):
    total = stats.get("total", 0)
    if total == 0:
        print("\n=======================================================")
        print("          VIOLENT DISTRICT - ТЕЛЕМЕТРИЯ               ")
        print("=======================================================")
        print("В папке replays/ пока нет сохранённых записей проверок.")
        print("Запусти бота (опция 1 или 2 в run.py) и поиграй — все проверки запишутся автоматически!")
        print("=======================================================\n")
        return

    completed = stats.get("completed_total", total)
    aborted = stats.get("aborted_count", 0)
    print("\n=======================================================")
    print("      VIOLENT DISTRICT - АНАЛИЗ ПОЛЁТНЫХ ДАННЫХ        ")
    print("=======================================================")
    print(f"Всего зафиксировано проверок: {total}")
    if aborted > 0:
        print(f"  🛑 Прервано игроком (ОТЖАТ ЛКМ): {aborted:3d} (не является промахом бота)")
    print(f"  🎯 Идеально (GREAT):            {stats['great_count']:3d} ({stats['great_pct']:5.1f}%)")
    print(f"  ⚡ Успешно  (GOOD):             {stats['good_count']:3d} ({stats['good_pct']:5.1f}%)")
    print(f"  ❌ Промахи  (MISS):             {stats['miss_count']:3d} ({stats['miss_pct']:5.1f}%)")
    success_rate = stats['great_pct'] + stats['good_pct']
    print(f"  ⭐️ ОБЩИЙ УСПЕХ (Great+Good):    {success_rate:5.1f}%")
    print("-------------------------------------------------------")
    print("ТОЧНОСТЬ И СМЕЩЕНИЕ ТАЙМИНГА:")
    print(f"  Среднее отклонение от центра:  {stats['median_error_ms']:+5.1f} мс ({stats['mean_error_ms']:+5.1f} мс ср.)")
    print(f"  Разброс (джиттер / шум кадров): ±{stats['std_error_ms']:4.1f} мс")
    print(f"  Текущая средняя задержка:      {stats['current_mean_latency_ms']:5.1f} мс")
    print(f"  👉 РЕКОМЕНДУЕМАЯ ЗАДЕРЖКА:     {stats['recommended_latency_ms']:5.1f} мс")
    print("-------------------------------------------------------")

    misses = stats.get("misses", [])
    if misses:
        print(f"ДЕТАЛИ ПРОМАХОВ ({len(misses)} шт.):")
        for i, m in enumerate(misses[:10], 1):
            cid = m.get("check_id", "unknown")
            ev = m.get("evaluation") or {}
            trig = m.get("trigger") or {}
            err_ms = ev.get("error_ms", 0.0)
            reason = trig.get("reason", "no trigger")
            sign = "+" if err_ms >= 0 else ""
            diag_file = m.get("_diagnostic_png", "нет файла")
            print(f"  [{i}] {cid}: ошибка {sign}{err_ms:.1f}мс ({reason})")
            if diag_file:
                print(f"      Снимок: {diag_file}")
    else:
        print("🎉 0 ПРОМАХОВ! Все проверки успешно попадают в зону!")
    print("=======================================================\n")


def apply_recommended_latency(rec_lat: float):
    if not CONFIG_PATH.exists():
        print("config.json не найден.")
        return
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        old_lat = cfg.get("latency_ms", 100.0)
        cfg["latency_ms"] = rec_lat
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        print(f"✅ config.json успешно обновлён: {old_lat:.1f} мс -> {rec_lat:.1f} мс")
    except Exception as e:
        print(f"Ошибка сохранения config.json: {e}")


def main():
    ap = argparse.ArgumentParser(description="Violent District Replay Telemetry Analyzer")
    ap.add_argument("--dir", type=str, default=str(REPLAYS_DIR), help="Папка с записями replays")
    ap.add_argument("--apply", action="store_true", help="Автоматически сохранить рекомендованную задержку в config.json")
    args = ap.parse_args()

    replays_path = Path(args.dir)
    episodes = load_all_episodes(replays_path)
    stats = analyze_episodes(episodes)
    print_analysis_report(stats)

    if args.apply and stats.get("total", 0) > 0:
        apply_recommended_latency(stats["recommended_latency_ms"])


if __name__ == "__main__":
    main()
