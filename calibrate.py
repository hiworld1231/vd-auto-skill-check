#!/usr/bin/env python3
"""
Hardware Pipeline Latency Calibrator for Violent District Skill Check AI.
Automatically measures the exact physical screen-to-python capture delay
by flashing encoded test patches and calculates the optimal compensation value.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT.parent / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
if VENV_PY.exists() and sys.prefix != str(VENV_DIR.resolve()):
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

import cv2
import numpy as np
from core.capture import ScreenGrabber, CAPTURE_REGION

CONFIG_PATH = ROOT / "config.json"


def calibrate_latency(trials=20, fps=240, region=None, auto_save=False):
    reg = region or CAPTURE_REGION
    w, h = reg["width"], reg["height"]
    x, y = reg["left"], reg["top"]

    print("==================================================")
    print("   HARDWARE SCREEN-TO-PYTHON LATENCY CALIBRATOR   ")
    print("==================================================")
    print(f"Область теста: {w}x{h}+{x}+{y}")
    print(f"Частота захвата: {fps} FPS")
    print(f"Количество замеров: {trials}")
    print("Инициализация тестового окна и KMS пайплайна...")
    print("--------------------------------------------------")

    win_name = "CALIBRATION_PROBE"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.moveWindow(win_name, x, y)
    cv2.resizeWindow(win_name, w, h)

    # Initial blank
    blank = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imshow(win_name, blank)
    cv2.waitKey(1)

    grabber = ScreenGrabber(reg, fps=fps)
    time.sleep(1.0)  # Allow pipeline to settle

    history = {}
    delays = []
    last_id = None

    print("Проведение аппаратных замеров задержки (мигание контрольными кодами)...")

    for i in range(1, trials + 1):
        code = (i * 13) % 240 + 10
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[20:100, 20:100] = (code, 120, 220)

        t_post = time.monotonic()
        history[code] = t_post
        cv2.imshow(win_name, img)
        cv2.waitKey(1)

        t_end = time.monotonic() + 0.160
        matched = False
        while time.monotonic() < t_end:
            f, fid, _ = grabber.grab(wait_new=True, last_id=last_id, timeout=0.010)
            if f is not None and fid != last_id:
                last_id = fid
                t_recv = time.monotonic()
                patch_val = int(np.median(f[40:80, 40:80, 0]))
                for c, t_p in list(history.items()):
                    if abs(patch_val - c) <= 3:
                        delay_ms = (t_recv - t_p) * 1000.0
                        delays.append(delay_ms)
                        del history[c]
                        matched = True
                        break
                if matched:
                    break
            time.sleep(0.001)

    grabber.close()
    cv2.destroyAllWindows()

    if not delays:
        print("[ОШИБКА] Не удалось зафиксировать кадры калибровки.")
        return None

    # Filter initial warmup outlier
    valid_delays = delays[1:] if len(delays) > 3 else delays
    mean_lat = float(np.mean(valid_delays))
    median_lat = float(np.median(valid_delays))
    min_lat = float(np.min(valid_delays))
    max_lat = float(np.max(valid_delays))
    std_lat = float(np.std(valid_delays))

    game_engine_ms = 19.0  # Опрос ввода и физический тик Sober/Roblox при ~70 FPS
    total_recommended = round(median_lat + game_engine_ms, 1)

    print("\n---------------- РЕЗУЛЬТАТЫ ----------------")
    print(f"Все замеры (мс): {[round(d, 1) for d in valid_delays]}")
    print(f"Задержка видеотракта:    {median_lat:.1f} мс (KMS -> GPU -> Python)")
    print(f"Задержка движка Roblox:  {game_engine_ms:.1f} мс (опрос ввода + тик физики при 70 FPS)")
    print(f"Джиттер захвата:         {std_lat:.2f} мс")
    print(f"ИТОГОВАЯ КОМПЕНСАЦИЯ:    {total_recommended:.1f} мс")
    print("--------------------------------------------")

    # Save to config.json with safety guards
    if len(valid_delays) < 5:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Недостаточно замеров ({len(valid_delays)} < 5). config.json НЕ изменён.")
    elif not (50.0 <= total_recommended <= 180.0):
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Задержка {total_recommended:.1f}мс вне допустимого диапазона [50..180мс]. config.json НЕ изменён.")
    elif auto_save:
        cfg = {}
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                cfg = {}
        cfg["latency_ms"] = total_recommended
        cfg["fps"] = fps
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[УСПЕХ] Значение latency_ms={total_recommended:.1f} сохранено в config.json.")
    else:
        print("[ИНФО] Запустите с флагом --save для автоматического сохранения в config.json.")

    return total_recommended


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill Check Latency Calibrator")
    parser.add_argument("--trials", type=int, default=20, help="Количество замеров")
    parser.add_argument("--fps", type=int, default=240, help="FPS захвата")
    parser.add_argument("--save", action="store_true", help="Автоматически сохранить в config.json")
    args = parser.parse_args()

    calibrate_latency(trials=args.trials, fps=args.fps, auto_save=args.save)
