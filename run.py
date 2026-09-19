#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT.parent / ".venv"
VENV_PY = VENV_DIR / "bin" / "python"
if VENV_PY.exists() and sys.prefix != str(VENV_DIR.resolve()):
    os.execv(str(VENV_PY), [str(VENV_PY)] + sys.argv)

PYTHON_BIN = str(VENV_PY) if VENV_PY.exists() else sys.executable

def main():
    ap = argparse.ArgumentParser(description="Violent District Launcher")
    ap.add_argument("--gen-rush", "--speed-perk", dest="gen_rush", action="store_true", help="Запустить бота в режиме GEN RUSH")
    ap.add_argument("--hud", action="store_true", help="Включить HUD")
    ap.add_argument("--dry-run", action="store_true", help="Тестовый режим без нажатий")
    args, unknown = ap.parse_known_args()

    if args.gen_rush:
        cmd = [PYTHON_BIN, str(ROOT / "skillcheck_bot.py"), "--gen-rush"]
        if args.hud:
            cmd.append("--hud")
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd + unknown, cwd=ROOT)
        return

    print("==================================================")
    print("     VIOLENT DISTRICT (DBD) - SKILL CHECK AI      ")
    print("==================================================")
    print("  Оптимизировано для 60-75 FPS (220 FPS KMS + Self-Learning AI)")
    print()
    print("1. AUTO BOT       — Автономный бот (Headless, без окна HUD, мин. задержка)")
    print("2. AUTO BOT + HUD — Автономный бот с окном-радаром (визуализация)")
    print("3. DRY-RUN        — Тестовый режим без нажатий (эмуляция расчетов)")
    print("4. CALIBRATE      — Авто-калибровка задержки экрана под твой дисплей")
    print("5. REAL TESTS     — Запуск аппаратных тестов (AI Learner, Frenzy, 18 сессий)")
    print("6. TELEMETRY      — Анализ сохранённых записей проверок (replays) и точности")
    print("7. GEN RUSH       — Режим ускоренной стрелки с отслеживанием фазы игры")
    print("8. PERK ANALYSIS  — Анализ точности и скоростных бакетов перка ускорения")
    print()

    while True:
        choice = input("Выбери режим [1-8]: ").strip()
        if choice in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            break
        print("Введи число от 1 до 8.")

    print()
    try:
        if choice == "1":
            print("=== ЗАПУСК AUTO BOT (HEADLESS) ===")
            print("Бот захватывает экран через GPU KMS (120 FPS), нажимает Space через UInput.")
            print("Черный ящик активен: все проверки автоматически записываются в replays/")
            subprocess.run([PYTHON_BIN, str(ROOT / "skillcheck_bot.py")], cwd=ROOT)

        elif choice == "2":
            print("=== ЗАПУСК AUTO BOT С HUD-ОКНОМ ===")
            print("Черный ящик активен: все проверки автоматически записываются в replays/")
            subprocess.run([PYTHON_BIN, str(ROOT / "skillcheck_bot.py"), "--hud"], cwd=ROOT)

        elif choice == "3":
            print("=== ЗАПУСК ТЕСТОВОГО РЕЖИМА (DRY-RUN) ===")
            subprocess.run([PYTHON_BIN, str(ROOT / "skillcheck_bot.py"), "--dry-run"], cwd=ROOT)

        elif choice == "4":
            print("=== КАЛИБРОВКА ЗАДЕРЖКИ ===")
            subprocess.run([PYTHON_BIN, str(ROOT / "calibrate.py"), "--save"], cwd=ROOT)

        elif choice == "5":
            print("=== ЗАПУСК ТЕСТОВ ===")
            subprocess.run([PYTHON_BIN, str(ROOT / "tests" / "test_skillcheck_real.py")], cwd=ROOT)

        elif choice == "6":
            print("=== АНАЛИЗ ТЕЛЕМЕТРИИ И ЗАПИСЕЙ REPLAYS ===")
            subprocess.run([PYTHON_BIN, str(ROOT / "tools" / "analyze_telemetry.py")], cwd=ROOT)

        elif choice == "7":
            print("=== ЗАПУСК GEN RUSH / SPEED PERK BOT ===")
            print("Робастная оценка скорости, отслеживание фазы отклика игры, теневые профили.")
            subprocess.run([PYTHON_BIN, str(ROOT / "skillcheck_bot.py"), "--gen-rush"], cwd=ROOT)

        elif choice == "8":
            print("=== АНАЛИЗ СКОРОСТНЫХ БАКЕТОВ И ТОЧНОСТИ ПЕРКА ===")
            subprocess.run([PYTHON_BIN, str(ROOT / "tools" / "analyze_perk_live.py")], cwd=ROOT)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
