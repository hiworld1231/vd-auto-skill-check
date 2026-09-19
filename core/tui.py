"""
Terminal User Interface (TUI) for Violent District Skill Check AI.
Built with `rich` for ultra-clean rendering, real-time metrics, visual deviation gauge,
zero-latency decoupled background rendering, and interactive hotkeys.
"""

import collections
import os
import select
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


class TerminalKeyboardListener:
    """Non-blocking keyboard reader for interactive hotkeys in terminal."""

    def __init__(self, on_key_callback: Callable[[str], None]):
        self.on_key = on_key_callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._orig_attr = None

    def start(self):
        if not sys.stdin.isatty():
            return
        try:
            import termios
            import tty

            self._orig_attr = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True, name="TUI-Keyboard")
            self._thread.start()
        except Exception:
            pass

    def stop(self):
        self._running = False
        if self._orig_attr is not None:
            try:
                import termios

                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_attr)
            except Exception:
                pass
            self._orig_attr = None

    def _run(self):
        while self._running:
            try:
                r, _, _ = select.select([sys.stdin], [], [], 0.08)
                if r and self._running:
                    ch = sys.stdin.read(1)
                    if ch:
                        self.on_key(ch)
            except Exception:
                break


class SkillCheckTUI:
    """Decoupled high-performance Rich TUI for SkillCheckBot."""

    def __init__(
        self,
        mode: str = "AUTO-BOT",
        target_mode: str = "GREAT",
        initial_latency: float = 85.0,
        initial_ratio: float = 0.45,
        backend: str = "UInput",
        fps: int = 120,
        on_action: Optional[Callable[[str], None]] = None,
        refresh_rate: float = 12.0,  # 12 Hz refresh is silky smooth with 0% CPU impact
    ):
        self.mode = mode
        self.target_mode = target_mode
        self.latency_ms = initial_latency
        self.target_ratio = initial_ratio
        self.backend = backend
        self.expected_fps = fps
        self.on_action = on_action
        self.refresh_interval = 1.0 / max(1.0, refresh_rate)

        # Live telemetry state
        self.current_fps = float(fps)
        self.status_text = "MONITORING (WAITING)"
        self.status_style = "bold green"
        self.in_check = False
        self.armed = False
        self.chain_count = 0
        self.max_chain = 0
        self.needle_speed = 0.0
        self.target_angle = 0.0
        self.rem_ms = 0.0

        # Latency control and override verification
        self.requested_base_latency: float = float(initial_latency)
        self.actual_used_delays: List[float] = []
        self.profile_overrides_count: int = 0

        # Accuracy statistics
        self.total_checks = 0
        self.great_count = 0
        self.good_count = 0
        self.miss_count = 0
        self.aborted_count = 0
        self.lmb_held = False

        # Deviation gauge data
        self.last_outcome = "WAITING"
        self.last_target_angle = 0.0
        self.last_hit_angle = 0.0
        self.last_error_deg = 0.0
        self.last_error_ms = 0.0

        # Event log (ring buffer of last 6 events)
        self.events = collections.deque(maxlen=6)
        self._lock = threading.Lock()

        # Background rendering & keyboard listener
        self.console = Console()
        self.live: Optional[Live] = None
        self.key_listener: Optional[TerminalKeyboardListener] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.log(f"⚡ Бот запущен: {mode} ({backend}, {fps} FPS KMS).", style="cyan")

    def start(self):
        """Starts the decoupled TUI rendering loop and keyboard listener."""
        if self._running:
            return
        self._running = True

        # Start terminal keyboard listener
        self.key_listener = TerminalKeyboardListener(self._handle_raw_key)
        self.key_listener.start()

        # Start Rich Live
        self.live = Live(
            self._render_view(),
            console=self.console,
            screen=True,
            auto_refresh=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self.live.start()

        self._thread = threading.Thread(target=self._update_loop, daemon=True, name="TUI-Renderer")
        self._thread.start()

    def stop(self):
        """Stops the TUI rendering loop and restores terminal state."""
        self._running = False
        if self.key_listener:
            self.key_listener.stop()
            self.key_listener = None
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.3)
        if self.live:
            try:
                self.live.stop()
            except Exception:
                pass
            self.live = None

    def _handle_raw_key(self, ch: str):
        if not self.on_action:
            return
        if ch in ("+", "="):
            self.on_action("inc_latency")
        elif ch in ("-", "_"):
            self.on_action("dec_latency")
        elif ch == "]":
            self.on_action("inc_offset")
        elif ch == "[":
            self.on_action("dec_offset")
        elif ch in ("q", "Q", "\x1b"):  # q or Esc
            self.on_action("quit")

    def _update_loop(self):
        while self._running:
            t0 = time.monotonic()
            try:
                if self.live:
                    self.live.update(self._render_view(), refresh=True)
            except Exception:
                pass
            elapsed = time.monotonic() - t0
            sleep_time = max(0.005, self.refresh_interval - elapsed)
            time.sleep(sleep_time)

    # -------------------------------------------------------------------------
    # State update methods (instantaneous, 0-lag for capture thread)
    # -------------------------------------------------------------------------

    def set_status(
        self,
        text: str,
        style: str = "bold green",
        in_check: bool = False,
        armed: bool = False,
        chain: int = 1,
    ):
        with self._lock:
            self.status_text = text
            self.status_style = style
            self.in_check = in_check
            self.armed = armed
            self.chain_count = chain
            if chain > self.max_chain:
                self.max_chain = chain

    def update_metrics(
        self,
        fps: float,
        latency_ms: float,
        target_ratio: float,
        needle_speed: float = 0.0,
        target_angle: float = 0.0,
        rem_ms: float = 0.0,
        lmb_held: Optional[bool] = None,
    ):
        with self._lock:
            self.current_fps = fps
            self.latency_ms = latency_ms
            self.target_ratio = target_ratio
            if needle_speed > 0:
                self.needle_speed = needle_speed
            if target_angle > 0:
                self.target_angle = target_angle
            self.rem_ms = rem_ms
            if lmb_held is not None:
                self.lmb_held = lmb_held

    def record_fire_telemetry(
        self,
        requested_latency_ms: float,
        actual_used_delay: float,
        is_override: bool = False,
    ):
        with self._lock:
            self.requested_base_latency = requested_latency_ms
            self.actual_used_delays.append(actual_used_delay)
            if is_override:
                self.profile_overrides_count += 1

    def record_hit(
        self,
        outcome: str,
        fact_angle: Optional[float],
        target_angle: Optional[float],
        error_deg: Optional[float],
        error_ms: Optional[float],
        chain: int = 1,
        latency_ms: Optional[float] = None,
        requested_latency_ms: Optional[float] = None,
        actual_used_delay: Optional[float] = None,
        is_profile_override: bool = False,
    ):
        with self._lock:
            if latency_ms is not None:
                self.latency_ms = latency_ms
            if requested_latency_ms is not None:
                self.requested_base_latency = requested_latency_ms

            if outcome == "GREAT":
                self.total_checks += 1
                self.great_count += 1
                tag_style = "bold green"
                tag_name = "GREAT"
            elif outcome == "GOOD":
                self.total_checks += 1
                self.good_count += 1
                tag_style = "bold yellow"
                tag_name = "GOOD"
            elif outcome in ("ABORTED_LMB", "ABORTED"):
                self.aborted_count += 1
                tag_style = "bold cyan"
                tag_name = "ABORTED (LMB)"
            elif outcome == "UNCONFIRMED":
                tag_style = "bold yellow"
                tag_name = "UNCONFIRMED"
            else:
                self.total_checks += 1
                self.miss_count += 1
                tag_style = "bold red"
                tag_name = "MISS"

            self.last_outcome = outcome
            self.last_target_angle = target_angle
            self.last_hit_angle = fact_angle
            self.last_error_deg = error_deg
            self.last_error_ms = error_ms

            chain_txt = f" [Серия #{chain}]" if chain > 1 else ""
            if outcome in ("ABORTED_LMB", "ABORTED"):
                msg = f"[{tag_style}]🛑 {tag_name}[/]{chain_txt} | Отжат ЛКМ игроком (проверка отменена, промах не засчитан)"
                self._log_internal(msg)
                self._log_internal("[dim]💾 [REC] ABORTED_LMB сохранён в replays/ (отжатие кнопки мыши)[/]")
            else:
                fact_txt = f"{float(fact_angle):5.1f}°" if fact_angle is not None else "N/A"
                target_txt = f"{float(target_angle):5.1f}°" if target_angle is not None else "N/A"
                if error_deg is not None and error_ms is not None:
                    sign = "+" if float(error_deg) >= 0 else ""
                    err_txt = f"{sign}{float(error_deg):4.1f}° / {sign}{float(error_ms):4.1f}мс"
                else:
                    err_txt = "N/A"
                msg = (
                    f"[{tag_style}]💥 {tag_name}[/]{chain_txt} | "
                    f"Факт={fact_txt} | Цель={target_txt} (ошибка: {err_txt})"
                )
                self._log_internal(msg)
                if outcome == "MISS":
                    self._log_internal("[bold red]💾 [REC] ПРОМАХ сохранён в replays/ (mp4 + json + diagnostic.png)[/]")
                else:
                    self._log_internal(f"[dim]💾 [REC] {outcome} сохранён в replays/ (mp4 + json + strip)[/]")

    def log(self, message: str, style: str = "white"):
        with self._lock:
            self._log_internal(f"[{style}]{message}[/]")

    def _log_internal(self, rich_text: str):
        now_str = time.strftime("%H:%M:%S")
        self.events.append(f"[dim]{now_str}[/] {rich_text}")

    # -------------------------------------------------------------------------
    # View rendering
    # -------------------------------------------------------------------------

    def _render_view(self) -> Panel:
        with self._lock:
            status_text = self.status_text
            status_style = self.status_style
            chain_count = self.chain_count
            max_chain = self.max_chain
            current_fps = self.current_fps
            latency_ms = self.latency_ms
            target_ratio = self.target_ratio
            speed = self.needle_speed
            total = self.total_checks
            great = self.great_count
            good = self.good_count
            miss = self.miss_count
            aborted = self.aborted_count
            lmb_held = self.lmb_held
            last_outcome = self.last_outcome
            last_target = self.last_target_angle
            last_hit = self.last_hit_angle
            last_err = self.last_error_deg
            last_err_ms = self.last_error_ms
            events = list(self.events)

        # 1. Header Table
        header_table = Table.grid(expand=True)
        header_table.add_column(justify="left", ratio=3)
        header_table.add_column(justify="center", ratio=2)
        header_table.add_column(justify="right", ratio=3)

        title_text = Text.from_markup(f"⚡ [bold cyan]VIOLENT DISTRICT[/] [white]— SKILL CHECK AI[/]")
        status_badge = Text.from_markup(f"[{status_style}]● {status_text}[/]")
        info_badge = Text.from_markup(f"Режим: [bold yellow]{self.mode}[/]  |  [bold green]● REC[/]  |  Ввод: [bold blue]{self.backend}[/]")
        header_table.add_row(title_text, status_badge, info_badge)

        header_panel = Panel(
            header_table,
            box=box.ROUNDED,
            border_style="bright_blue",
            padding=(0, 1),
        )

        # 2. Main Stats (Two Columns)
        # Left: Telemetry & Timing
        left_table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        left_table.add_column("Key", style="dim white", width=22)
        left_table.add_column("Value", style="bold white")

        fps_color = "bright_green" if current_fps >= 100 else ("yellow" if current_fps >= 60 else "red")
        left_table.add_row("Захват экрана (KMS):", f"[{fps_color}]{current_fps:5.1f} FPS[/]")
        left_table.add_row("Компенсация задержки:", f"[bright_cyan]{latency_ms:5.1f} мс[/]")
        left_table.add_row("Смещение в Great зоне:", f"[bright_magenta]{target_ratio*100:4.0f}%[/] (центр)")
        lmb_color = "bright_green" if lmb_held else "dim white"
        lmb_status = "🟢 ЗАЖАТ" if lmb_held else "⚪️ ОТЖАТ"
        left_table.add_row("Статус ЛКМ (Мышь):", f"[{lmb_color}]{lmb_status}[/]")
        speed_str = f"{speed:5.1f} °/с" if speed > 0 else "ожидание..."
        left_table.add_row("Скорость стрелки:", f"[white]{speed_str}[/]")

        left_panel = Panel(
            left_table,
            title="[bold cyan]⚡ Видеотракт и тайминг[/]",
            box=box.ROUNDED,
            border_style="cyan",
        )

        # Right: Accuracy & AI Stats
        right_table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        right_table.add_column("Key", style="dim white", width=22)
        right_table.add_column("Value", style="bold white")

        great_pct = (great / total * 100) if total > 0 else 0.0
        good_pct = (good / total * 100) if total > 0 else 0.0
        miss_pct = (miss / total * 100) if total > 0 else 0.0

        right_table.add_row("Всего проверок:", f"[white]{total}[/]")
        right_table.add_row("Идеально (GREAT):", f"[bright_green]{great:3d} ({great_pct:5.1f}%)[/]")
        right_table.add_row("Успешно (GOOD):", f"[bright_yellow]{good:3d} ({good_pct:5.1f}%)[/]")
        right_table.add_row("Промахи (MISS):", f"[bright_red]{miss:3d} ({miss_pct:5.1f}%)[/]")
        if aborted > 0:
            right_table.add_row("Отменено (ЛКМ):", f"[bright_cyan]{aborted:3d}[/] [dim](пропуск)[/]")

        frenzy_str = f"{chain_count}x" if chain_count > 1 else "нет"
        right_table.add_row("Серия (Frenzy):", f"[bright_yellow]{frenzy_str}[/] [dim](Рекорд: {max_chain}x)[/]")

        right_panel = Panel(
            right_table,
            title="[bold green]🎯 Точность и статистика[/]",
            box=box.ROUNDED,
            border_style="green",
        )

        metrics_table = Table.grid(expand=True)
        metrics_table.add_column(ratio=1)
        metrics_table.add_column(ratio=1)
        metrics_table.add_row(left_panel, right_panel)

        # 3. Hit Deviation Gauge
        gauge_panel = self._build_gauge_panel(last_outcome, last_target, last_hit, last_err, last_err_ms)

        # 4. Recent Events Log
        log_content = Table.grid(expand=True)
        log_content.add_column()
        if not events:
            log_content.add_row(Text.from_markup("[dim italic]Пока нет событий...[/]"))
        else:
            for ev in events:
                log_content.add_row(Text.from_markup(ev))

        log_panel = Panel(
            log_content,
            title="[bold yellow]📜 Журнал событий[/]",
            box=box.ROUNDED,
            border_style="yellow",
            padding=(0, 1),
        )

        # 5. Footer / Controls Helper
        footer_table = Table.grid(expand=True)
        footer_table.add_column(justify="center")
        footer_table.add_row(
            Text.from_markup(
                "[dim]Управление:[/] [bold yellow]+/-[/] [dim]Упреждение (±4мс: + раньше / - позже)  │[/] "
                "[bold yellow][[/][bold yellow]][/] [dim]Смещение (±5%)  │[/] "
                "[bold yellow]Q[/] [dim]/[/] [bold yellow]Ctrl+C[/] [dim]Выход[/]"
            )
        )
        footer_panel = Panel(footer_table, box=box.ROUNDED, border_style="dim white", padding=(0, 1))

        # Combine all into main layout group
        main_group = Group(
            header_panel,
            metrics_table,
            gauge_panel,
            log_panel,
            footer_panel,
        )

        return Panel(
            main_group,
            box=box.HEAVY,
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_gauge_panel(
        self,
        outcome: str,
        target_angle: float,
        hit_angle: float,
        err_deg: float,
        err_ms: float,
    ) -> Panel:
        if outcome == "WAITING":
            bar_text = "[dim]───────────────[ Ожидание первого скилл чека ]───────────────[/]"
            desc = "[dim]Шкала отклонения активируется после первого удара по пробелу[/]"
        else:
            # Range: -12.0° to +12.0° (31 discrete slots)
            width = 31
            center_idx = width // 2  # 15
            clamped_err = max(-12.0, min(12.0, err_deg))
            slot = int(round(center_idx + (clamped_err / 12.0) * (center_idx - 2)))
            slot = max(0, min(width - 1, slot))

            # Great zone is roughly ±4.75° around center
            great_half = int(round((4.75 / 12.0) * (center_idx - 2)))
            g_left = max(0, center_idx - great_half)
            g_right = min(width - 1, center_idx + great_half)

            bar_chars = []
            for i in range(width):
                if i == slot:
                    bar_chars.append("[bold bright_white on red]●[/]" if outcome == "MISS" else "[bold bright_white on green]●[/]")
                elif i == center_idx:
                    bar_chars.append("[bright_cyan]│[/]")
                elif i in (g_left, g_right):
                    bar_chars.append("[bright_green]│[/]")
                elif g_left < i < g_right:
                    bar_chars.append("[green]─[/]")
                else:
                    bar_chars.append("[dim]─[/]")

            bar_str = "".join(bar_chars)
            bar_text = f"[yellow]Раньше (-12°)[/]  {bar_str}  [yellow]Позже (+12°)[/]"

            outcome_color = "bright_green" if outcome == "GREAT" else ("bright_yellow" if outcome == "GOOD" else "bright_red")
            sign = "+" if err_deg >= 0 else ""
            desc = (
                f"Результат: [{outcome_color} bold]{outcome}[/]  │  "
                f"Цель: [white]{target_angle:.1f}°[/]  │  "
                f"Факт: [white]{hit_angle:.1f}°[/]  │  "
                f"Ошибка: [{outcome_color} bold]{sign}{err_deg:.1f}° ({sign}{err_ms:.1f}мс)[/]"
            )

        gauge_table = Table.grid(expand=True)
        gauge_table.add_column(justify="center")
        gauge_table.add_row(Text.from_markup(bar_text))
        gauge_table.add_row(Text.from_markup(desc))

        return Panel(
            gauge_table,
            title="[bold magenta]🎯 Шкала точности последнего удара[/]",
            box=box.ROUNDED,
            border_style="magenta",
            padding=(0, 1),
        )

    def print_summary(self):
        """Prints a final summary table to stdout upon shutdown."""
        with self._lock:
            total = self.total_checks
            great = self.great_count
            good = self.good_count
            miss = self.miss_count
            max_chain = self.max_chain
            final_lat = self.latency_ms
            req_lat = self.requested_base_latency
            actual_delays = list(self.actual_used_delays)
            overrides_cnt = self.profile_overrides_count
            recent_events = list(self.events)[-5:]

        table = Table(title="📊 ИТОГИ ИГРОВОЙ СЕССИИ SKILL CHECK AI", box=box.ROUNDED)
        table.add_column("Метрика", style="cyan")
        table.add_column("Значение", style="bold white")

        great_pct = (great / total * 100) if total > 0 else 0.0
        good_pct = (good / total * 100) if total > 0 else 0.0
        miss_pct = (miss / total * 100) if total > 0 else 0.0

        table.add_row("Всего скилл чеков", str(total))
        table.add_row("Идеально (GREAT)", f"[green]{great} ({great_pct:.1f}%)[/]")
        table.add_row("Успешно (GOOD)", f"[yellow]{good} ({good_pct:.1f}%)[/]")
        table.add_row("Промахи (MISS)", f"[red]{miss} ({miss_pct:.1f}%)[/]")
        table.add_row("Максимальная серия", f"[yellow]{max_chain}x[/]")

        if actual_delays:
            min_d = min(actual_delays)
            try:
                med_d = float(np.median(actual_delays))
            except Exception:
                import statistics
                med_d = float(statistics.median(actual_delays))
            max_d = max(actual_delays)
            actual_str = f"min={min_d:.1f} / med={med_d:.1f} / max={max_d:.1f} мс"
        else:
            actual_str = f"{final_lat:.1f} мс (нет выстрелов)"

        table.add_row("Requested Base Latency", f"[cyan]{req_lat:.1f} мс[/]")
        table.add_row("Actual Used Delay", f"[magenta]{actual_str}[/]")
        table.add_row("Profile Overrides Count", f"[yellow]{overrides_cnt}[/]")

        self.console.print()
        self.console.print(table)
        if recent_events:
            self.console.print("[bold yellow]Последние события сессии:[/]")
            for ev in recent_events:
                self.console.print(f"  {ev}")
        self.console.print()
