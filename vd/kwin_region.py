"""Build and register the opt-in KWin region capture helper."""
import os
from pathlib import Path
import shutil
import subprocess


APP_ID = 'vd-region-capture'
INTERFACE = 'zkde_screencast_unstable_v1'


def ensure_kwin_region_worker():
    root = Path(__file__).resolve().parents[1]
    if os.environ.get('XDG_SESSION_TYPE') != 'wayland' or 'KDE' not in os.environ.get('XDG_CURRENT_DESKTOP', ''):
        raise RuntimeError('Region capture requires a KDE Wayland session')

    worker = root / 'build' / 'kwin-region' / 'kwin-region-worker'
    build = root / 'native' / 'build_kwin_region_worker.sh'
    sources = (root / 'native' / 'kwin_region_worker.cpp', build)
    rebuild = not worker.is_file() or any(
        path.is_file() and path.stat().st_mtime_ns > worker.stat().st_mtime_ns
        for path in sources
    )
    if rebuild:
        try:
            result = subprocess.run([str(build)], check=False, capture_output=True,
                                    text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f'Cannot build KWin region worker: {exc}') from exc
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError('Cannot build KWin region worker; install the KDE, '
                               'PipeWire, Wayland and C++ development packages.'
                               + (f'\n{detail}' if detail else ''))

    if not worker.is_file():
        raise RuntimeError(f'KWin region worker was not created: {worker}')

    data_home = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
    applications = data_home / 'applications'
    applications.mkdir(parents=True, exist_ok=True)
    desktop_file = applications / f'{APP_ID}.desktop'
    exec_path = str(worker).replace('\\', '\\\\').replace('"', '\\"')
    content = ('[Desktop Entry]\n'
               'Type=Application\n'
               'Name=VD Region Capture Worker\n'
               f'Exec="{exec_path}"\n'
               'NoDisplay=true\n'
               f'X-KDE-Wayland-Interfaces={INTERFACE}\n')
    previous = None
    if desktop_file.exists():
        existing = desktop_file.read_text(errors='replace')
        if 'Name=VD Region Capture Worker' not in existing:
            raise RuntimeError(f'KDE desktop entry conflicts with {desktop_file}')
        if existing != content:
            previous = existing
    if previous is not None or not desktop_file.exists():
        desktop_file.write_text(content)
        refresh = shutil.which('kbuildsycoca6')
        if refresh is None:
            if previous is None:
                desktop_file.unlink(missing_ok=True)
            else:
                desktop_file.write_text(previous)
            raise RuntimeError('kbuildsycoca6 is required to register KWin region access')
        try:
            result = subprocess.run([refresh, '--noincremental'], capture_output=True,
                                    text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = None
            detail = str(exc)
        else:
            detail = (result.stderr or result.stdout).strip() if result else ''
        if result is None or result.returncode:
            if previous is None:
                desktop_file.unlink(missing_ok=True)
            else:
                desktop_file.write_text(previous)
            raise RuntimeError('KDE could not register the region capture helper: ' + detail)
    return worker
