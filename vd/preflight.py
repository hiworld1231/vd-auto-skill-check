"""Read-only environment report; never create UInput or start capture."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def inspect_system():
    import cv2
    import evdev
    checks=[]
    def check(name,ok,detail):
        checks.append(dict(name=name,ok=bool(ok),detail=detail))
    check('wayland',bool(os.environ.get('WAYLAND_DISPLAY')),os.environ.get('WAYLAND_DISPLAY'))
    probe="""import json, gi, dbus, numpy
 gi.require_version('Gst','1.0')
 from gi.repository import Gst
 Gst.init(None)
 print(json.dumps({name:bool(Gst.ElementFactory.find(name)) for name in ['pipewiresrc','videoconvert','appsink','videotestsrc']}))
""".replace('\n ','\n')
    try:
        result=subprocess.run(['/usr/bin/python','-c',probe],capture_output=True,text=True,timeout=10)
        plugins=json.loads(result.stdout) if result.returncode==0 else None
        check('system_capture_dependencies',plugins and all(plugins.values()),
              plugins if plugins else result.stderr.strip())
    except (OSError,subprocess.TimeoutExpired,ValueError) as exc:
        check('system_capture_dependencies',False,str(exc))
    template=Path(__file__).resolve().parents[1]/'assets/space_template.png'
    check('prompt_template',cv2.imread(str(template)) is not None,str(template))
    mice=[]
    errors=[]
    for path in evdev.list_devices():
        device=None
        try:
            device=evdev.InputDevice(path)
            if evdev.ecodes.BTN_LEFT in device.capabilities().get(evdev.ecodes.EV_KEY,[]):
                mice.append(dict(path=path,name=device.name))
        except OSError as exc:
            errors.append(str(exc))
        finally:
            if device is not None:
                device.close()
    check('mouse_read_access',bool(mice),dict(devices=mice,errors=errors))
    check('uinput_write_access',os.access('/dev/uinput',os.W_OK),'/dev/uinput (not opened)')
    versions={name:importlib.metadata.version(name)
              for name in ('numpy','opencv-python-headless','evdev')}
    return dict(platform=platform.platform(),python=platform.python_version(),
                versions=versions,checks=checks,ok=all(c['ok'] for c in checks),
                physical_input_tested=False,portal_selection_tested=False)
