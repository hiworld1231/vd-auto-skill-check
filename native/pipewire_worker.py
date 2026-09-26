#!/usr/bin/python
"""Uncompressed portal/PipeWire capture. stdout is framed binary, stderr is text.

Runs with the system Python providing distro GI/DBus bindings. Never imports
legacy VD code or injects input. A synthetic source exercises the same transport.
"""
import argparse
import json
import os
import struct
import sys
import time
import uuid

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gst, GLib, GstVideo
import numpy as np

Gst.init(None)


def portal_video_caps(fps):
    return f'video/x-raw,framerate=0/1,max-framerate={fps}/1'


def capture_pipeline(source):
    return Gst.parse_launch(source +
        ' ! videoconvert ! video/x-raw,format=BGR' +
        ' ! appsink name=frames sync=false max-buffers=1 drop=true')


def open_portal():
    import dbus
    from dbus.mainloop.glib import DBusGMainLoop
    DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()
    destination = 'org.freedesktop.portal.Desktop'
    path = '/org/freedesktop/portal/desktop'
    portal = dbus.Interface(bus.get_object(destination, path),
                            'org.freedesktop.portal.ScreenCast')
    session = None

    def request(method, *args, **options):
        token = 'vd_' + uuid.uuid4().hex
        sender = bus.get_unique_name()[1:].replace('.', '_')
        response_path = path + '/request/' + sender + '/' + token
        loop = GLib.MainLoop()
        result = []
        expired = False
        def response(code, values):
            result.append((int(code), values))
            loop.quit()
        match = bus.add_signal_receiver(response, signal_name='Response',
            dbus_interface='org.freedesktop.portal.Request', path=response_path)
        def timeout():
            nonlocal expired
            expired = True
            loop.quit()
            return False
        timer = GLib.timeout_add_seconds(60, timeout)
        try:
            options['handle_token'] = token
            method(*args, dbus.Dictionary(options, signature='sv'))
            if not result:
                loop.run()
            if not result:
                raise RuntimeError('Screen selection timed out')
            code, values = result[0]
            if code:
                raise RuntimeError('Screen selection cancelled or rejected: ' + str(code))
            return values
        finally:
            match.remove()
            if not expired:
                GLib.source_remove(timer)
            if not result:
                try:
                    dbus.Interface(bus.get_object(destination, response_path),
                                   'org.freedesktop.portal.Request').Close()
                except dbus.DBusException:
                    pass

    try:
        session = request(portal.CreateSession,
                          session_handle_token='vd_' + uuid.uuid4().hex)['session_handle']
        request(portal.SelectSources, session, types=dbus.UInt32(1),
                multiple=dbus.Boolean(False), cursor_mode=dbus.UInt32(1))
        print('Select the game monitor in the KDE screen-sharing dialog.',
              file=sys.stderr, flush=True)
        result = request(portal.Start, session, '')
        streams = result['streams']
        if len(streams) != 1:
            raise RuntimeError('Exactly one monitor is required')
        node = int(streams[0][0])
        fd = portal.OpenPipeWireRemote(session, dbus.Dictionary({}, signature='sv')).take()
        def close():
            try:
                dbus.Interface(bus.get_object(destination, session),
                               'org.freedesktop.portal.Session').Close()
            finally:
                os.close(fd)
        return fd, node, close
    except BaseException:
        if session is not None:
            try:
                dbus.Interface(bus.get_object(destination, session),
                               'org.freedesktop.portal.Session').Close()
            except Exception:
                pass
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--frames', type=int, default=0)
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--roi', default='800,420,320,240')
    args = ap.parse_args()
    if not 1 <= args.fps <= 240:
        raise ValueError('FPS must be in [1, 240]')
    left, top, width, height = map(int, args.roi.split(','))
    if min(left, top) < 0 or min(width, height) <= 0:
        raise ValueError('Invalid ROI')
    close = lambda: None
    pipeline = None
    try:
        if args.synthetic:
            source = f'videotestsrc is-live=true pattern=ball ! video/x-raw,width=1920,height=1080,framerate={args.fps}/1'
        else:
            fd, node, close = open_portal()
            source = (f'pipewiresrc fd={fd} path={node} do-timestamp=false'
                      f' ! {portal_video_caps(args.fps)}')
            print(f'Requesting {portal_video_caps(args.fps)}', file=sys.stderr, flush=True)
        pipeline = capture_pipeline(source)
        sink = pipeline.get_by_name('frames')
        bus = pipeline.get_bus()
        pipeline.set_state(Gst.State.PLAYING)
        count = 0
        last_sample = time.monotonic()
        while not args.frames or count < args.frames:
            while GLib.MainContext.default().pending():
                GLib.MainContext.default().iteration(False)
            sample = sink.emit('try-pull-sample', 100 * Gst.MSECOND)
            if sample is None:
                message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    raise RuntimeError(str(message.parse_error()) if message.type == Gst.MessageType.ERROR else 'Capture ended')
                if time.monotonic() - last_sample > 10:
                    raise RuntimeError('No video frames for 10 seconds')
                continue
            last_sample = time.monotonic()
            if count == 0:
                print(f'Capture negotiated: {sample.get_caps().to_string()}',
                      file=sys.stderr, flush=True)
            b = sample.get_buffer()
            caps = sample.get_caps().get_structure(0)
            w, h = caps.get_value('width'), caps.get_value('height')
            if left + width > w or top + height > h:
                raise RuntimeError(f'ROI outside captured monitor {w}x{h}')
            video_info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
            ok, mapping = b.map(Gst.MapFlags.READ)
            if not ok:
                raise RuntimeError('Cannot map video buffer')
            try:
                received_ns = time.monotonic_ns()
                clock = pipeline.get_clock()
                clock_now_ns = clock.get_time()
                segment = sample.get_segment()
                running_pts = segment.to_running_time(Gst.Format.TIME, b.pts)
                pts_valid = b.pts != Gst.CLOCK_TIME_NONE and running_pts != Gst.CLOCK_TIME_NONE
                source_ns = (received_ns - (clock_now_ns - pipeline.get_base_time() - running_pts)
                             if pts_valid else None)
                pixels = np.ndarray((h, w, 3), dtype=np.uint8,
                    buffer=mapping.data, offset=video_info.offset[0],
                    strides=(video_info.stride[0], 3, 1))
                payload = pixels[top:top + height, left:left + width].tobytes()
                header = json.dumps(dict(seq=count, width=width, height=height,
                    bytes=len(payload), stride=width * 3,
                    pts_ns=int(b.pts) if pts_valid else None,
                    negotiated_caps=sample.get_caps().to_string(),
                    requested_fps=args.fps,
                    media_monotonic_ns=source_ns, received_ns=received_ns,
                    timestamp_kind='gst_running_time_mapped' if pts_valid else 'arrival_only',
                    synthetic=args.synthetic), separators=(',', ':')).encode()
                sys.stdout.buffer.write(struct.pack('!I', len(header)))
                sys.stdout.buffer.write(header)
                sys.stdout.buffer.write(payload)
                sys.stdout.buffer.flush()
                count += 1
            finally:
                b.unmap(mapping)
    finally:
        if pipeline is not None:
            pipeline.set_state(Gst.State.NULL)
        close()


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, BrokenPipeError):
        pass
    except Exception as exc:
        print(f'CAPTURE ERROR: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
