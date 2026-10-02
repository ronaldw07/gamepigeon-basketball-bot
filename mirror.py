"""Find and capture the iPhone Mirroring window, and send it mouse input.

Copied from gamepigeon-wordhunt-bot so this repo stands alone.
"""
import threading
import time

import numpy as np
from pynput import keyboard
from Quartz import (
    CGDataProviderCopyData,
    CGEventCreateMouseEvent,
    CGEventPost,
    CGImageGetBytesPerRow,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGRectNull,
    CGWindowListCopyWindowInfo,
    CGWindowListCreateImage,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseDragged,
    kCGEventLeftMouseUp,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
    kCGNullWindowID,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowListOptionIncludingWindow,
    kCGWindowListOptionOnScreenOnly,
)

CLICK_HOLD = 0.05
# Hold after touching down so the phone registers the touch before it moves.
TOUCH_DOWN_PAUSE = 0.02
# Time between drag events in a flick, about the phone's 120Hz touch rate.
DRAG_STEP_SECONDS = 0.008

stop_requested = threading.Event()


def start_kill_switch():
    def on_press(key):
        if key == keyboard.Key.esc:
            stop_requested.set()
            print("\nEsc pressed, stopping.")
            return False

    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.daemon = True
        listener.start()
        print("Press Esc at any time to stop.")
    except Exception as error:
        print(f"Esc kill switch unavailable ({error}). Press Ctrl+C in the terminal to stop.")


def find_mirroring_window():
    """(window id, bounds) of the iPhone Mirroring window, with bounds
    (x, y, w, h) in screen points on whichever display it is, or None."""
    windows = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    matches = [
        w for w in windows
        if w.get("kCGWindowOwnerName") == "iPhone Mirroring"
        and w.get("kCGWindowName") == "iPhone Mirroring"
    ]
    if not matches:
        return None
    w = max(matches, key=lambda w: w["kCGWindowBounds"]["Width"] * w["kCGWindowBounds"]["Height"])
    b = w["kCGWindowBounds"]
    return w["kCGWindowNumber"], (int(b["X"]), int(b["Y"]), int(b["Width"]), int(b["Height"]))


def screenshot(window_id):
    """Capture just the mirroring window as RGB, on any display and even if
    covered. In-process, so it takes milliseconds."""
    image = CGWindowListCreateImage(
        CGRectNull, kCGWindowListOptionIncludingWindow, window_id, kCGWindowImageBoundsIgnoreFraming,
    )
    width, height = CGImageGetWidth(image), CGImageGetHeight(image)
    data = CGDataProviderCopyData(CGImageGetDataProvider(image))
    bgra = np.frombuffer(data, dtype=np.uint8).reshape(height, CGImageGetBytesPerRow(image) // 4, 4)
    return np.ascontiguousarray(bgra[:, :width, 2::-1])


def capture():
    """(window bounds, image) taken together. The window moves and resizes as
    the phone changes screens, so positions are only valid for the bounds
    they were captured with."""
    found = find_mirroring_window()
    if found is None:
        return None, None
    window_id, window = found
    return window, screenshot(window_id)


def to_screen_points(pixel, window, image):
    """Screenshots are in display pixels (2x on Retina), the mouse moves in
    points measured from the main display's corner."""
    scale = image.shape[1] / window[2]
    return window[0] + pixel[0] / scale, window[1] + pixel[1] / scale


def post_mouse(kind, point):
    CGEventPost(kCGHIDEventTap, CGEventCreateMouseEvent(None, kind, point, kCGMouseButtonLeft))


def click(point):
    post_mouse(kCGEventLeftMouseDown, point)
    time.sleep(CLICK_HOLD)
    post_mouse(kCGEventLeftMouseUp, point)


def flick(start, end, duration):
    """Touch down at start and slide to end at a steady speed over duration
    seconds, then let go. Each drag event is sent on schedule rather than
    after a fixed sleep, so the speed the phone sees matches duration."""
    steps = max(2, round(duration / DRAG_STEP_SECONDS))
    post_mouse(kCGEventLeftMouseDown, start)
    time.sleep(TOUCH_DOWN_PAUSE)
    began = time.perf_counter()
    for i in range(1, steps + 1):
        fraction = i / steps
        wait = began + duration * fraction - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        point = (start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction)
        post_mouse(kCGEventLeftMouseDragged, point)
    post_mouse(kCGEventLeftMouseUp, end)
