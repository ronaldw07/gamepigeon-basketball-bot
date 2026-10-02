"""Play a GamePigeon Basketball round through iPhone Mirroring."""
import argparse
import json
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

from aim import FLIGHT_SECONDS, predict_hoop_x, tilt_to_hit
from mirror import (
    TOUCH_DOWN_PAUSE,
    capture,
    click,
    find_mirroring_window,
    flick,
    start_kill_switch,
    stop_requested,
    to_screen_points,
)
from screen import find_ball, find_hoop, find_start_button

ROUND_SECONDS = 45
START_WAIT_SECONDS = 10
# Upward swipe length in window widths, and how long it takes. Every swipe
# faster than about 0.15s throws the same distance, so these only need to be
# comfortably quick.
SWIPE_LENGTH = 0.35
SWIPE_SECONDS = 0.08
# A new ball appears about 0.6s after a throw; until then the thrown ball can
# still show at the spot it left from.
RESPAWN_SECONDS = 0.5
# The ball must move less than this many pixels between captures to be resting.
STILL_PIXELS = 3
# Hoop sightings kept for working out its speed, about two seconds' worth.
HOOP_HISTORY = 120
# Time for the last throw to land before a recording stops.
LANDING_SECONDS = 2
POLL_SECONDS = 0.005
HERE = Path(__file__).resolve().parent
LOG_FILE = HERE / "throws.jsonl"


class FrameFeed(threading.Thread):
    """Captures the mirroring window nonstop in the background so the main
    loop always has a fresh frame, and optionally records every frame."""

    def __init__(self, video_path=None):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_times = []
        self.latest = None
        self.done = threading.Event()

    def run(self):
        writer = size = None
        while not self.done.is_set():
            window, image = capture()
            if image is None:
                time.sleep(0.1)
                continue
            taken = time.perf_counter()
            self.latest = (taken, window, image)
            if self.video_path:
                if writer is None:
                    size = (image.shape[1], image.shape[0])
                    writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"MJPG"), 60, size)
                frame = image if (image.shape[1], image.shape[0]) == size else cv2.resize(image, size)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.frame_times.append(taken)
        if writer is not None:
            writer.release()

    def next_frame(self, after):
        """(time, window, image) of the first capture taken after `after`,
        or None if stopped."""
        while not stop_requested.is_set():
            latest = self.latest
            if latest is not None and latest[0] > after:
                return latest
            time.sleep(POLL_SECONDS)
        return None


def press_start(feed):
    """Press Start once it shows. Returns when it was pressed, or None."""
    deadline = time.perf_counter() + START_WAIT_SECONDS
    taken = 0
    while time.perf_counter() < deadline:
        frame = feed.next_frame(taken)
        if frame is None:
            return None
        taken, window, image = frame
        start = find_start_button(image)
        if start is not None:
            click(to_screen_points(start, window, image))
            return time.perf_counter()
    return None


def shoot(ball, hoop_history, window, image):
    """Swipe the ball toward where the hoop will be when it gets there.
    Returns what was thrown, for the log."""
    width = image.shape[1]
    released = time.perf_counter() + TOUCH_DOWN_PAUSE + SWIPE_SECONDS
    rim_x = predict_hoop_x(hoop_history, at=released + FLIGHT_SECONDS)
    tilt = tilt_to_hit(ball[0] / width, rim_x)
    length = SWIPE_LENGTH * width
    end = (ball[0] + tilt * length, ball[1] - length)
    flick(to_screen_points(ball[:2], window, image), to_screen_points(end, window, image), SWIPE_SECONDS)
    return {
        "released": time.perf_counter(),
        "ball": [v / width for v in ball],
        "hoop_x": hoop_history[-1][1],
        "rim_x": rim_x,
        "tilt": tilt,
        "window": window,
    }


def play_round(feed, round_end):
    """Throw every ball as soon as it comes to rest until the round ends."""
    hoop_history = deque(maxlen=HOOP_HISTORY)
    throws, previous_ball, last_release, taken = [], None, 0, 0
    while True:
        frame = feed.next_frame(taken)
        if frame is None:
            break
        taken, window, image = frame
        if taken > round_end:
            break
        hoop = find_hoop(image)
        if hoop is not None:
            hoop_history.append((taken, hoop / image.shape[1]))
        ball = find_ball(image) if taken - last_release > RESPAWN_SECONDS else None
        resting = (
            ball is not None and previous_ball is not None
            and abs(ball[0] - previous_ball[0]) <= STILL_PIXELS
            and abs(ball[1] - previous_ball[1]) <= STILL_PIXELS
        )
        previous_ball = ball
        if not resting or not hoop_history:
            continue
        throws.append(shoot(ball, hoop_history, window, image))
        last_release, previous_ball = throws[-1]["released"], None
    return throws


def save(throws, pressed, feed, folder):
    with LOG_FILE.open("a") as log:
        for throw in throws:
            log.write(json.dumps({**throw, "released": throw["released"] - pressed, "round": folder.name}) + "\n")
    if feed.video_path is None:
        return
    # Same layout the analysis scripts in debug/ read.
    entries = [
        {
            "sideways": t["tilt"] * SWIPE_LENGTH, "length": SWIPE_LENGTH, "duration": SWIPE_SECONDS,
            "ball": [v * t["window"][2] for v in t["ball"]],
            "ball_px": [v * t["window"][2] for v in t["ball"][:2]],
            "released": t["released"] - pressed, "rim_x": t["rim_x"], "hoop_x": t["hoop_x"],
        }
        for t in throws
    ]
    times = [t - pressed for t in feed.frame_times]
    (folder / "log.json").write_text(json.dumps({"throws": entries, "frame_times": times}))


def main():
    parser = argparse.ArgumentParser(description="Play a GamePigeon Basketball round through iPhone Mirroring.")
    parser.add_argument("--record", action="store_true", help="save video of the round to debug/ for analysis")
    args = parser.parse_args()

    start_kill_switch()
    subprocess.run(["open", "-a", "iPhone Mirroring"], check=False)
    time.sleep(0.5)
    if find_mirroring_window() is None:
        print("iPhone Mirroring isn't open. Open it, open a Basketball game, and try again.")
        return

    folder = HERE / "debug" / datetime.now().strftime("round-%Y%m%d-%H%M%S")
    if args.record:
        folder.mkdir(parents=True)
    feed = FrameFeed(folder / "round.avi" if args.record else None)
    feed.start()
    pressed = press_start(feed)
    if pressed is None:
        print("No Start button. Open a Basketball game to the Round screen and try again.")
        feed.done.set()
        return

    throws = play_round(feed, pressed + ROUND_SECONDS)
    if args.record:
        time.sleep(LANDING_SECONDS)
    feed.done.set()
    feed.join()
    save(throws, pressed, feed, folder)
    print(f"{len(throws)} throws." + (f" Recording in {folder}" if args.record else ""))


if __name__ == "__main__":
    main()
