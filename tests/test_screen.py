from pathlib import Path

import numpy as np
from PIL import Image

from screen import find_ball, find_hoop, find_start_button

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name):
    return np.array(Image.open(FIXTURES / name).convert("RGB"))


def test_start_button_found_only_before_the_round():
    assert find_start_button(load("round_start.png")) is not None
    assert find_start_button(load("playing.png")) is None


def test_finds_the_resting_ball_where_it_sits():
    x, y, diameter = find_ball(load("playing.png"))

    # The ball sits at the bottom right, about a quarter of the window wide.
    assert abs(x - 329) < 6 and abs(y - 778) < 6
    assert 95 < diameter < 120


def test_finds_the_hoop_at_the_backboard_center():
    image = load("playing.png")

    hoop_x = find_hoop(image)

    assert abs(hoop_x - image.shape[1] / 2) < 6


def test_finds_everything_at_any_window_size():
    original = load("playing.png")
    ball = find_ball(original)
    hoop = find_hoop(original)

    # A smaller window, a bigger one, and a Retina display's 2x pixels.
    for scale in (0.75, 1.5, 2.0):
        image = np.array(Image.fromarray(original).resize(
            (round(original.shape[1] * scale), round(original.shape[0] * scale)), Image.BILINEAR,
        ))

        scaled_ball = find_ball(image)
        scaled_hoop = find_hoop(image)

        # Same spot as a fraction of the window, within 1% of its width.
        tolerance = 0.01 * image.shape[1]
        assert abs(scaled_ball[0] - ball[0] * scale) < tolerance
        assert abs(scaled_ball[1] - ball[1] * scale) < tolerance
        assert abs(scaled_hoop - hoop * scale) < tolerance
