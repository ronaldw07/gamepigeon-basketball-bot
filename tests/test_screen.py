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
