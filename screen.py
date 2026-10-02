"""Find the Start button, the ball, and the hoop in a screenshot.

Everything is located by color rather than by matching saved images, so it
works at any screen size or window position.
"""
import cv2
import numpy as np

# The ball rests in the bottom part of the screen; the hoop and scoreboard
# above it have orange-ish details too.
BALL_SEARCH_TOP = 0.5
# Fraction of the window width the ball must span. The resting ball is about
# 0.25; balls bouncing back off the wall are under 0.15.
BALL_MIN_WIDTH = 0.18
# The backboard and pole are one white shape, far bigger than the white
# scoreboard border and buttons.
HOOP_MIN_AREA = 0.02


def blobs(mask):
    """(x, y, w, h, fill) for each connected region of a boolean mask."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    return [(x, y, w, h, area / (w * h)) for x, y, w, h, area in stats[1:]]


def find_start_button(image):
    """Center of the orange Start pill, or None."""
    r, g, b = (image[..., i].astype(int) for i in range(3))
    mask = (r > 180) & (g > 90) & (g < 160) & (b < 60)
    min_width = image.shape[1] * 0.15
    for x, y, w, h, fill in blobs(mask):
        if w > min_width and 1.8 < w / h < 3.5 and fill > 0.6:
            return x + w // 2, y + h // 2
    return None


def find_ball(image):
    """(center x, center y, diameter) of the basketball in the lower part of
    the screen, or None. The ball's dark seams split its orange into pieces,
    so the mask is closed before looking for a round blob."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = ((h >= 8) & (h <= 25) & (s > 120) & (v > 120)).astype(np.uint8)
    mask[: int(image.shape[0] * BALL_SEARCH_TOP)] = 0
    kernel_size = max(3, image.shape[1] // 50)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
    min_width = image.shape[1] * BALL_MIN_WIDTH
    balls = [
        (x, y, w, h)
        for x, y, w, h, fill in blobs(mask)
        if w > min_width and 0.75 < w / h < 1.33 and fill > 0.55
    ]
    if not balls:
        return None
    x, y, w, h = max(balls, key=lambda box: box[2] * box[3])
    return x + w / 2, y + h / 2, (w + h) / 2


def find_hoop(image):
    """Center x of the backboard, which the rim hangs from, or None."""
    white = (image.min(axis=2) > 200).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(white)
    min_area = HOOP_MIN_AREA * image.shape[0] * image.shape[1]
    boards = [(x, w, area) for x, y, w, h, area in stats[1:] if area > min_area]
    if not boards:
        return None
    x, w, _ = max(boards, key=lambda board: board[2])
    return x + w / 2
