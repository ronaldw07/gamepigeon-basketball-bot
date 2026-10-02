"""Which way to swipe so the ball goes through the hoop.

Every shot from a settled ball flies the same distance: swipe speed and
length don't change it (tested from 2.5 to 17 window widths per second).
Only the swipe's direction matters. Positions are fractions of the window
width.

Fitted from 31 tracked throws in a recorded round: a ball resting at ball_x,
swiped with tilt (sideways travel per unit of upward travel), crosses rim
height at VANISHING_X + SPREAD * (ball_x - VANISHING_X) + TILT_GAIN * tilt,
to within about half a percent of the window width.
"""
VANISHING_X = 0.515
SPREAD = 0.743
TILT_GAIN = 0.807
# Seconds from letting go to the ball reaching the rim, as seen in captures.
FLIGHT_SECONDS = 1.03
# How far back to look when measuring the hoop's speed.
HOOP_SPEED_WINDOW = 0.3


def crossing_x(ball_x, tilt):
    """Where a throw crosses rim height."""
    return VANISHING_X + SPREAD * (ball_x - VANISHING_X) + TILT_GAIN * tilt


def tilt_to_hit(ball_x, rim_x):
    """The swipe tilt that makes a ball at ball_x cross rim height at rim_x."""
    return (rim_x - VANISHING_X - SPREAD * (ball_x - VANISHING_X)) / TILT_GAIN


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def predict_hoop_x(history, at):
    """Where the hoop will be at time `at`, from recent (time, x) sightings.
    A ball passing in front of the backboard throws a sighting off, so speed
    and position are medians (a Theil-Sen line) rather than a plain fit."""
    latest_time = history[-1][0]
    recent = [(t, x) for t, x in history if latest_time - t <= HOOP_SPEED_WINDOW]
    slopes = [
        (x2 - x1) / (t2 - t1)
        for i, (t1, x1) in enumerate(recent)
        for t2, x2 in recent[i + 1:]
        if t2 > t1
    ]
    speed = median(slopes) if slopes else 0
    position = median([x + speed * (latest_time - t) for t, x in recent])
    return position + speed * (at - latest_time)
