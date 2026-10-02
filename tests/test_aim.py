import random

from aim import crossing_x, predict_hoop_x, tilt_to_hit


def test_the_chosen_tilt_always_crosses_at_the_rim():
    rng = random.Random(0)
    for _ in range(200):
        ball_x, rim_x = rng.uniform(0.1, 0.9), rng.uniform(0.2, 0.8)

        tilt = tilt_to_hit(ball_x, rim_x)

        assert abs(crossing_x(ball_x, tilt) - rim_x) < 1e-12


def test_a_ball_off_to_one_side_is_swiped_back_toward_the_hoop():
    assert tilt_to_hit(0.8, 0.5) < 0 < tilt_to_hit(0.2, 0.5)


def test_a_still_hoop_is_aimed_at_where_it_is():
    history = [(t / 10, 0.5) for t in range(10)]

    assert predict_hoop_x(history, at=2.0) == 0.5


def test_a_moving_hoop_is_led_by_its_speed():
    history = [(t / 10, 0.3 + 0.1 * t / 10) for t in range(10)]

    # Moving 0.1 per second, so one second past the last sighting adds 0.1.
    assert abs(predict_hoop_x(history, at=1.9) - (0.39 + 0.1)) < 1e-9


def test_a_ball_blocking_the_backboard_doesnt_make_the_hoop_look_moving():
    history = [(t / 50, 0.5) for t in range(15)]
    history[-1] = (history[-1][0], 0.42)

    assert predict_hoop_x(history, at=1.3) == 0.5
