from enhancer.models.safmn import choose_scale


def test_single_scale_always_picked() -> None:
    assert choose_scale(640, [4], 1920) == 4


def test_picks_scale_closest_to_target() -> None:
    assert choose_scale(960, [2, 4], 1920) == 2
    assert choose_scale(400, [2, 4], 1920) == 4
    assert choose_scale(1200, [2, 4], 1920) == 2


def test_tie_prefers_smaller_scale() -> None:
    assert choose_scale(640, [2, 4], 1920) == 2
