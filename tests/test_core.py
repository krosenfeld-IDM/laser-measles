from laser_measles import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"
