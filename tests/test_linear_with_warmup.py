import pytest

from rau.training.linear_with_warmup_lr_scheduler import LinearWithWarmupFunction

def test_linear_with_warmup_function() -> None:
    f = LinearWithWarmupFunction(0, 10, 100)
    f_0 = f(0)
    assert 0 < f_0 < 1
    f.counter = 1
    f_1 = f(0)
    assert f_0 < f_1 < 1
    f.counter = 5
    f_5 = f(0)
    assert f_5 == f(0)
    assert f_1 < f_5 < 1
    f.counter = 10
    assert f(0) == 1
    f.counter = 11
    f_11 = f(0)
    assert 0 < f_11 < 1
    f.counter = 80
    f_80 = f(0)
    assert 0 < f_80 < f_11
    f.counter = 99
    f_99 = f(0)
    assert 0 < f_99 < f_80
    f.counter = 100
    assert 0 < f(0) < f_99
    f.counter = 101
    with pytest.raises(ValueError):
        f(0)
