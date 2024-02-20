from rau.training.early_stopping import UpdatesWithoutImprovement

def test_updates_without_improvement_min():
    c = UpdatesWithoutImprovement(mode='min', patience=3)
    assert c.update(100) == (True, False)
    assert c.update(95) == (True, False)
    assert c.update(98) == (False, False)
    assert c.update(90) == (True, False)
    assert c.update(92) == (False, False)
    assert c.update(90) == (False, False)
    assert c.update(85) == (True, False)
    assert c.update(87) == (False, False)
    assert c.update(88) == (False, False)
    assert c.update(86) == (False, True)

def test_updates_without_improvement_max():
    c = UpdatesWithoutImprovement(mode='max', patience=3)
    assert c.update(10) == (True, False)
    assert c.update(15) == (True, False)
    assert c.update(11) == (False, False)
    assert c.update(12) == (False, False)
    assert c.update(15) == (False, True)
