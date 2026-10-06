import numpy as np
import pytest

from integrations.spatial_action_sampling import public_capital_threat_gather_bonus


def _position():
    values = np.zeros(16 * 441, np.float32)
    legal = np.zeros(3529, bool)
    general = 10 * 21 + 10
    adjacent = general + 1
    outer = general + 2
    enemy = general + 3
    for cell, count in ((general, 5), (adjacent, 10), (outer, 10), (enemy, 15)):
        values[cell] = np.log1p(count) / 8
    values[1 * 441 + general] = 1
    values[4 * 441 + general] = 1
    values[4 * 441 + adjacent] = 1
    values[4 * 441 + outer] = 1
    values[5 * 441 + enemy] = 1
    direct = 2 * 441 + adjacent
    approach = 2 * 441 + outer
    legal[[direct, approach, 1764 + direct, 1764 + approach, 3528]] = True
    return values, legal, direct, approach, enemy


def test_capital_gather_uses_only_public_threat_and_legal_full_routes():
    values, legal, direct, approach, enemy = _position()
    bonus = public_capital_threat_gather_bonus(values, legal, 4.0, np)
    assert bonus.shape == (3529,)
    assert bonus[direct] == pytest.approx(2 * bonus[approach])
    assert bonus[approach] > 0
    assert np.count_nonzero(bonus[1764:]) == 0
    assert np.count_nonzero(bonus) == 2

    legal[direct] = False
    assert public_capital_threat_gather_bonus(values, legal, 4.0, np)[direct] == 0
    legal[direct] = True
    values[5 * 441 + enemy] = 0  # An unobserved enemy is never a threat input.
    assert not public_capital_threat_gather_bonus(values, legal, 4.0, np).any()
    values[5 * 441 + enemy] = 1
    values[enemy] = np.log1p(5) / 8
    assert not public_capital_threat_gather_bonus(values, legal, 4.0, np).any()


def test_capital_gather_zero_strength_is_exact_noop():
    values, legal, _, _, _ = _position()
    assert np.array_equal(public_capital_threat_gather_bonus(values, legal, 0.0, np),
                          np.zeros(3529, np.float32))
    with pytest.raises(ValueError, match="matching legal masks"):
        public_capital_threat_gather_bonus(values, legal[:-1], 2.0, np)


def test_capital_gather_batches_active_and_inactive_public_views():
    values, legal, direct, _, enemy = _position()
    quiet = values.copy()
    quiet[5 * 441 + enemy] = 0
    bonus = public_capital_threat_gather_bonus(
        np.stack((values, quiet)), np.stack((legal, legal)), 2.0, np)
    assert bonus.shape == (2, 3529)
    assert bonus[0, direct] > 0
    assert not bonus[1].any()
