"""Construction tactics and frozen-v2 parity, separate from strength evaluation."""

import base64
import io
import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from generals.agents.sentinel_agent import SentinelAgent
from generals.agents.sentinel_v4_agent import SentinelV4Agent, _build_threat
from generals.core import game
from generals.core.observation import Observation
from tests.test_sentinel_agent import board

KEY = jax.random.PRNGKey(0)


def construction_board(*, alternate=False, mountain=False, enemy_army=19):
    owned = [(0, 0, 20, True), (5, 5, 40, False)]
    if alternate:
        owned.append((8, 1, 40, False))
    return board(
        owned,
        [(5, 7, enemy_army, False)],
        mountains=[(5, 6)] if mountain else (),
        shape=(10, 10),
        time=283,
    )


def test_two_step_build_threat_uses_campaign_fallback():
    obs = construction_board()
    old = SentinelAgent(build_castles=True).act(obs, KEY)
    assert tuple(np.asarray(old)) == (2, 5, 5, 0, 0)
    action, telemetry = SentinelV4Agent(build_castles=True).decision(obs, KEY)
    np.testing.assert_array_equal(action, SentinelAgent(build_castles=False).act(obs, KEY))
    assert action[0] == 0  # A useful campaign move remains; no build-veto pass.
    assert telemetry["build_rejected_threat_count"] == 1
    assert telemetry["build_candidate_count"] == 0


def test_selects_safe_alternate_build_instead_of_rejecting_every_site():
    obs = construction_board(alternate=True)
    action, telemetry = SentinelV4Agent(build_castles=True).decision(obs, KEY)
    np.testing.assert_array_equal(action, [2, 8, 1, 0, 0])
    assert telemetry["build_rejected_threat_count"] == 1
    assert telemetry["build_candidate_count"] == 1
    assert telemetry["selected_build_threat"] == 0
    assert telemetry["selected_build_remaining"] == 5


@pytest.mark.parametrize("fog_structure", [False, True])
def test_known_obstacle_blocks_two_step_path(fog_structure):
    obs = construction_board(mountain=True)
    if fog_structure:
        obs = obs._replace(mountains=jnp.zeros_like(obs.mountains), structures_in_fog=obs.mountains)
    assert _build_threat(obs, 2)[5, 5] == 0
    np.testing.assert_array_equal(SentinelV4Agent(build_castles=True).act(obs, KEY), [2, 5, 5, 0, 0])


def test_weak_visible_army_and_clear_observation_do_not_create_stale_threats():
    agent = SentinelV4Agent(build_castles=True)
    unsafe = construction_board()
    assert agent.act(unsafe, KEY)[0] != 2
    clear = unsafe._replace(opponent_cells=jnp.zeros_like(unsafe.opponent_cells))
    assert agent.act(clear, KEY)[0] == 2
    safe = construction_board(enemy_army=1)
    assert agent.act(safe, KEY)[0] == 2
    # A larger garrison can safely cover the same two-step attack envelope.
    covered = unsafe._replace(armies=unsafe.armies.at[5, 5].set(58))
    assert agent.act(covered, KEY)[0] == 2


@pytest.mark.parametrize("time", [0, 800, 1150])
def test_build_rules_payback_and_winning_capture_priorities(time):
    obs = construction_board()._replace(timestep=jnp.int32(time))
    np.testing.assert_array_equal(SentinelV4Agent().act(obs, KEY), SentinelAgent().act(obs, KEY))
    late = construction_board(alternate=True)._replace(timestep=jnp.int32(1150))
    assert SentinelV4Agent(build_castles=True).act(late, KEY)[0] != 2
    win = board([(0, 0, 20, True), (2, 2, 2, False), (8, 1, 80, False)], [(2, 3, 100, True)], shape=(10, 10), time=800)
    np.testing.assert_array_equal(
        SentinelV4Agent(build_castles=True, deathtouch_turn=800).act(win, KEY), [0, 2, 2, 3, 0]
    )


def test_horizon_one_matches_v2_over_complete_engine_observations_and_vmaps():
    size, shape = 64, (6, 6)
    grid = jnp.zeros(shape, jnp.int32).at[0, 0].set(1).at[5, 5].set(2).at[2, 3].set(25)
    states = jax.vmap(game.create_initial_state)(jnp.broadcast_to(grid, (size,) + shape))
    rng = np.random.default_rng(93407)
    owners = rng.integers(0, 3, (size,) + shape)
    owners[:, 0, 0], owners[:, 5, 5] = 0, 1
    states = states._replace(
        ownership=jnp.array(np.stack((owners == 0, owners == 1), axis=1)),
        ownership_neutral=jnp.array(owners == 2),
        armies=jnp.array(rng.integers(1, 90, (size,) + shape, dtype=np.int32)),
        time=jnp.array(rng.choice([0, 50, 283, 799, 800, 1150], size), jnp.int32),
    )
    observations = jax.vmap(lambda state: game.get_observation(state, 0))(states)
    keys = jax.random.split(KEY, size)
    base = SentinelAgent(build_castles=True, deathtouch_turn=800)
    ablation = SentinelV4Agent(build_castles=True, deathtouch_turn=800, build_threat_horizon=1)
    reference = jax.jit(jax.vmap(base.decision))(observations, keys)
    actions, telemetry = jax.jit(jax.vmap(ablation.decision))(observations, keys)
    np.testing.assert_array_equal(actions, reference[0])
    for name, value in reference[1].items():
        np.testing.assert_array_equal(telemetry[name], value)
    candidate = SentinelV4Agent(build_castles=True, deathtouch_turn=800)
    batched = jax.jit(jax.vmap(candidate.act))(observations, keys)
    for index in (0, 1, 17, 31, 63):
        obs = jax.tree.map(lambda value: value[index], observations)
        np.testing.assert_array_equal(batched[index], candidate.act(obs, keys[index]))


def test_rejects_unsupported_horizon():
    with pytest.raises(ValueError, match="1 or 2"):
        SentinelV4Agent(build_threat_horizon=3)


def test_recorded_development_turn_283_rejects_unsafe_investment():
    # Exact public observation, t283, .cache/runs/sentinel-v3/external-diagnostic/
    # r2-game0.npz. Embedded numeric NPZ keeps this regression independent of
    # ignored local run artifacts. This consumed development case is not a holdout.
    packed = (
        "c-rk+O=uHA6rQ(jMVgkTZEDrV8bL@4X`#}CR#C7R4^@c=4_*vwI&A}+P1xNi#R>`)^(Y=a=s`ie6a@vrn;yN02QPZ_Ts*gm&NMsi"
        "{H2ZCKsTWy>t?6(eLHXGoA=(#Zf2^bbp%z)5!AEt_0V6xw4)d@okAYFqjqT?P!!GWskCF~$Cs{L%v=R@1KmnaV>jm{Clbk1`SD~b"
        "k(?<yo@3hCqBD(YecH5KOzPb^vxLd@(6O;pV(j>EDsejzu(YElE%7?>KnQsZa<p*2gYxiP`~lptRWUdoL+a1}^iRI8A>G3WU@J&C"
        "Bj8v-6&oSovyzV?4iDSZMz7SjdK3fgmPRibAx4cK{dj`Ks16dNX0eSO5?xr7TJfdB&e|xoaD6P?Q3`g4z0SIM6lnvdn+VZO&U8Hs"
        "H<@f2yFdtKb1#y9#$?k;WK$@XZO_cx%`Y6NaeSbbp;KZyfbduF|A$&`DK!IaUtx!Z^XA3!H71;HA{-l+$=F-j9JU%eSOEl^xB$8_"
        "sZzVCcBD4IlPP$$#{Uffy7oKdi$w8U9*n<d;^`vdDc-d4bmL1Wn3DmmXh5wIs{IQ9q||^2l>AAhnL<@*7y>u5g>)w77bcl>dZ=_t"
        "rJ{{(uNl?T*x3$*A_o%Hv-;rMYd+nXDb9vmIF)${8?%->`#+I%U3v0+OH}a`Y~J&c0hosX)dR^njK+~D&PEw`jWg56SWuFn4Y$N1"
        "+Isi-C&8L>b=*_q-5b|fs^n>XgeaUe{<6%3(?^8kdQLg#l^yJ6^LCcVD69ekUEToICnSjDApoITHVoS27J;TwFt?#lAtgX6-2yqE"
        "=DQC<@&)qK;hq|Rv=V?S%Y$`VA0Y~;)cNZj6HshD<*`hg<UMrF#q_i2BiMOcZy~m$9A7^ZiX~AuIjLG=I)(XgTk=G)<m{i(yUdb)"
        "K9>mpu$)nhIsfzAXRR^eADO3$G4Z7jOZ+aVmoH$~!=<oVFaWXz_kZ`VG7F;gU87~?n{$n`Tk4&qzU`YC=93LyJK^_AHJcKCpHwjg"
        "jFA@c`vLlO06qB0a%XM+ZhTE3<LEmbI>+hJA>f`9CjJeNjHJ8z3>guS-}fSZk6#ATeO(6Z6o9>iN7&_+p>$`DLAwR$1;V84^T}|!"
        "x5MCF0{kJlly-V#FkRg<aE}1~M7pA`^<_L=-|5XZVn|%=$xxc%GiaXxJ-H9*^L?3u=6H*~hGPQ7!hWR9_+$o}Tj?3n2aq<?k{M`D"
        "!x;JnhUh_5EyIyXXfB~A`N5NH-V++b8HuR+R|80h=rWSVPBs)#0cpjN5=mt=bvf+5;y}I}M0#IZJB51~>3vvcs%`DNp>Fao)r!#R"
        "1iktPiD2~^"
    )
    with np.load(io.BytesIO(zlib.decompress(base64.b85decode(packed))), allow_pickle=False) as arrays:
        obs = Observation(**{name: jnp.asarray(arrays[name]) for name in Observation._fields})
    assert obs.armies[12, 9] == 40 and obs.armies[13, 10] == 19
    assert _build_threat(obs, 1)[12, 9] == 0 and _build_threat(obs, 2)[12, 9] == 18
    kwargs = dict(build_castles=True, deathtouch_turn=800, max_turns=1200)
    old = SentinelAgent(**kwargs).act(obs, KEY)
    np.testing.assert_array_equal(old, [2, 12, 9, 0, 0])
    np.testing.assert_array_equal(SentinelV4Agent(**kwargs, build_threat_horizon=1).act(obs, KEY), old)
    action, telemetry = SentinelV4Agent(**kwargs).decision(obs, KEY)
    assert action[0] != 2 and telemetry["build_rejected_threat_count"] == 1
    np.testing.assert_array_equal(action, SentinelAgent(deathtouch_turn=800).act(obs, KEY))
