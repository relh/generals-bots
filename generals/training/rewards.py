"""Terminal outcome plus bounded potential shaping, using public observations."""
import jax.numpy as jnp


def potential(obs):
    """Economic advantage in [-1, 1]; no action bonuses or hidden-map access.

    Scoreboard army/land totals are public in the simulator observation contract.
    Own castles use diminishing returns: a sustainable economy, not build spam.
    """
    army = (obs.owned_army_count - obs.opponent_army_count) / (
        obs.owned_army_count + obs.opponent_army_count + 1.0)
    land = (obs.owned_land_count - obs.opponent_land_count) / (
        obs.owned_land_count + obs.opponent_land_count + 1.0)
    castles = jnp.sum(obs.castles & obs.owned_cells).astype(jnp.float32)
    economy = castles / (castles + 3.0)
    return 0.5 * army + 0.3 * land + 0.2 * economy


def shaped_reward(prior_obs, final_obs, winner, team, terminated, gamma=0.99, weight=0.2):
    """Return reward and separately logged components.

    `terminated` includes task-defined terminal draws and sets potential to zero.
    For artificial time limits only, pass False to retain final potential and
    bootstrap the final observation's value. The caller chooses the semantics.
    The shaping term telescopes under the SAME discount as the learning return.
    """
    outcome = jnp.where(terminated & (winner >= 0), jnp.where(winner == team, 1.0, -1.0), 0.0)
    prior_phi = potential(prior_obs)
    final_phi = jnp.where(terminated, 0.0, potential(final_obs))
    shaping = weight * (gamma * final_phi - prior_phi)
    return outcome + shaping, (outcome, shaping, prior_phi, final_phi)
