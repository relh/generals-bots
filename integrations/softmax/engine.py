"""The Coworld boundary around the engine's regular, capture-only rules."""

import jax
import jax.numpy as jnp
import numpy as np

from generals import GeneralsEnv
from generals.core import coworld_game
from generals.core.match import make_board
from integrations.classic_contract import CLASSIC_MAP_OPTIONS, verify_engine

from .protocol import PASS, VERSION

RULESET = "classic"


@jax.jit
def executed_moves(state, actions):
    """Classic move receipts use the pinned Coworld resolution order."""
    executed = jnp.zeros((2,), dtype=bool)
    for player in coworld_game._determine_move_order(state, actions):
        action = actions[player]
        r, c = action[1], action[2]
        before = state.armies[r, c]
        state = coworld_game.execute_action(state, player, action)
        executed = executed.at[player].set((action[0] == 0) & (state.armies[r, c] < before))
    return executed


class Match:
    def __init__(self, seed: int):
        verify_engine()
        self.env = GeneralsEnv(**CLASSIC_MAP_OPTIONS)
        self.state = make_board(self.env, seed)
        self._game = coworld_game
        self._executed_moves = executed_moves
        self.transition = jax.jit(lambda state, actions: coworld_game.step(state, actions, general_trade=False))
        self.last_move_executed = [None, None]
        # Compile before /healthz and before player deadlines begin.
        jax.block_until_ready(self.transition(self.state, jnp.array([PASS, PASS], dtype=jnp.int32)))
        jax.block_until_ready(self._executed_moves(self.state, jnp.array([PASS, PASS], dtype=jnp.int32)))
        for slot in range(2):
            jax.block_until_ready(self._game.get_observation(self.state, slot))
        self.height, self.width = self.state.armies.shape

    @property
    def turn(self):
        return int(self.state.time)

    def observation(self, slot: int) -> dict:
        obs = self._game.get_observation(self.state, slot)
        kinds = np.ones((self.height, self.width), dtype=np.int32)
        for name, value in (
            ("fog_cells", 0),
            ("structures_in_fog", 5),
            ("mountains", 2),
            ("castles", 3),
            ("generals", 4),
        ):
            kinds[np.asarray(getattr(obs, name), dtype=bool)] = value
        owners = np.zeros_like(kinds)
        owners[np.asarray(obs.owned_cells)] = 1
        owners[np.asarray(obs.opponent_cells)] = 2
        return {
            "type": "observation",
            "protocol_version": VERSION,
            "slot": slot,
            "turn": self.turn,
            "last_move_executed": self.last_move_executed[slot],
            "height": self.height,
            "width": self.width,
            "my_land": int(obs.owned_land_count),
            "my_army": int(obs.owned_army_count),
            "opp_land": int(obs.opponent_land_count),
            "opp_army": int(obs.opponent_army_count),
            "type_grid": kinds.tolist(),
            "owner_grid": owners.tolist(),
            "army_grid": np.asarray(obs.armies).tolist(),
        }

    def frame(self) -> dict:
        s = self.state
        kinds = np.ones((self.height, self.width), dtype=np.int32)
        for mask, value in ((s.mountains, 2), (s.castles, 3), (s.generals, 4)):
            kinds[np.asarray(mask)] = value
        owners = np.zeros_like(kinds)
        for slot in range(2):
            owners[np.asarray(s.ownership[slot])] = slot + 1
        info = self._game.get_info(s)
        return {
            "turn": self.turn,
            "type_grid": kinds.tolist(),
            "owner_grid": owners.tolist(),
            "army_grid": np.asarray(s.armies).tolist(),
            "army": np.asarray(info.army).tolist(),
            "land": np.asarray(info.land).tolist(),
        }

    def advance(self, actions: list[list[int]]) -> int:
        if any(action[0] not in (0, 1) for action in actions):
            raise ValueError("classic rules accept only moves and passes")
        batch = jnp.array(actions, dtype=jnp.int32)
        executed = self._executed_moves(self.state, batch)
        self.state, info = self.transition(self.state, batch)
        self.last_move_executed = [bool(executed[s]) if actions[s][0] == 0 else None for s in range(2)]
        return int(info.winner) if bool(info.is_done) else -1
