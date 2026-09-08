# Sentinel strategic agent

Sentinel v2 is the fork's deterministic strategic reference agent. It is separate
from the learned PPO policy: its wins demonstrate the scripted strategy's
strength, not a training gain. Its source is frozen before held-out evaluation.

Source: `generals/agents/sentinel_agent.py`  
SHA-256: `16f9942624d82c4fa14ec664b8e2289067b13ef1aa23744c7906a9a21e826834`

```python
from generals.agents.sentinel_agent import SentinelAgent

agent = SentinelAgent(
    build_castles=env.build_castles,
    deathtouch_turn=env.deathtouch_turn,
    max_turns=env.truncation,
)
action = agent.act(observation, key)
action, telemetry = agent.decision(observation, key)
```

Both methods support JAX `jit` and `vmap`. The key is accepted for the common
agent interface but does not affect decisions. Only the player's observation
enters the policy; there is no hidden-state access or episode memory.

## Decision algorithm

1. **Choose a shared objective.** Prefer a visible enemy general, then an
   affordable neutral castle, then visible enemy territory, then distant
   reachable fog or unowned land. Mountains and unidentified fog structures
   block routes. Expensive neutral castles also block scouting paths.
2. **Route and consolidate.** Convergent Bellman distance fields guide armies
   toward the objective, charging extra for visible defenders. Evaluate every
   legal adjacent move with both full and half armies. Favor advancing larger
   stacks, friendly pooling, captures, and profitable structures. Ordinary
   attacks that cannot capture their destination are excluded.
3. **Protect the general.** Retain a dynamic reserve based on nearby visible
   enemies, discounting threats that must travel and fight before reaching home.
   Penalize moves that leave the general vulnerable to the strongest remaining
   adjacent attacker. Credit third-tile interception and urgent reinforcement.
   A safe one-army founding expansion can break an adjacent-general standoff.
4. **Invest and finish.** Castle construction uses the modifier's exact observable
   price, requires a local defensive buffer, and reserves repayment time plus a
   profit window before the turn limit. A winning general capture takes priority.
   When configured deathtouch is active, reaching the enemy general wins regardless
   of its army; defense favors removing the attacker's source from a third tile.

The decision telemetry exposes score, candidate count, general army/reserve,
visible adjacent threat, selected objective category, construction, and deathtouch.
Replay tooling records the full game state separately for diagnosis.

## Tests and development evidence

Twelve focused CPU tests cover winning capture, rejecting a losing attack,
deathtouch activation, ordinary/deathtouch third-tile interception, reinforcement,
castle capture, exact construction prices, construction rules and time horizon,
wall routing and vectorization, safe founding, and dispatch through a contested
corridor. They verify tactical contracts; the arena measures playing strength.

V1 replays exposed two failures. In an adjacent-general standoff, Sentinel passed
all 500 turns while both sides maintained equal production. A safe founding
sortie gave it additional land income; v2 won the replay at turn 152. In a
mountain corridor with generals two steps apart, v1 reserved the entire enemy
stack and passed all 308 turns before losing. Discounting distant threats allowed
an outgoing stack to contest the corridor; v2 won at turn 154. The immediate
adjacent-attack check was retained.

The same development suite used 32 base boards with both player IDs and general
label swaps, yielding 128 games per matchup. Entries below are wins/losses/draws;
these paired games are not 128 independent maps.

| Suite / opponent | V1 | V2 |
| --- | ---: | ---: |
| Open 4×4 / Hunter | 102 / 0 / 26 | 128 / 0 / 0 |
| Terrain 4×4 / Hunter | 84 / 4 / 40 | 128 / 0 / 0 |
| Terrain 4×4 / Harvester | 82 / 8 / 38 | 124 / 4 / 0 |
| Terrain 4×4 / Expander | 114 / 14 / 0 | 112 / 16 / 0 |
| Classic 8×8 / Expander | 118 / 8 / 2 | 116 / 10 / 2 |
| Classic 8×8 / Hunter | 122 / 6 / 0 | 122 / 6 / 0 |
| Classic 8×8 / Harvester | 120 / 8 / 0 | 120 / 8 / 0 |

The change removes the diagnosed stalemates but is not uniformly better: it loses
two additional paired games against Expander in each of two suites. An initial
competition-format check on eight base maps produced 32/0/0 against Expander and
30/2/0 against Hunter, with 212 and 76 successful Sentinel builds respectively.
No physically invalid Sentinel move attempts were recorded in these development
runs. Details and map-cluster confidence intervals are in the run summaries:
`.cache/runs/sentinel-dev-v1/`, `.cache/runs/sentinel-dev-v2/`, and
`.cache/runs/sentinel-competition-dev-v2/`.

## Limits and next experiments

These are development results against local scripted opponents. Held-out
classic8/classic12/competition evaluation follows the v2 freeze; no holdout
observations were used here. They do not establish universal dominance or a
public competition ranking.

The policy uses heuristic threat estimates rather than exhaustive opponent-move
search. Scouting forgets previously visible information, objective changes can
cause wasted movement, and small expansions can fragment armies. Castle building
is opportunistic rather than a planned construction campaign. Seeing a defended
general can suppress otherwise valuable economic objectives. Remaining losses,
including the Expander regression, need separately versioned v3 experiments with
paired measurements; they should not be patched into the frozen candidate.
