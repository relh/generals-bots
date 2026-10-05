# Coworld Classic engine

`coworld_game.py` is an unmodified copy of `generals/core/game.py` from the
official `strakam/generals-bots` `softmax` branch at commit
`0fcb5a00226387670624d2f326f6d5ad61914584`.
Its SHA-256 is
`f39e448a6b2822869d75cb07cce4cb43d589c4112fef04007ade951809d4a318`.

`GeneralsEnv(coworld_classic_rules=True)` uses this engine with
`general_trade=False` and the existing dynamic map pool. It selects the
published capture-only Classic rules and rejects conflicting modifiers.
The Classic Puffer wrapper selects this flag automatically.

The generic engine remains available for its existing rulesets and replay
experiments. Its move order differs from the hosted Classic game.

Validation includes targeted move-order and combat tests, actual staged
Puffer wrapper checks, and exact reproduction of 31,919 transitions from
64 completed hosted Classic matches. Replay states were used only for
postgame validation, never as policy inputs or training labels.
