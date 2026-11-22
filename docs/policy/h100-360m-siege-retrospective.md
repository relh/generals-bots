# Where the H100 checkpoint lost ground against Classic Siege

The 360,710,144-step checkpoint gained 2.34 percentage points in mean match score
across the full population and lost **3.26 percentage points** against `classic_siege_padded` on the same
906 cases. Seat 1 accounts for 22 of the 29.5 score points lost. This is a
descriptive result for these held-out maps. The other 3,190 cases gained
125.5 score points, giving a net gain of 96 across the full panel. Uncertainty
for Siege requires a separate interval.

| Learner seat | Games per arm | Baseline W / D / L | Candidate W / D / L | Win-rate change | Draw-rate change | Match-score change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 453 | 269 / 1 / 183 | 260 / 4 / 189 | −1.99 pp | +0.66 pp | −1.66 pp |
| 1 | 453 | 252 / 1 / 200 | 228 / 5 / 220 | −5.30 pp | +0.88 pp | −4.86 pp |
| Both | 906 | 521 / 2 / 383 | 488 / 9 / 409 | −3.64 pp | +0.77 pp | −3.26 pp |

**Map distribution.** The evaluation used 4,096 first episodes per arm, a
1,024-map pool, map seed 20261007 and action seed 20261008. The original
`summarize-funded-evaluation.py` required identical initial-state SHA-256,
learner seat and opponent label arrays, element for element, and equal seeds,
population build, opponent weights and action-selection settings before writing
the [published comparison](h100-360m-evaluation.json). Filtering that paired
panel to Siege therefore gives the same map frequencies in both arms. The
1,008 distinct states are a count for the full panel. Multiple games can share
a state. The Siege-only distinct count, an equal-weight-per-map estimate and a
Siege-only map-clustered interval require the underlying rows.

**Outcome attribution.** Wins fell by 33, losses rose by 26 and draws rose by
seven. Draw rate rose from 0.22% to 0.99%. Giving draws half credit returns
3.5 points of the 33-win deficit. Even without knowing the exact transitions,
at least 24 more games flipped from win to loss than from loss to win. To see
the bound, subtract net win-to-draw flips from the 33 missing wins. At most
nine baseline wins could end as candidate draws, and at most two baseline
draws could become candidate wins. The resulting net decisive-flip bounds are
24–35 overall, 5–10 for seat 0 and 19–25 for seat 1. These are bounds on
net flips. Marginal totals cannot identify gross flips or a tactical cause.

**Next experiment.** First recover the two outcome vectors and stratify the
paired transition table by seat and initial-state hash. If the seat-1 deficit
persists with equal map weights, inspect the first divergent public action on
a small, deterministically selected set of distinct seat-1 baseline-win /
candidate-loss maps, alongside retained-win controls. Use the same frozen
actors, Siege build and sampling keys. That isolates a decision to test before
considering a policy or opponent change.

## Reproducing the comparison

The seat rows above are the Siege entries printed by the original arms'
`evaluation.json` reports. Their panel totals (2,743 / 1,317 / 36 and
2,831 / 1,213 / 52 in W/L/D order) and wall times (237.339 and 251.670
seconds) match the checked-in baseline and candidate receipts. To reproduce
each rate difference in points, divide the candidate-minus-baseline count by
games per arm and multiply by 100. Use wins plus half of draws for match score.

The original paired export is `art-46jba`, 24,145,920 bytes, SHA-256
`69aecf5c2735e39ce5430f54fa0e4cb189729c3b548277e72967ba68d532049c`.
At retrospective time the preserved MBP extraction contained empty
`baseline-eval` and `candidate-eval` directories, and the configured provider
workspace could not resolve the export. The historical pairing check and
printed seat counts survived. This retrospective could not independently
recompute the row-level transition matrix or its map-clustered interval.

With the verified export extracted under `$PANEL`, run the historical
comparison tool from the repository root. This version accepts the recorded
zero-valued `log_gap_scale` sampler field. The current tool rejects that
retired field. The tool’s signed-score scale is twice the match-score scale above.

```sh
tmp=$(mktemp -d)
git show 2895e9c:integrations/analyze_spatial_population_pair.py > "$tmp/compare.py"
python "$tmp/compare.py" --baseline "$PANEL/baseline-eval" \
  --candidate "$PANEL/candidate-eval" --output "$tmp/pair.json"
jq '.by_opponent_and_seat.classic_siege_padded' "$tmp/pair.json"
```

After that equality check, reproduce the Siege-only paired breakdown from the
same arrays. Each transition table has baseline rows and candidate columns in
loss, draw, win order. The bootstrap resamples whole map hashes.

```sh
PYTHONPATH="$tmp" python - "$PANEL" <<'PYTHON'
import sys
from pathlib import Path
import numpy as np
from compare import cluster_interval, load_arm

root = Path(sys.argv[1])
base, b = load_arm(root / "baseline-eval")
candidate, c = load_arm(root / "candidate-eval")
assert base["action_selection"] == candidate["action_selection"]
index = list(base["by_opponent_and_seat"]).index("classic_siege_padded")
siege = b["opponent_labels"] == index
for seat in (0, 1, None):
    keep = siege if seat is None else siege & (b["initial_sides"] == seat)
    old, new = (arm["outcomes"][keep].astype(int) for arm in (b, c))
    transitions = np.zeros((3, 3), dtype=int)
    np.add.at(transitions, (old + 1, new + 1), 1)
    ids = b["initial_state_sha256"][keep]
    delta = (new - old) / 2
    _, group, counts = np.unique(ids, return_inverse=True, return_counts=True)
    equal_map = np.mean(np.bincount(group, weights=delta) / counts)
    maps, interval = cluster_interval(delta, ids, seed=35687, resamples=10000)
    print(seat, len(delta), maps, transitions.tolist(), delta.mean(), equal_map, interval)
PYTHON
```
