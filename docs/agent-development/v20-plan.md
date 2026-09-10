# V20: assemble offensive force from owned branches

This experimental candidate tests a capability absent from the current
single-path collector: gathering several owned branches into one rally under a
shared serial-action budget. The [post-V19 audit](v20-audit.md) ruled out another
broad coverage-preserving veto as a new explanation. It did not prove that
branching collection wins. Complete adaptive comparisons must decide that.

## Collection contract

V20 wraps exactly one default V10 call. Disabled operation returns the exact
V10 tuple and its native nineteen-field memory. Enabled memory contains that
parent plus nine int32 scalar fields for one branch plan. V2 remains default.

Use V8's existing visible objective and rally ranking. Eligible gather cells
are owned, nonhome cells without adjacent visible movable enemy force. Build a
deterministic shortest-owned tree toward the rally, breaking parent ties in
UP/DOWN/LEFT/RIGHT order. A dynamic program considers connected rooted subtrees
with exact budgets zero through six edges. Each used edge is one full-transfer
action leaving one army behind. Children execute before their parent's outgoing
transfer. An intermediate cell with zero or one army can participate only if
its children first supply enough to make that outgoing transfer executable.
No donor or edge receives duplicate credit.

The planner maximizes delivered current army separately at each exact cost;
it does not optimize arbitrary trees, objectives or adaptive enemy replies.
The wrapper chooses the cheapest sufficient result, requiring the initial
assembled force to be at least 1.5 times its largest contributing garrison.
A new branch plan is admitted only when no raw-force-sufficient single owned
path of at most six steps exists to the same rally. The single-path comparison
does not apply the 1.5 multiplier: failure of that heuristic alone is not
branching necessity. Path estimates may conservatively reject an admission
when a path's proposed moves are not executable.

Sufficiency retains V8's target army, largest adjacent visible counterforce,
deployment departure costs and projected deathtouch convention. The rally's
owned deployment path is initially two or three steps. These estimates do not
predict enemy growth, merges, alternate routes or unseen force.

## Replanning, priorities and execution

Recompute the remaining tree from actual current owned garrisons after every
issued move. Keep the objective, rally and original expiry. Check the last
recipient remains owned and has at least the expected post-transfer army;
never add an old donor estimate to current armies. Every actual gather action
reduces the remaining budget; replanning can choose a cheaper sufficient
subtree. Deploy immediately once the observed rally army suffices. Deployment
uses only the actual packet's army and a nonincreasing owned-path distance.
The remaining work must fit the fixed expiry. A completed target capture
clears the branch plan.

Preserve parent BUILD, actual offensive plans, pursuit, mobilization, incoming
or returned defenders, guards, adjacent threats, positive visible home deficit,
insufficient reserve and winning general captures. A newly feasible parent
collection or direct attack can take over after a branch move: return its
actual action and memory and release the branch plan. This is an intentional
handoff, not a promise to finish every admitted subtree. Reset and observation
gaps discard branch transport; stale incoming defenders must not block a new
game after the parent has reset.

A branch action must pass the existing current public home-route check. It is
not a future safety certificate. On an issued branch action, clear unissued
parent offensive transport while retaining real defender memory and public
stationary-general knowledge, even if the physical action agrees with the
parent. Inherited issued-action events require `branch_parent_action_issued`
and the existing outer selection masks. Branch availability, issued collection,
deployment, takeover/release and changed physical output are separate events.

## Qualification and complete-game gate

The planner's independent exhaustive tiny-tree oracle checks force, action
cost, first executable postorder move, deterministic ties, zero-army transit,
blocked terrain and typed JIT/batched output. Wrapper checks must cover actual
branch admission, residual execution, early deployment, fixed expiry, recipient
loss, parent takeover, reset and original archived parent tuples. Selected
historical fixtures are preservation checks, not evidence of whole-game wins.

Profile synchronized full calls against V10 and disabled V20 in the pinned
Python/JAX runtime, with representative active gather/deploy and blocked cases,
including original game-sized observations. Report cold compilation separately.
Require at least 50% parent throughput in each measured context before using
strategy outcomes as evidence; this does not qualify standalone startup.

After source, fixtures, checks and the game harness are frozen, compare V10,
V10-v6, V19 and V20 from turn zero against the unchanged original Amin main8
iter160, Juraj V3.5 and my_bot9 on the consumed seed83000 physical placements
and their original mirrored keys. This is 24 complete games. Verify all eighteen
control histories before releasing the six candidate games, retain all attempts,
raw/applied joint actions, public observations, native memory, telemetry,
opponent invalid actions, enforced deadlines, terminal exits and cleanup.
No source/helper mutation during games or healthy restart is allowed.

V20 must win all six candidate games to advance past this small development
screen and preserve the stronger controls. Full failure audits follow any
rejection. Even six wins do not establish dominance: distinct-opponent breadth,
fresh paired confidence-interval lower bounds above 50%, runtime qualification,
and the remaining classic/FFA doctrine requirements remain outstanding.
Reserved seed143000 is untouched during development.
