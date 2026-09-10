# Competitive objective

The [playing doctrine](playing-doctrine.md) is the owner's strategic direction:
castle acquisition appropriate to the rules, compact growth around land ticks,
contact-dependent FFA play and a concealed 1v1 approach. Construction remains in
scope for competition mode. Evaluate modes explicitly; construction/deathtouch
1v1 results do not establish classic or FFA performance.

The goal is to develop our own agent that beats every available distinct opponent,
not merely to complete an audit or training run. This goal is **not achieved**.
The v3 through v15 experiments have not established competitive superiority.
V2 remains the default control. The [v9 remembered-general experiment](v9-development.md)
improves Amin, Hunter and original my_bot9 over V8, but still fails preservation
against stronger V6 controls and scores only 18.75% against Juraj V3.5.
A favorable development score against some opponents does not establish
superiority over every opponent or justify promotion.

The [v10 home-mobilization experiment](v10-development.md) turns both selected
my_bot9 losses into wins in complete games from turn zero. These are mirrored
cases from one consumed position, not independent fresh strength evidence.
Its [completed 576-game comparison](v10-development-gate.md) passes deployment
and runtime checks and improves several V9 development results, but still
fails stronger-control preservation. Default V10 scores 25% against Juraj V3.5
(two wins, ten losses, four draws), while V6 scores 37.5%. V10 over V6 matches
V6's aggregate scores without improving them. Neither candidate establishes
the required winning advantage against every opponent; both remain experimental.

The [V11 selected comparison](v11-development.md) also fails: immediate enemy
capture arbitration preserves the selected my_bot9 wins but regresses both Amin
wins and still loses to Juraj V3.5. A local action-cost improvement or longer
survival does not meet the competitive objective. V11 is not promoted.

The [V12 route-guard comparison](v12-development.md) confirms that preserving
static defensive deficits is also insufficient: its replacements pass an
independent certificate check but regress Amin and do not change the Juraj losses.
It remains experimental; the full goal remains active.

The [V13 collection-only comparison](v13-development.md) preserves ready direct
deployments but still turns the selected Amin wins into losses and fails to beat
Juraj. Its earlier my_bot9 wins do not compensate for those gaps.

The [V14 preflight and eighteen-game comparison](v14-development.md) first rule
out pursuit-only composition as a preservation candidate: V6 loses the selected
Amin position without ever seeing an enemy general. V14 instead tests service
between collections and campaign moves, with both V10 and V10-v6 controls. Its
ledger is correct, but it still loses Amin and Juraj. Next work must distinguish
actual frontier/economic progress from generic serviced transfers and preserve
useful concentration. Explicit expansion selection and productive packet
completion remain hypotheses, not established improvements.

The [V15 complete opening screen](v15-development.md) rejects both precontact
expansion modes: each loses all six selected games despite retained tile income.
Additional territory and aggregate armies do not ensure that force reaches home
or a useful attack route. Next work must follow complete packet progress,
capital-search/disclosure and defensive releases during temporary threat fog.
The user's full FFA/1v1 doctrine remains only partially implemented and unproven.

The [V16 read-only audit](v16-audit.md) rejects route completion as the next broad
repair because it cannot activate in the audited losing controls. It also rules
out an affordable-build veto as the explanation for the early Juraj loss. These
findings motivate investigating connected force and productive campaign or
investment objectives; they select no policy and establish no winning advantage.

The subsequent [V16 route experiment](v16-results.md) completes eighteen games
and fails preservation: its six games yield two my_bot9 wins and four Amin/Juraj
losses. The intended routes gather and capture correctly, yet useful movement
after capture and timely local defense remain unresolved. Total army, completed
routes and delayed capital exposure do not establish strength. V16 is not
promoted; the default and stronger controls remain intact.

The [V17 capture-chain screen](v17-results.md) also fails preservation: all 24
games complete, but V17 loses both Amin games at 235 and both Juraj games at 790,
while winning my_bot9 at 961. All 214 chain captures are confirmed; they do not
repair earlier force allocation or defensive releases during threat fog.
The original my_bot9 makes 590 invalid builds in the V17 games; those failures
remain part of the results. Host qualification and corrected integrity checks
pass, without promotion or a claim of broad strength.

The [V18 rear-expansion screen](v18-results.md) completes all 24 games and also
fails preservation: Amin losses at 546, Juraj losses at 447 and my_bot9 wins
at 888. All 522 rear captures execute and 480 remain held through their next
land tick, but greater economy does not become timely concentrated defense.
The unchanged original my_bot9 makes 34 invalid builds in the V18 games.
Correctness, runtime and complete integrity checks pass; V18 is not promoted.

Success must name the tested opponent set and establish a winning advantage
against each opponent on maps reserved before the final candidate freeze. Use
paired seats/general labels, complete episodes, and map-cluster uncertainty;
require a score confidence-interval lower bound above 50% against each distinct
opponent. Preserve the existing local strength gates and validate deployment
under the actual runtime limits. Report untested or unavailable policies instead
of extending a measured result to all public entrants.

The next strategy work must address the opening economy and field-force gap
while preserving successful Amin plans and the recovered my_bot9 wins. The
completed V10 diagnosis retains a Juraj V3.5 case where V9 and V10 lose at turn
422 with identical complete actions while V6 wins. Its recovered Amin and Juraj
draws also end economically behind without ever seeing the enemy general.
General discovery and attack remain separate hypotheses from late home defense.

The [earlier failure diagnosis](v9-diagnosis.md) still identifies limitations in
defense under enemy merges and our own donation moves. A force certificate can
become stale when a source is vacated; rejecting that move alone does not establish
a safe alternative. Preserve stronger controls and useful successful plans while
testing complete alternative trajectories. Greater stack size, fewer reversals or
more captured tiles alone are not success.

Previously consumed seeds, including 83000, 93000 and 113000, may support
explicitly labeled development experiments. They are never fresh holdouts again.
Reserve 143000 for a later final competition comparison, after candidate and
opponent selection; do not generate or inspect it during development.

Separate synchronous strategy experiments from deadline-enforced deployment
tests. The former can distinguish policy outcomes from shared-host late-reply
penalties; they do not qualify a submission. Preserve the untouched opponent
implementation, its public wire observations and per-game memory, and verify
adapter action parity before using this additional evaluation path.

Deliver validated owned work to our fork's `origin/main`, preserving unrelated
remote changes, and leave the primary workspace on `main` after integration.
An inconclusive or losing experiment keeps the competitive goal active.
