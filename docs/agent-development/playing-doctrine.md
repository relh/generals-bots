# How Sentinel should play

This is the owner's strategic direction for our bot, applied to the rules of
each game. It guides policy design,
instrumentation, training experiments and review. The objective is to beat every
available distinct opponent; **the current bot has not achieved that objective**.
These are intended behaviors, not claims that Sentinel already implements them
or that they have passed competitive evaluation.

## Castles and game modes

In classic play, castles are **captured from neutral territory or other players**.
Treat them as defended capture objectives and sources of future production, with
an army cost and a need to hold the surrounding territory. Do not plan to build
castles in classic 1v1 or FFA.

Our competition experiments use this repository's separate
`mode="competition"` variant, which removes neutral castles and enables a BUILD
action, plus deathtouch at turn 800. The owner has confirmed that construction
is in scope. In this mode, compare construction's cost and future income with
expansion and military needs; castles owned by opponents can still be captured.
Apply the opening and contact guidance below as appropriate in either mode.
Experiments must name their rules explicitly: competition 1v1 results do not
establish classic or FFA strength, and accepting construction does not make BUILD
a legal classic action.

## FFA opening: grow locally before the land tick

Capture as many affordable local tiles as possible before the first whole-land
production tick. Leave a broad pool of owned one-army tiles ready to become two.
Keep the footprint compact and close to home so growth requires fewer transfers
and is less likely to expose us to other players. Protect the capital and avoid
spending the opening army on an expensive neutral castle at the expense of this
initial tile income.

The relevant event is the whole-land tick, distinct from the capital's frequent
production. In this simulator, ordinary owned tiles grow every 50 steps and
generals/castles every two; derive the next event from the actual rules and turn
phase. Measure tiles actually held across the tick and their resulting income,
not merely how many expansion moves were proposed.

## FFA after the local 1-to-2 tick: choose by contact

1. **No contact and no evidence we have been seen:** fan out again into nearby
   unvisited territory. Turn the new local armies into more tiles eligible for
   the next land tick. Keep expansion compact and avoid unnecessary exposure.
2. **One opponent encountered:** collect our distributed army into a deathball
   and plan a direct attack on that opponent's capital. Finding the capital is
   part of the attack: use observed territory and scouting to choose a plausible
   route, update it as information arrives, and remember a capital once revealed.
   Collect enough force to sustain the advance; repeated gathering without
   deployment is not progress. Preserve a viable defense at home.
3. **Two or more opponents encountered:** treat this as a dangerous position.
   Avoid pushing borders near either opponent and expand into unvisited areas
   away from contact. Read each player's public scoreboard to identify the weaker
   opponent and assess the combined threat. If we are not stronger than both
   relevant opponents together, committing to one risks letting the other clean
   us up. Favor growth and preservation over a premature attack. Even a favorable
   aggregate army comparison must account for how much force can arrive, travel
   time, capital safety and the army left after an attack.

“Encountered” means a distinct hostile player, not a count of visible tiles or
disconnected blobs. Remember contacts after they disappear into fog; one frame
without a visible enemy does not restore the no-contact opening. Being unseen
is an inference from public evidence, never access to another player's fogged
map. Record uncertainty and evidence of exposure instead of claiming certainty.

## 1v1: establish the attack line and conceal the capital

Quickly establish a useful line toward the opponent while obscuring the location
of our capital. Use observed terrain and contact to refine the approach; the
opponent's hidden capital is not a known destination. Avoid an obvious trail of
movements that unnecessarily reveals home. Combine local income with gathering
and a purposeful advance, then exploit a discovered capital when feasible.

FFA's desire to avoid creating two fronts must not turn 1v1 into passive local
growth indefinitely. A line is useful when it supports scouting, reinforcement
and an attack, not simply because a route has been drawn.

## What the implementation must expose and measure

- Public player identities on visible tiles and individual public scoreboard
  totals, with fog preserved. The current simulator observation aggregates all
  enemies, and the live adapter assumes 1v1; these interfaces cannot yet support
  the complete FFA contact and weaker-opponent decisions above.
- The strategic phase, next land tick, local one-army tiles held through that
  tick, newly captured tiles, travel cost and observed contact history.
- The selected frontier or capital-search route, collection cost, deployable
  deathball strength and actual route/capture progress. An owned transfer alone
  is not completed economic service.
- Separate opponent scores, combined nearby threat, home defense and the force
  expected to remain after attacking. Label estimates and distinguish total
  armies on the scoreboard from armies available at the front.

Keep the simulator's hidden state out of policy decisions. Diagnostic replays
may inspect it separately, but cannot supply opponent locations, identity under
fog or private visibility to the bot.

## How improvements earn promotion

Implement and test the phase decisions explicitly, then evaluate complete games
against strong controls and distinct opponents under the named rules. Report
classic 1v1, classic FFA and the construction/deathtouch competition variant
separately. Preserve successful controls and compare paired placements; two
mirrors of a position are not independent maps.

For learned policies, shaped rewards should reflect useful state progress and
remain subordinate to winning. Do not reward repeated transfers, collection
loops, oscillating phase changes or a larger deathball that never advances.
Test reward changes against an outcome-only control before crediting them with
strength. The [competitive objective](competitive-goal.md) defines the evidence
required for success; the [development runbook](README.md) records actual results.
