# Modes and Controls

The app builds a radio from **Now Playing**, or a complete result queue through **Find Music**.
Relevance comes from normalized 768-dimensional CLaMP3 embeddings. Metadata supplies recording
identity, artist limits, dates, and display, not musical scoring.

## Shared Settings

| Control | Effect |
| --- | --- |
| Queue size | Request 10-100 tracks, in steps of 10, for radio and Find Music |
| Added to index | All dates, or exactly the last N rolling 24-hour periods, using the embedding DB's original `tracks.created_at`. Server additions use the phone merge date; reindexing retains the original date. |
| Artist-credit limits | Enable the per-artist cap and spacing constraints for radio |
| Max per artist | Limit occurrences of the same complete artist credit in a new radio queue |
| Min artist spacing | Require this many intervening tracks between matching artist credits in radio |

Unknown added dates qualify only under All dates. Artist matching ignores case and surrounding
whitespace; blank credits are unlimited. Radio enforces artist constraints during selection,
so they can change subsequent choices. Find Music queues its displayed results instead of applying
the radio artist limits afterward.

Queues may be shorter than requested when constraints, available tracks, graph reachability,
or DPP's remaining rank prevent filling them.

### Reading the Rank

The result's nearest/match number is an **ordinal rank**, not a physical distance or confidence:
`#100` means 99 reference identities rank ahead of it.

- **Radio:** cosine rank from the original seed across the full active indexed library, excluding
  the seed. Selection mode, drift, date window, and memory do not change this reference.
- **One positive text description:** full-active-library text-to-audio cosine rank.
- **Composed Find Music:** full-active-library weighted All-of rank for the same ingredients,
  directions, and weights, excluding recording anchors themselves, including when Refine or
  Varied selects the queue.

Changing ingredients or weights changes the reference. Changing the library can also change ranks.
Expanded details show the selection domain and, for Refine, the primary and secondary ranks.

## Radio Modes

### Closest to Seed

**Choose this for the nearest matches.** Sort the eligible library by seed cosine and select in
that order, subject to artist constraints. There is no relevance knob because this mode has one
fixed objective.

### Uniform Shuffle

**Choose this for discovery without a similarity preference.** Give every eligible recording
identity an equal-opportunity deterministic permutation. **New order** changes the permutation;
saved sessions keep the exact order they produced.

### MMR

**Choose this for nearby music with less repetition.** At each step, choose the candidate maximizing

```text
lambda * cosine(candidate, query)
  - (1 - lambda) * max cosine(candidate, previously selected track)
```

The relevance/variety slider is `lambda`: higher values emphasize closeness; at 1, selection
follows nearest order. Lower values penalize resemblance more strongly.
Before any track is selected or remembered, the overlap term is zero.

**Selection pool** restricts which candidates MMR can consider. Without drift it is a fixed
seed-nearest subset; with drift it is retrieved around the evolving query after each pick.
For eligible count `N`, pool fraction `f`, and requested length `K`, its size is
`min(N, max(floor(N*f), 100, K))`. The default is 2%; All eligible uses the full domain.
A larger pool gives low-relevance MMR more room to explore, but need not change a high-relevance queue.

#### Drift

Only MMR supports drift. It changes the query for later picks; the displayed seed rank still
refers to the original seed.

| Direction | Controls and behavior |
| --- | --- |
| Seed + last pick | Normalize a weighted blend of the original seed and last pick. Starting seed pull sets the seed share; Seed-pull fade changes it over the queue. |
| Rolling direction | Blend each pick into the previous direction and normalize. Higher Prior-direction memory changes direction more slowly. |

Seed-pull fade schedules:

- **Hold:** keep the starting share.
- **Linear:** halve it at the chosen point; reach zero after twice that many picks.
- **Exponential:** halve it every chosen half-life, independently of queue length.
- **Step:** hold until the drop point, then use one fifth of the starting share.

### DPP

**Choose this for a relevant set covering different embedding directions.** Greedily add the track
with greatest marginal determinant gain, measuring redundancy against the selected set as a whole.
MMR instead penalizes resemblance to the single most similar previous track.

The kernel is `L[i,j] = q[i] * q[j] * cosine(i,j)`, where
`q[i] = (max(0, seed cosine[i]) / maximum positive candidate cosine)^p`.
**Seed pull** is the exponent `p`: stronger values favor seed relevance over set diversity.
At zero, all qualities are equal and the first tie follows the stable nearest ordering.

**Selection pool:**

- **All eligible** (default): select from the entire eligible non-seed domain. The implementation
  grows a working prefix until bounds prove the same choices as full-domain greedy selection.
- **Nearest subset:** intentionally restrict that domain using **Subset size**, with the same
  count rule as MMR above.

The full-domain certificate proves agreement with the greedy algorithm, not globally optimal
DPP MAP selection. DPP does not drift.

### Remember Previous

Enable this for DPP or MMR when repeating a radio from the same seed.
All successfully enqueued recordings from that seed are excluded from new membership and
participate in diversity scoring:

- **MMR:** the maximum-overlap term also includes remembered recordings.
- **DPP:** project candidates away from the span of remembered vectors, then run greedy DPP.
  If `U` is an orthonormal basis of that span, the kernel becomes
  `L[i,j] = q[i] * q[j] * x[i]^T * (I - U*U^T) * x[j]`.

Memory is uncapped and persists independently of session history. Successful ordinary radios
are recorded even while the checkbox is off; previews and failed enqueues are not.
On first use, available saved radio queues populate memory once.
**Clear memory** resets every seed and Find Music recipe without deleting saved queues or importing them again.
Turning the checkbox off ignores memory without erasing it.

Freshness costs closeness: especially with DPP, later queues can move farther from the seed.
Once remembered vectors span the embedding space, DPP can return no further independent directions.
It never silently forgets tracks or switches algorithms to fill a queue. Saved-queue replay sends
the saved list directly, independently of memory.

### Graph Explorer

**Choose this for indirect connections through nearby music.** Compute the exact terminal
probabilities of a non-backtracking traversal over the index's nearest-neighbor graph, following
up to 100 links. Each outgoing link is equally likely; the immediately previous node cannot be
the next step.

**Stop chance per link** controls route length: higher values favor shorter routes, lower values
carry probability farther. Results rank by terminal probability, not direct seed cosine.
The graph must match the active index. Inactive/copy nodes are removed and affected neighbor
lists are repaired by exact cosine search before traversal.

## Find Music

Use a text description, select a specific recording, or add several ingredients.
Recording lookup requires you to confirm the recording; it never silently uses the first title match.
The result list is the queue: replace upcoming tracks or append exactly the displayed order.

### One Description

- **Closest:** text-to-audio cosine order across the selected date scope.
- **Varied (DPP):** the same cosine relevance supplies DPP quality for a less redundant set.

This uses the whole selected scope, not the radio's nearest-subset setting.

### All Of

**Choose this for music satisfying multiple ingredients together.** Each ingredient receives
tie-aware cosine percentiles within the selected library/date scope. For **Less like**, cosine
is negated before calculating percentiles. A percentile is the fraction of library identities
scoring at or below that candidate; equal scores receive equal percentiles. Combine them as:

```text
score(track) = product(percentile(track, ingredient)^weight)
```

Weights sum to 1. A low percentile for an ingredient reduces the combined score, which favors
joint matches rather than an interleaved union of each ingredient's favorites.

- **Ranked:** return that objective's strongest matches in order.
- **Varied (DPP):** use the same objective as quality and select a less redundant set from the
  complete scope. Its fixed quality exponent is 64, chosen from the comparisons in
  [Measurements](EVALUATION.md#recommendation-tradeoffs).

**Priority** sets exact shares from 10/90 to 90/10 in 10-point steps for two ingredients. With more ingredients,
adjusting one weight redistributes the remaining unheld budget. **Hold** preserves a share while
editing; adding/removing ingredients releases holds. Hold is an editor aid, not another algorithm input.

### Refine

**Choose this for a secondary preference inside one anchor's neighborhood.** Take the primary
ingredient's nearest 0.25%, 0.5%, 1%, or 2% of eligible non-seed identities, then rank inside that
set by the secondary ingredient. Candidate count is exactly `ceil(eligible count * fraction)`.
Secondary ties use primary percentile, then stable identity order.

The primary must be Like; the secondary can be Like or Less like. There are no weights because
the ingredients have different jobs. The neighborhood never widens to fill a queue.

### Remember Previous in Find Music

The shared Settings switch also remembers successful Find Music enqueues for each recipe.
Ranked and Closest return the strongest unseen matches in their original ranking order.
Refine skips remembered tracks inside its original primary neighborhood without widening it.
Varied conditions DPP on their embeddings as well as excluding them.

Memory follows the ingredients, directions, and weights (or Refine's primary role), not the
result count, date filter, planner, or neighborhood width. Searching without enqueueing does not
add memory. All confirmed queues are recorded even with the switch off; Clear memory resets them.
Global match ranks stay unchanged. Replaying a saved queue still sends its exact track list.

### History

Saved queues can be replayed exactly. Recent searches can also be repeated using their original
settings or current Settings, without overwriting your defaults. Opening a recent search restores
its text, confirmed recordings, weights, and planner; Back leaves those inputs ready to edit.

## Favorites

Favorites ranks individual recording and text seeds by confirmed queue requests. Each active
ingredient in a multi-seed search counts once per request. Failed deliveries and unqueued search
previews do not count. Counts are independent of history deletion and Clear memory; initial counts
come from retained queue history because older request frequencies were not recorded.

Opening a seed ranks the current full active library by cosine similarity, excluding the seed's
recording copies. Settings controls this list's size (default 100). Radio date, diversity, and
memory controls do not filter this nearest list. Tapping a match puts that recording first in the
upcoming queue, followed by radio generated using current Settings; playback is left alone.
The song seed header starts radio from the favorite itself using the same settings.

**New** marks recordings added by the latest index publication, only until the next publication.
Opening Favorites does not clear it. An added copy of an older recording is not marked New.
The Favorites list also marks seeds with any New result inside their configured nearest list.
Nearest rows show elapsed days since their embedding DB row was first added, using the same
`tracks.created_at` clock as the shared Added to index filter. Favorites itself always ranks the full library.
Expanded main-screen queue rows show the same age beside album and duration.
These checks run in the background and are cached for the index, active library, and list size.
This tracking starts with the first index update after installing Favorites.

## Identity and Repeatability

Complete-content copies with matching indexed spans share a recommendation identity when the app
has evidence they are equal. Different masters, live versions, and FLAC/MP3 copies can still remain
separate. Similar embeddings alone are not proof of duplicate identity.

The same index, active library, request, and remembered set produce the same ordered identities.
Uniform shuffle's order value is part of the request. Independently generating embeddings on
different hardware is not promised to produce byte-identical vectors.
