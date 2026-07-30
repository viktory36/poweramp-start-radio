# Modes and Knobs

Poweramp Start Radio offers two main recommendation surfaces:
- radio from the current Poweramp track
- Find Music from text and confirmed recording ingredients

This guide walks through the modes and controls behind both, and how they shape the results you see.

## App Offerings

### Current-track radio

The app reads the current Poweramp track, matches it to the embedding database, and builds a queue from that seed.

Current-track radio is shaped by:
- the selection mode
- mode-specific knobs such as MMR relevance, DPP seed pull, or graph stop chance
- optional drift
- the shared Poweramp-added date window
- artist repetition controls
- queue size

### Find Music

A single positive description uses raw text-to-audio cosine over the complete active indexed
library scope selected in Settings. The result screen is the queue: it shows the exact ordered rows
and can replace or append that displayed ranking in Poweramp.

Structured Find Music combines up to several text descriptions and recording ingredients. The
request explicitly declares:
- `All of`: a weighted geometric mean of tie-aware active-library percentiles; Like and Avoid
  ingredients are supported
- `Refine`: an exact nearest neighborhood around one positive primary ingredient, ordered by the
  other ingredient

Final weights or Refine neighborhood, signs, operator, result count, exact recording anchors,
active-library binding, ranking evidence, and copy reduction are saved with the result and its
session history.

## Common Radio Controls

### Queue Size

The shared Settings slider chooses how many tracks a new radio or Find Music request asks for.
It ranges from 10 to 100 in steps of 10.

Larger queues give the mode more room to express its selection behavior. Smaller queues make the earliest choices matter more.

### Poweramp Added Date

The shared Settings control can use all dates or an exact rolling number of days. Poweramp's
first-added timestamp defines age, and each day means one rolling 24-hour period.

The filter changes the eligible recommendation domain for every radio mode and Find Music. It does
not change the global `# nearest` evidence: that rank remains measured against the complete active
identity library so queues from different date scopes remain comparable. Recordings with no usable
first-added timestamp are included only by `All dates`.

### Artist-Credit Limits

The shared switch enables or disables both artist-credit controls. When disabled, neither the cap
nor spacing affects selection.

### Max Per Artist

A cap on how many tracks with the same complete artist-credit text can appear in the queue.
Matching ignores case and surrounding whitespace; blank credits are unlimited.

Each selection mode enforces the cap while building the queue. A rejected candidate can therefore
make the mode choose another track and change the later trajectory; this is not a cosmetic cleanup
of an already finished result.

### Min Artist Spacing

How many tracks the app tries to keep between appearances by the same artist.

This is often one of the most noticeable cleanup controls in the app. Even a strong selector can feel repetitive if artist repeats land too close together.

## Selection Modes

### Closest to seed

Closest ranks the complete eligible identity domain by cosine similarity to the original seed.

It has no selector knob. Adding one would only rename an implementation detail: the promise is the
exact nearest order. Queue size and explicit artist-credit limits can still change which rows are
delivered.

### Uniform shuffle

Uniform Shuffle assigns each eligible active indexed span one deterministic priority from the
stored shuffle seed. Full-file copies proven equal by complete content identity and exact indexed
span share one place; sampled, unproven, and legacy rows remain distinct.

Its only selector input is the reproducible shuffle seed. Every eligible identity receives equal
membership opportunity, preserving the distinct utility of ordinary shuffle.

### Maximum Marginal Relevance (MMR)

MMR starts with tracks closest to the current search, then re-scores the remaining candidates one by one.

A candidate keeps its relevance to the current query, but is penalized when it overlaps too much with the single chosen track it most resembles.

Mental model:
- retrieve a nearby neighborhood around the seed
- pick one strong result
- keep rejecting tracks that are too redundant with one already-chosen neighbor

What it brings:
- a close, readable relationship to the seed
- reduced overlap among already chosen recommendations
- a direct relevance-versus-variety tradeoff

Controls:
- `Relevance weight`
- `Neighborhood reach`

How `Relevance weight` changes MMR:
- higher values keep more of the queue close to the seed
- lower values let the diversity penalty bite harder
- very high values approach straight nearest-neighbor retrieval
- lower values still stay in the same neighborhood, but spend more of the queue covering different parts of it

`Neighborhood reach` is the nearest fraction of the active indexed library that MMR may
rerank. Without drift this is one fixed seed-nearest subset. With drift it is a nearest frontier
retrieved again around the evolving query after every pick. The measured default is `2%`. A wider
reach can expose farther directions, especially at low relevance weight; at high relevance weight
it may produce the same queue because no farther candidate can overcome the relevance term.

### Determinantal Point Process (DPP)

DPP greedily scores each remaining candidate against the chosen set as a whole.

Mental model:
- choose either the complete eligible non-seed identity domain or an explicit nearby neighborhood
- build the list as a set
- once several chosen tracks already occupy the same local neighborhood, another very similar track from that neighborhood ranks lower for the next slot

What it brings:
- broader coverage of the eligible search domain
- stronger resistance to the queue collapsing into one dense clump
- a more globally balanced set than MMR usually produces

Controls:
- `Seed pull exponent`
- `All eligible` or `Nearest subset`
- subset reach when `Nearest subset` is selected

`Seed pull exponent` raises non-negative seed relevance before DPP constructs its quality
terms. Higher values favor seed relevance more strongly; lower values let determinant novelty
carry more influence.

The default `All eligible` option uses a bounded working set and widens it until every greedy
choice is proven against every unseen eligible candidate. This reproduces the complete-domain
greedy sequence; it does not claim global DPP MAP optimality. Turning it off intentionally
changes the eligible domain to the chosen nearest fraction.

Useful contrast with MMR:
- `MMR` compares a candidate to the single chosen track it most overlaps with
- `DPP` compares a candidate to the chosen set together

### Graph Explorer

Graph Explorer uses the graph bound to the active embedding generation instead of ranking directly
from the seed embedding at runtime.

It computes the exact terminal distribution of a non-backtracking traversal, up to 100 followed
links. Each outgoing nearest-neighbor link is equally likely. Tracks rise when probability reaches
them through intermediate connections, not simply because they are the seed's closest direct
cosine neighbors.

Mental model:
- the library is a graph of local similarity links
- the walk starts at the seed
- after each followed link, probability either stops at that track or continues outward
- the immediately previous node is not revisited on the next link
- tracks with high terminal probability through plausible routes rise in the ranking

What it brings:
- indirect connections instead of only direct neighbors
- deeper exploration of the library's local structure
- a more exploratory mode than the embedding-scan selectors

Main knob:
- `Stop chance per link`

How `Stop chance per link` changes Graph Explorer:
- higher values favor shorter routes
- lower values carry more probability through longer routes
- every branch follows one of the stored nearest-neighbor links with equal probability

The graph topology contains one node for each proven active full-content identity. Sampled and
legacy rows remain separate. Any stored neighbor row affected by inactive or proven-copy removal is
refilled by exact cosine top-K before propagation. This is a correctness rule, not a user control.

Graph Explorer fails closed if the active generation has no matching graph.

## Drift

Drift is an optional modifier on the sequential radio path.

In the current app:
- drift applies with `MMR`
- drift is not used with `Graph Explorer`
- drift is disabled for `DPP`

Drift changes the query after each pick, so later selections are not based only on the original seed.

### Seed interpolation

Each step builds the next query as a weighted mix of:
- the original seed
- the most recently chosen track

Controls:
- `Seed weight`
- decay schedule

How `Seed weight` changes seed interpolation:
- high values keep the original seed strongly present throughout the queue
- low values let the latest pick steer the next search more aggressively

Decay schedules:
- `Hold`: the seed keeps the same strength throughout
- `Linear`: the seed fades steadily over the queue
- `Exponential`: the seed fades quickly at first, then more gently
- `Step`: the seed holds, then drops more abruptly

Step only offers drop points early enough to affect at least one remaining pick in the requested
queue. Its label states the number of picks completed before the lower seed pull takes effect.

This is the more anchor-aware version of drift.

### Momentum

Momentum keeps a running average of where the queue has been heading.

Instead of mixing only the seed and the latest pick, it blends each new pick into a continuing state that becomes the next query.

Control:
- `Carry-over`

How `Carry-over` changes momentum:
- high values make the running average change slowly
- low values let new picks redirect the query quickly

Momentum usually produces a smoother, less anchor-conscious trajectory than seed interpolation.

## How the Radio Controls Work Together

A compact way to read the radio surface:
- `Selection Mode` chooses the selection logic
- mode-specific controls change the character inside that logic
- `Drift` decides whether the query stays fixed or evolves over the queue
- `Max Per Artist` and `Min Artist Spacing` constrain artist repetition during selection

Examples:
- `MMR` + high relevance + no drift: tight neighborhood around the seed
- `MMR` + lower relevance + drift: a wider local neighborhood with a changing query
- `DPP` + full-library result: complete-domain greedy set coverage
- `DPP` + fixed reach: set coverage inside an intentional relevance boundary
- `Graph Explorer` + low stop chance: longer routes through the graph

## Search Controls

### Results

Find Music uses the shared Settings queue length. It chooses how many ranked rows are displayed and
therefore how many rows the direct queue action will submit. It does not change the ranking
objective.

### Ingredients

An ingredient is either:
- a text description embedded by the pinned CLaMP3 text model
- one explicitly confirmed active-library recording embedding

The app never silently picks the first title match for a recording ingredient. Every ingredient
has an explicit Like/Avoid sign where that direction has a truthful interpretation. `All of`
supports Like and Avoid ingredients. Refine requires a Like primary; its secondary ingredient may
be Like or Avoid.

### Weight and Hold

For two-ingredient All-of requests, Priority moves in 10-point steps from 10/90 through 90/10.
Those weights are exact shares of the weighted geometric mean. For larger All-of compositions,
moving one unheld weight redistributes the remaining unheld budget; Hold preserves an ingredient's
share while tuning the current set. Adding or removing an ingredient releases all holds before
allocating the new 100% mix. Hold is an editor aid, not a ranking input: history saves the exact
resulting weights, not the temporary Hold checkboxes used to reach them.

Refine has no weight control because its two ingredients have different jobs rather than shares of
one objective.

### All of

For every active ingredient, the app computes exact cosine against the selected library/date scope
and converts the tie-aware order into a scope percentile. Like ingredients reward high percentile;
Avoid ingredients reverse their percentile. The weighted geometric mean ranks rows that satisfy
the declared intersection.

All-of has one contextual Result set choice:
- `Ranked`: return the strongest weighted intersection in objective order
- `Varied (DPP)`: use that same complete All-of ranking as DPP quality and select a broader set
  from the joint neighborhood

Varied uses a fixed, measured quality exponent of 64. It is not a user knob: weaker exponents
produced visibly broader queues by sacrificing too much weakest-anchor satisfaction. The planner
starts from a bounded prefix but widens until its greedy choices are certified against every
candidate in the complete promised All-of domain. It retains each selected row's original All-of
rank as evidence.

The result evidence shows the All-of objective rank and each ingredient's relative position in the
active library. The exact objective score remains available only as diagnostic evidence; it is not
relabeled as cosine or confidence.

### Refine

Refine first takes the declared nearest 0.25%, 0.5%, 1%, or 2% of eligible non-seed recording
identities around its positive primary ingredient. The size is exactly the ceiling of the eligible
identity count times that fraction. It then orders only those candidates by the secondary
ingredient's effective percentile, followed by primary percentile and stable identity tie-breaks.
The secondary ingredient may be Like or Avoid.

The app never widens the primary neighborhood to fill Results. If the chosen neighborhood contains
fewer identities than the requested result count, it returns the available set. Result evidence
shows the exact primary and refiner ranks, the neighborhood size, and the full ingredient-ranking
domain.

### Single-description search

Exactly one positive text ingredient takes the simpler raw-cosine path. Results show the exact
eligible-domain rank and top-domain fraction. The raw cosine remains optional diagnostic evidence,
not the listener-facing measure. This keeps the common description-to-music workflow direct
instead of wrapping it in a one-ingredient composition that adds no choice.

Its contextual Result set choice is:
- `Closest`: return the strongest text/audio cosine matches in objective order
- `Varied (DPP)`: preserve text relevance as quality while selecting a less redundant set from
  the complete promised text-candidate domain

### Direct queueing and history

`Queue N` queues the rows already on screen, either replacing upcoming tracks or appending after
them, as one exact result-list session. The durable session stores the query, library binding,
ranking evidence, result order, and copy-reduction evidence so history and requeue preserve the
request that actually ran.

## Control Inclusion Rule

A visible control must be a stable semantic input to the stated algorithm, have at least two
distinct applicable values in the current context, and be capable of changing its declared plan.
Mathematical aliases, current-library count aliases, scheduler tick sizes, derived buffer sizes,
and unbenchmarked resource profiles are not user agency and stay out of the ordinary UI. Saved
intent may be preserved while an inapplicable control is hidden or disabled.

This rule does not require every value to change every queue. It requires the choice itself to be
real, reproducible, and understandable from the promise of the feature.

## Source of Truth

Implementation lives in:
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/RecommendationEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/ActiveRecommendationDomain.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/ClosestSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/MmrSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/DppSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/GraphExplorerSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/UniformShuffleSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/DriftEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/GeoMeanSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/FindMusicAllOfQueuePlanner.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/FindMusicTextQueuePlanner.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/ui/MainViewModel.kt`
