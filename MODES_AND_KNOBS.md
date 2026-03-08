# Modes and Knobs

Poweramp Start Radio offers two main recommendation surfaces:
- radio from the current Poweramp track
- retrieval from text and multi-seed search

This guide walks through the modes and controls behind both, and how they shape the results you see.

## App Offerings

### Current-track radio

The app reads the current Poweramp track, matches it to the embedding database, and builds a queue from that seed.

Current-track radio is shaped by:
- the selection mode
- mode-specific knobs such as `Similarity` or `Return Frequency`
- optional drift
- artist repetition controls
- queue size

### Text retrieval

The app embeds a text description into the same `768d` CLaMP3 space as the audio library and retrieves matching tracks.

### Multi-seed retrieval

The app combines:
- a text description
- one or more song seeds
- positive and negative directions
- relative weights between those directions

Multi-seed retrieval can be used as a search surface on its own or as the starting point for queueing a chosen result list into Poweramp.

## Common Radio Controls

### Queue Size

How many tracks the app tries to queue.

Larger queues give the mode more room to express its selection behavior. Smaller queues make the earliest choices matter more.

### Max Per Artist

A cap on how many tracks by one artist can appear in the queue.

This sits on top of the selection mode. It does not decide the queue by itself; it simply stops one artist from taking over the finished result.

### Min Artist Spacing

How many tracks the app tries to keep between appearances by the same artist.

This is often one of the most noticeable cleanup controls in the app. Even a strong selector can feel repetitive if artist repeats land too close together.

## Selection Modes

### Maximum Marginal Relevance (MMR)

MMR starts with tracks closest to the current search, then re-scores the remaining candidates one by one.

A candidate keeps its relevance to the current query, but is penalized when it overlaps too much with the single chosen track it most resembles.

Mental model:
- retrieve a nearby neighborhood around the seed
- pick one strong result
- keep rejecting tracks that are too redundant with one already-chosen neighbor

What it brings:
- a close, readable relationship to the seed
- protection against obvious near-duplicates
- a direct relevance-versus-variety tradeoff

Main knob:
- `Similarity`

How `Similarity` changes MMR:
- higher values keep more of the queue close to the seed
- lower values let the diversity penalty bite harder
- very high values approach straight nearest-neighbor retrieval
- lower values still stay in the same neighborhood, but spend more of the queue covering different parts of it

### Determinantal Point Process (DPP)

DPP also starts from tracks closest to the current search, but it re-scores each remaining candidate against the chosen set as a whole.

Mental model:
- retrieve a nearby neighborhood around the seed
- build the list as a set
- once several chosen tracks already occupy the same local neighborhood, another very similar track from that neighborhood ranks lower for the next slot

What it brings:
- broader coverage within the nearby search region
- stronger resistance to the queue collapsing into one dense clump
- a more globally balanced set than MMR usually produces

Mode-specific controls:
- none

DPP is presented as a distinct selection behavior rather than a tunable family.

Useful contrast with MMR:
- `MMR` compares a candidate to the single chosen track it most overlaps with
- `DPP` compares a candidate to the chosen set together

### Random Walk

Random Walk uses the precomputed similarity graph instead of ranking directly from the seed embedding at runtime.

It starts at the seed track and follows local graph links from track to track. Tracks rise when many walks tend to end on them, not simply because they are the seed's closest direct cosine neighbors.

Mental model:
- the library is a graph of local similarity links
- the walk starts at the seed
- it either continues outward or jumps back and starts again
- tracks that are easy to reach through many plausible paths rise in the ranking

What it brings:
- indirect connections instead of only direct neighbors
- deeper exploration of the library's local structure
- a more exploratory mode than the embedding-scan selectors

Main knob:
- `Return Frequency`

How `Return Frequency` changes Random Walk:
- higher values jump back to the seed more often
- lower values let the walk travel farther before restarting
- high return frequency stays tighter around the seed's immediate graph neighborhood
- low return frequency gives the walk more chance to reach deeper or less obvious terminals

Random Walk works best when the graph is present and current.

## Drift

Drift is an optional modifier on the sequential radio path.

In the current app:
- drift applies with `MMR`
- drift is not used with `Random Walk`
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
- `None`: the seed keeps the same strength throughout
- `Linear`: the seed fades steadily over the queue
- `Exponential`: the seed fades quickly at first, then more gently
- `Step`: the seed holds, then drops more abruptly

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
- `Similarity` or `Return Frequency` changes the character inside that logic
- `Drift` decides whether the query stays fixed or evolves over the queue
- `Max Per Artist` and `Min Artist Spacing` clean up artist repetition after selection

Examples:
- `MMR` + high `Similarity` + no drift: tight neighborhood around the seed
- `MMR` + lower `Similarity` + drift: local neighborhood with a changing query
- `DPP`: nearby pool, but spread across more of it as a set
- `Random Walk` + low `Return Frequency`: longer excursions through the graph

## Search Controls

### Text Search Results

How many results the search screen shows for text-only and multi-seed retrieval.

This does not change the meaning of the search. It changes how much of the ranked list is shown or available for direct queueing.

## Multi-Seed Search

Multi-seed search combines several directions at once.

The active seeds can be:
- a text description
- one or more song seeds

Each seed contributes through three properties: weight, sign, and lock state.

### Weight

Weight controls how much a seed matters relative to the other active seeds.

The active weights are normalized against each other, so what matters is each seed's share of the total active mix.

Important details in the current app:
- the text seed can drop to `0%` when song seeds are present, allowing song-only multi-seed search
- song seeds keep a small nonzero floor while active so they stay present in the mix

### Sign

Every seed can be either:
- positive: `more like this`
- negative: `less like this`

Sign changes the search direction.

A negative seed tells the ranking system to favor tracks that do well on the other active seeds while ranking poorly against that seed.

### Lock

Lock freezes a seed's current share while the other unlocked seeds redistribute the remaining weight budget.

This is useful when one ingredient should stay fixed while the others are adjusted around it.

## How Multi-Seed Ranking Works

The app does not rank multi-seed search by a simple averaged embedding.

It uses a geo-mean-of-percentiles ranking.

In practical terms:
- for each seed, the app measures how every track ranks against that seed across the full library
- those raw similarities are converted into percentile positions
- the percentile positions are combined using the seed weights
- tracks rise when they do well across the weighted combination of directions

What this offers:
- better behavior than naive vector blending when seeds live in different parts of the embedding space
- clean handling of positive and negative seeds
- stronger bridge-finding between distant seeds

## What the Percentage Means in Multi-Seed Search

The shown `%` in multi-seed results is a display metric, not the internal geo-mean ranking score.

The ranking score is good for ordering results, but it bunches too close to `100%` to be useful as a reader-facing number.

The app therefore shows:
- cosine similarity to a signed weighted blend of the active seeds

That displayed `%` answers this question:
- how closely does this result match the combined direction implied by the active seed mix?

That makes the number:
- easier to read across single-seed and multi-seed search
- more spread out
- more informative than the raw geo-mean score as a user-facing percentage

So multi-seed search has two separate concepts:
- ranking score: used internally to order the results
- display percentage: shown in the UI because it is easier to interpret

## How the Multi-Seed Controls Work Together

A compact way to read the multi-seed surface:
- `Weight` decides how much each seed matters
- `Sign` decides whether each seed attracts or repels the search
- `Lock` decides which shares stay fixed while the others move

Examples:
- text `70%` + song `30%`: mostly the text world, nudged toward that song's neighborhood
- text `50%` + song `50%`: look for tracks that can satisfy both directions
- text `70%` + negative song `30%`: keep the text world, avoid the part of it that leans toward that song
- two positive song seeds plus one negative seed: triangulate a zone while carving out an unwanted nearby influence

## Source of Truth

Implementation lives in:
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/RecommendationEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/MmrSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/DppSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/RandomWalkSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/DriftEngine.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/similarity/algorithms/GeoMeanSelector.kt`
- `android-plugin/app/src/main/java/com/powerampstartradio/ui/MainViewModel.kt`
