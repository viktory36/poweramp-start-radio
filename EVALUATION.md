# Measurements

These are measurements on one personal CLaMP3 library, not a benchmark of universal musical taste.
July 2026 runs used 89,737 vectors and 89,528 active recording identities; October runs used
95,786 vectors and 95,576 active identities. Vectors have 768 dimensions.
Unless stated otherwise, queues requested 30 tracks. Device timings exclude Poweramp delivery.

## Recommendation Tradeoffs

### All-of: Ranked vs Varied

Thirteen composed requests (twelve with two ingredients, one with three) were run twice per planner.
Every repeat returned the same ordered IDs. Relative to Ranked, Varied changed:

| Measure | Mean change |
| --- | ---: |
| Weighted All-of objective | -0.000640 |
| Pairwise queue cosine | -0.048865 |
| Unique artist credits | +3.46 |
| Membership overlap | 22.15 of 30 tracks retained |

Varied's farthest selected All-of rank had median 46, range 37-329.
At quality exponent 64, it reduced redundancy with little loss of joint match score in these requests.
This supports a Ranked/Varied choice; it does not establish that Varied always sounds better.

### Refine Neighborhood Size

Seven two-ingredient requests were evaluated at each width:

| Primary width | Candidates | Median primary percentile floor | Median secondary percentile gain |
| --- | ---: | ---: | ---: |
| 0.25% | 224 | 0.99754 | 0.2444 |
| 0.5% | 448 | 0.99512 | 0.2907 |
| 1% | 896 | 0.99024 | 0.2987 |
| 2% | 1,791 | 0.98075 | 0.3029 |

Widening consistently traded primary fidelity for stronger secondary matches.
Top-30 overlap with Ranked All-of ranged from 1 to 25, so Refine changed membership, not just order.
The default is 1%.

### Ingredient Controls

In two-ingredient sweeps, own-ingredient satisfaction correlated with requested weight at
0.965-0.980 for Ranked and 0.908-0.954 for Varied. Three-ingredient sweeps gave 0.929-0.966.
Small adjustments sometimes changed only order. Like/Less like changed both scores and membership
in three probes; this tests the control, not the model's understanding of every description.

### Repeated Radios with Memory

A host pilot used eight musically diverse seeds, four settings, and five successive queues:
160 queues filled, with no repeated recording IDs within each sequence.
Metadata-identical copies were grouped for this pilot; it did not reproduce the phone's full
duplicate-identity policy.

| Setting | Initial mean seed cosine | Fifth queue mean seed cosine |
| --- | ---: | ---: |
| DPP seed pull 1 | 0.6443 | 0.5263 |
| DPP seed pull 2 | 0.7158 | 0.6321 |
| MMR relevance 0.5 | 0.6782 | 0.6646 |
| MMR relevance 0.9 | 0.7695 | 0.7000 |

Memory creates freshness, not automatically better matches. DPP moved farther from the seed;
stronger seed pull retained more closeness. This pilot covers only five iterations,
with at most 120 prior tracks. Production memory has no cap.

An October read-only device run then used **201 remembered tracks** and the actual active catalog:

| Check | Result |
| --- | --- |
| DPP | 30 new tracks; first generation 23.203 s |
| MMR | 30 new tracks; first generation 958 ms |
| Repeated requests | Identical ordered IDs; no remembered identities returned |
| Seed ranks | Every returned rank matched an independent full-library calculation |
| Composed rank lookup | 100 rows agreed with full-library All-of order |
| Memory storage | Survived reopening and history removal; duplicate writes were harmless |
| Clear memory | Stayed empty after reopening and presenting old history |

A native full-domain DPP run and the adaptive path returned the same 30-track sequence with
30 remembered vectors. A 128d orthogonal fixture returned one DPP result after 127 remembered
directions and none after 128; MMR could still return the remaining candidate.
This confirms DPP's rank exhaustion without arbitrary forgetting or fallback filling.
These runs did not enqueue music or assess the new UI/listening experience.

### Find Music with Memory

An October 12 native device simulation used 96,284 vectors and 96,074 active recording identities.
After remembering the first 30 ambient matches, three Varied queues each returned 30 unseen
recordings. Planning took 1.135, 1.199, and 1.569 s, excluding model loading and ranking.
Every queue matched full-domain conditional DPP exactly and reproduced its ordered IDs.
Ambient + guitar Varied returned 30 unseen recordings in 771 ms of planning, also matching
full-domain DPP. Original objective ranks were retained throughout.

Closest continued at ranks 31-60 after the first 30 were remembered. Refine kept its original
241-recording neighborhood: with 236 remembered it returned five, then none with all remembered.
A temporary store retained 251 recordings across reopening, ignored failed enqueues, and stayed
empty after clearing despite old history being presented again. Small domains shortened or
exhausted rather than repeating tracks. The simulation did not enqueue music, change radio
memory or session history, or assess listening preference and the new UI.

### Favorites

A read-only October device run used 96,558 vectors and 96,417 active recording identities.
Recording and text top-100 lists exactly matched an independent full cosine sort; ranking and
materialization took 74-87 ms and 25 ms respectively over two runs, excluding library/model setup. Opening one
recording favorite through the actual screen took 623 ms after app startup.

Isolated device stores verified per-request individual multi-seed counts, repeat-safe history
import, persistence, and failed-delivery exclusion. Generation-local addition lists survived
previous-generation pruning and retries, then reset at the next publication. Newly added copies
of existing recording identities were not marked New. Exact selected-song Poweramp binding and
current radio configuration were verified without delivering a queue. Both new screens were
visually inspected. Existing session history was byte-identical before and after these checks.

First-screen New indicators were compared with materialized top-100 results for all 115 song
favorites using simulated addition IDs: two marked and 113 unmarked, all agreeing. Song/text
cutoff checks marked a seed for a new result at rank 100 with N=100, but not N=99. Cached flags
survived reopening and were not reused after index, active-domain, or list-size changes.

### Index-Added Dates

On October 20, all 96,627 embedding-row dates matched independent UTC parsing of `tracks.created_at`.
The active library's 7/30/90-day windows contained 343/841/15,713 rows, versus
253/763/15,635 under Poweramp dates. Radio and Find Music now use the embedding dates.
All five radio modes stayed inside an index-date-filtered fixture despite deliberately conflicting
Poweramp dates, while seed ranks remained global. Replacing an imported row retained its original
timestamp through retry. Favorites age labels matched the DB and were visually inspected;
generated main-screen queue rows carry the same timestamp for expanded details and saved history.

## Device Runtime

Earlier day-window measurements below used Poweramp dates.

Warm July measurements:

| Request | Results ready |
| --- | ---: |
| Closest radio | 304-382 ms |
| MMR, maximum relevance | 805-1,088 ms |
| DPP radio | 2.7-19.6 s, depending on seed |
| Graph Explorer, full queue | 877-976 ms |
| Graph Explorer, sparse one-result case | 110-114 ms |
| All-of Ranked | 339-356 ms |
| All-of Varied | 524-567 ms |
| Refine, 1% | approximately 347 ms |
| Four-text Varied, 14-day scope of 9,079 identities | approximately 208 ms |

A five-seed full-domain DPP check averaged 8.227 s. An extreme-control case in the broader matrix
took 46.498 s. Quality-first, full-domain DPP is not always quick.

The broad matrix ran 660 executions: 11 seeds, 30 configurations, and exact repeats.
There were no request failures, inactive results, repeated embedded IDs, or queue mutations.
After subsequent selector repairs, a focused 40-execution check again reproduced exact order.
Fourteen composed-mode executions also reproduced their results without queue mutations.

Graph identity repair collapsed 198 copy occurrences and repaired 520 neighbor slots in 343 rows.
Of 11 tested seeds, a broken two-result queue became 30; nine full queues were unchanged;
one genuinely sparse component still returned one result.

## Audio Indexing and Server Merge

| Measurement | Result |
| --- | ---: |
| Device/host audio embedding cosine, two tracks | 0.99903 and 0.99709 |
| MERT GPU inference | 196-198 ms per five-second window |
| CLaMP3 audio projection | 23-24 ms per segment |
| Incremental graph update | 13.4-13.8 s |
| Server CPU embedding rate, 31 publication batches | Median 30.907 s/track; p90 42.342 s/track |
| Server export | 8,832 embeddings, 43.3 MiB |
| Phone no-op merge of that export | 42.8 s; all rows already indexed |

Pause/resume reproduced the same device vector. A controlled 25-second track completed the
ready -> indexed -> cleanup lifecycle. A lost graph-validity record caused an avoidable
14m39s full rebuild; the propagation fix was made, but a subsequent real addition on that final
build was not measured.

The server merge serialized with a concurrent library comparison and preserved the active index.
That run establishes overlap-safe **no-op** behavior, not positive-add merge validation.
Server rates are derived from batches, not independent timing samples for all 8,832 tracks.

## Limits and Reproduction

- CLaMP3 can miss niche genres, languages, specific recordings, or descriptions.
- Duplicate identity remains incomplete: seven apparent duplicate slots remained in six of
  54 reviewed queues. More aggressive metadata grouping also merged distinct recordings.
- DPP certification establishes the complete-domain **greedy** sequence, not global MAP optimality.
- Determinism requires the same vectors, library, request, and remembered set; cross-hardware
  model outputs are not promised byte-identical.
- Earlier measurements describe the builds and corpora tested, not every later source change.

[Host experiments](desktop-indexer/experiments/README.md) contains the reference evaluators;
[Android diagnostics](android-plugin/README.md#device-diagnostics) describes on-device checks.
Full corpus runs require your own database snapshot. Personal library rows and device logs are
not included in the repository.
