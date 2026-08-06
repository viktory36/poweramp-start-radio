# Repository Guide

This is the implementation reference for coding agents working on Poweramp Start Radio.
User-facing behavior belongs in `README.md`, `MODES_AND_KNOBS.md`, and `SETUP.md`.
Measured claims belong in `EVALUATION.md`.

## Product Boundary

Poweramp Start Radio is an offline intelligence layer over a Poweramp library. It:

- builds one CLaMP3 embedding per indexed track or logical span;
- creates radio queues from Poweramp's current recording;
- retrieves music from text and explicitly confirmed recording ingredients;
- sends the resulting ordered identities to Poweramp's queue;
- grows the index on-device or by merging a compatible server-produced index.

Musical relevance comes from embeddings and declared selection algorithms. Metadata is used for
identity, matching, artist constraints, added-date scope, and display. It is not used as a hidden
recommendation signal.

## Repository Layout

- `desktop-indexer/`
  - creates and inspects `embeddings.db`;
  - exports the four Android model/tokenizer files;
  - optionally watches server folders and publishes a graphless merge database.
- `android-plugin/`
  - binds the embedding generation to the current Poweramp library;
  - runs recommendation and Find Music requests;
  - queues exact displayed results into Poweramp;
  - performs crash-resumable on-device indexing and immutable index publication.
- `MODES_AND_KNOBS.md`
  - exact user-facing selector and control contracts.
- `EVALUATION.md`
  - latest curated host and device evidence.
- `RESEARCH_CHARTER.md`
  - rules for accepting or rejecting future intelligence changes.

Local evaluation snapshots and device captures are evidence, not distributable application
assets. Do not commit personal music metadata, databases, model caches, or raw device dumps.

## Model Contract

The active model path is CLaMP3:

```text
audio
  -> decode to mono
  -> resample to 24 kHz
  -> complete indexed-span 5 second MERT windows
  -> CLaMP3 audio encoder
  -> normalized 768d embedding

text
  -> official SentencePiece tokenizer
  -> CLaMP3 text encoder
  -> normalized 768d embedding
```

Android requires these exact private app files:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `sentencepiece.bpe.model`

The installed model receipt binds their hashes to an embedding specification. Initial database
import, server merge, and on-device indexing fail closed when this contract is absent or
incompatible.

## Immutable Music Index

Android does not mutate one public `embeddings.db` in place. A successful import, server merge,
cleanup, or indexing job:

1. prepares a private generation;
2. validates database, embedding, graph, model, and active-library bindings;
3. fsyncs the generation;
4. atomically publishes the active pointer;
5. prunes unreferenced generations after pointer commit while protecting nonterminal jobs.

The main entry points are under:

- `indexing/v2/V2IndexGenerationActivation.kt`
- `indexing/v2/V2BootstrapGenerationImporter.kt`
- `indexing/v2/V2ServerBundleMerger.kt`
- `indexing/v2/V2IndexingRuntime.kt`

Recommendation work and index mutation are serialized through:

- `services/RecommendationWorkAdmission.kt`
- `services/MusicIndexMutationAdmission.kt`
- `services/SingleFlightRequestReservation.kt`

Do not add a second mutable database path or bypass generation publication.

## Poweramp Binding And Identity

The current Poweramp library is read from its provider and stored as a generation-bound active
catalog. Matching prefers exact provider paths and exact indexed spans, then uses deliberately
bounded metadata fallbacks where path identity is unavailable. Read these together:

- `poweramp/TrackMatcher.kt`
- `indexing/V2ActiveLibraryCatalogLoader.kt`
- `data/StableTrackIdentityCatalog.kt`
- `services/StableTrackSpanReceiptReader.kt`

Full-content copies with proof-grade identity may collapse to one visible recommendation identity.
Sampled, legacy, and merely similar rows remain distinct. CLaMP3 cosine is not proof of duplicate
file identity.

The global `# nearest` evidence is computed against the complete active identity ranking even when
an added-date filter restricts which recordings are eligible for selection.

## Recommendation Paths

`similarity/RecommendationEngine.kt` is the single-seed integration point. Its visible selectors
are:

- `ClosestSelector`: exact seed-cosine order;
- `UniformShuffleSelector`: deterministic equal-opportunity order;
- `MmrSelector`: relevance minus redundancy to the most similar prior pick;
- `DppSelector`: greedy set selection with explicit quality and domain contracts;
- `GraphExplorerSelector`: exact terminal probability on the generation-bound identity graph.

Only MMR supports drift. Artist-credit constraints participate during selection rather than
cleaning up a finished queue.

Find Music uses separate contextual planners:

- `GeoMeanSelector` and `FindMusicComposition` for All-of and Refine;
- `FindMusicAllOfQueuePlanner` for certified complete-domain Varied All-of;
- `FindMusicTextQueuePlanner` for Closest or Varied single-text result sets.

All-of Ranked, All-of Varied, and Refine are contextual requests, not global radio modes. Keep the
visible vocabulary compact and preserve each option's distinct musical promise.

## Durable Requests And Queue Delivery

New work must retain exact request meaning across display, queueing, history, and replay.

- `ui/FindMusicQuerySpec.kt` owns the versioned Find Music request.
- `services/RadioRequestStore.kt` validates persisted request and result evidence.
- `services/RadioService.kt` performs radio selection and Poweramp delivery.
- `poweramp/PowerampHelper.kt` owns Poweramp queue operations.
- `ui/SessionEvidenceText.kt` converts saved evidence into user-facing explanations.

Find Music queues the already displayed ordered result list. It must not silently rerun the query
between display and delivery. Radio and Find Music must not mutate the Poweramp queue during
preview or benchmark-only work.

## On-Device Indexing

The production indexing path is a foreground service with a durable job ledger:

```text
Poweramp provider snapshot
  -> unindexed/exclusion resolution
  -> per-track decode and complete-span receipt
  -> MERT window cache
  -> CLaMP3 audio embedding
  -> private database update
  -> exact incremental graph update when proof permits
  -> immutable generation publication
```

Important invariants:

- only Full speed is user-selectable;
- pause/resume and process recovery use the durable job, not an in-memory counter;
- failed tracks retain an explicit retry/exclusion disposition;
- progress names the current measured stage and ETA uses observed stage rates;
- model hashes are computed once and reused until the files change;
- decode may overlap later inference only when the persisted stage contract remains recoverable;
- the app owns its wake lock and foreground-service lifetime during active work;
- a stale graph proof must never be inherited into a new generation.

Start with:

- `indexing/IndexingActivity.kt`
- `indexing/IndexingService.kt`
- `indexing/v2/V2IndexingJobPreflightPlanner.kt`
- `indexing/v2/V2IndexingPreflightPrimitives.kt`
- `indexing/GraphUpdater.kt`

## Server Indexer

`poweramp-indexer server` maintains a private ledger for watched folders and atomically publishes a
cumulative, graphless merge database. The phone remains authoritative about which Poweramp
recordings exist. Server rows are accepted only when their model and source-span contracts match
the phone merger.

Relevant modules:

- `desktop-indexer/src/poweramp_indexer/server_config.py`
- `desktop-indexer/src/poweramp_indexer/server_indexer.py`
- `desktop-indexer/src/poweramp_indexer/server_bundle.py`
- `android-plugin/app/src/main/java/com/powerampstartradio/indexing/v2/V2ServerBundleMerger.kt`

## Build And Verification

Desktop:

```bash
cd desktop-indexer
uv sync --extra dev --extra export
uv run --extra dev pytest -q
uv run poweramp-indexer --help
```

Android:

```bash
cd android-plugin
./gradlew :app:testDebugUnitTest :app:assembleDebug :app:assembleRelease
./gradlew :app:lintDebug
```

The current development application ID is `com.powerampstartradio.v2`. Debug-only receivers and
acceptance activities live in `app/src/debug`; do not move them into the release manifest.

Validate the contracts affected by the change. Recommendation changes may require formula, exact
candidate domain, stable ordering, filters, copy reduction, and native/reference parity. Indexing
changes may require a real track lifecycle, complete source span, device/host embedding cosine,
pause/resume, ETA evidence, and published generation hashes. Do not rerun unrelated matrices, and
do not substitute a green isolated unit test for the corresponding runtime claim.

## Change Discipline

- Read the current implementation before updating its documentation.
- Preserve same-generation determinism and fail closed on incomplete evidence.
- Keep metadata out of musical scoring.
- Do not expose controls that are inert, aliases, or unmeasured implementation parameters.
- Do not call greedy DPP globally optimal; the certificate proves equality to the complete-domain
  greedy sequence.
- Do not claim independently generated model outputs are byte-identical across hardware.
- Keep raw personal evidence local and publish only aggregate, reproducible claims.
