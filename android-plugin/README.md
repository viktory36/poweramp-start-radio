# Android App

This directory contains the Android application for Poweramp Start Radio. It is the offline
recommendation and indexing client: CLaMP3 embeddings supply musical relevance, explicit
algorithms turn them into ordered result sets, and Poweramp supplies the current library and queue
boundary.

The current build uses application ID `com.powerampstartradio.v2`.

## User Workflows

- **Radio from Now Playing** builds a queue from Poweramp's current track, whether it is playing or
  paused.
- **Find Music** retrieves recordings from text and confirmed song ingredients. `All of` can return
  the strongest joint matches or a less redundant DPP-selected set. `Refine` holds one ingredient
  as a hard neighborhood and orders inside it with a second ingredient.
- **Queue delivery** either replaces Poweramp's upcoming queue or appends after it, then reads the
  queue back before reporting success.
- **History and widget** use the same durable request and verified delivery path. Saved sessions can
  be requeued in their recorded order.
- **On-device indexing** finds Poweramp tracks absent from the active index, embeds selected tracks,
  and atomically publishes a new index generation.
- **Server merge** accepts a compatible graphless export from the desktop server indexer and merges
  only rows that can be bound to the current Poweramp library.

Poweramp metadata is used for identity, CUE spans, dates, display, explicit artist-credit limits,
and queue delivery. It is not a hidden musical relevance signal.

## Recommendation Surface

Radio from one seed exposes five distinct choices:

- **Closest to seed**: cosine order from the fixed seed.
- **MMR**: seed or evolving-direction relevance minus resemblance to earlier picks.
- **DPP**: quality-weighted, set-wide diversity over either all eligible recordings or a declared
  nearest subset.
- **Graph Explorer**: terminal probabilities from deterministic non-backtracking paths over the
  validated similarity graph.
- **Uniform shuffle**: a reproducible uniform permutation without similarity ranking.

MMR alone supports an evolving direction. DPP, MMR, queue length, date scope, and artist limits
expose only controls that enter the recorded request. Results retain exact configuration and domain
evidence, including true seed-nearest rank over the full active library.

Find Music uses tie-aware corpus percentiles so ingredients with different raw cosine
distributions remain comparable:

- **Single description / Closest** orders the complete eligible scope by text-to-audio cosine.
- **Single description / Varied** uses that same relevance as DPP quality for a less redundant set.
- **All of / Ranked** orders by the weighted geometric mean of ingredient percentiles.
- **All of / Varied** uses that same objective as DPP quality to reduce result-set redundancy.
- **Refine** selects an exact primary neighborhood and ranks it by the secondary ingredient.
- **Like / Less like** explicitly controls an ingredient's ranking direction.

All requests are deterministic for the same validated index generation, active Poweramp library
binding, and complete recorded request.

## Build

One-time WSL setup:

```bash
cd android-plugin
./scripts/setup-wsl-android-env.sh
```

Build the debug app:

```bash
./gradlew --no-daemon :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Run the maintained host checks:

```bash
./gradlew --no-daemon \
  :app:testDebugUnitTest \
  :app:assembleDebug \
  :app:assembleDebugAndroidTest \
  :app:externalNativeBuildDebug \
  :app:lintDebug
```

Avoid `connectedDebugAndroidTest` on a phone whose private app data matters; Android's connected
test lifecycle may uninstall the target package. The acceptance scripts install fixed APKs and run
individual instrumentation classes instead.

## Required Private Assets

The app expects one exact model bundle at its private `filesDir` root:

- `mert.tflite`
- `clamp3_audio.tflite`
- `clamp3_text.tflite`
- `sentencepiece.bpe.model`

Export these with `poweramp-indexer export all`. They are not bundled in the APK, and the current UI
does not install them. A debug build can be provisioned with `run-as`:

```bash
MODEL_DIR=../desktop-indexer/models

for file in mert.tflite clamp3_audio.tflite clamp3_text.tflite sentencepiece.bpe.model; do
  adb push "$MODEL_DIR/$file" "/data/local/tmp/$file"
  adb shell run-as com.powerampstartradio.v2 \
    cp "/data/local/tmp/$file" "files/$file"
  adb shell rm -f "/data/local/tmp/$file"
done
```

Use **Settings > Import a music index** for the initial desktop `embeddings.db`. The importer
validates the SQLite rows, model identity, and embedding policy before publishing an immutable
generation. Do not overwrite files inside an active generation with ADB.

On-device indexing and server merge stage replacement generations privately. The active pointer
changes atomically only after the complete replacement reopens and validates.

## On-Device Indexing

The visible execution profile is **Full speed**. It embeds the complete logical track; ordinary
files end at physical EOS and CUE entries use their exact half-open Poweramp span. The native-rate
decoder, TorchAudio-Hann resampling policy, MERT windows, and CLaMP3 projection are bound by the
generation contract.

Before work begins, the app persists the exact selected occurrences and base generation. The
foreground service then maintains a durable ledger with:

- truthful stage progress and evidence-based ETA;
- pause and resume across activity or process loss;
- retryable and permanently blocked failure states;
- explicit Never-index choices;
- recoverable checksummed intermediate artifacts; and
- atomic database, embedding, graph, and receipt publication.

Android can stop `mediaProcessing` foreground work when its platform time budget is exhausted. The
service checkpoints and exposes a resumable state; it does not claim to bypass that limit.

## Permissions And Poweramp

`READ_MEDIA_AUDIO` is requested on first launch for source verification and decoding.
`POST_NOTIFICATIONS` is requested by the indexing workflow for foreground progress. Poweramp
content-provider access is a separate Poweramp-owned consent flow available from the app.

Current-track input is revalidated against Poweramp's provider before recommendation work. Queue
delivery pins exact provider occurrences, preserves playback state, and verifies the resulting
queue. The widget follows the same path.

## Current Limits

- Cross-encode, alternate-master, and live/studio duplicates are collapsed only when the app has
  proof that they share a stable recording identity. Similar metadata plus a close embedding is not
  treated as proof, so some apparent duplicates can remain.
- A clean install still needs the manual model provisioning step above. Find Music now checks for
  its index, text encoder, and tokenizer before opening, but the app does not yet install them.

## Validation

The curated concluding record is in [`../EVALUATION.md`](../EVALUATION.md), with PR-ready evidence
and limitations in [`../V2_RELEASE_NOTES.md`](../V2_RELEASE_NOTES.md). It covers the visible
selector/knob matrix, repeated composed searches, a server-merge overlap, full-span indexing,
pause/resume, activation, and cleanup. Its device evidence is bound to the accepted build named
there rather than every later source edit.

Useful scripts:

```bash
scripts/snapshot_device_acceptance.sh /tmp/pasr-before
scripts/snapshot_device_acceptance.sh /tmp/pasr-after
scripts/compare_device_acceptance.sh /tmp/pasr-before /tmp/pasr-after
```

`run_feature_acceptance.sh` exercises the selector matrix, while
`run_production_indexing_acceptance.sh` drives explicitly confirmed indexing cohorts. Debug-only
activities and receivers live in the debug manifest. They may change the Poweramp queue; use the
read-only snapshot scripts around any device experiment.

## Code Map

- `MainActivity.kt`, `ui/` - Compose UI, settings, Find Music, evidence, and persistence.
- `services/RadioService.kt` - durable recommendation requests, history, and verified queue
  delivery.
- `similarity/` - active identity domain, selectors, graph traversal, and result reduction.
- `indexing/IndexingService.kt`, `indexing/v2/` - durable indexing, merge, receipts, and immutable
  generations.
- `poweramp/` - current-track, provider, and queue integration.
- `widget/` - RemoteViews widget and private action ingress.
- `benchmark/`, `debug/` - debug-only measurement entry points.
