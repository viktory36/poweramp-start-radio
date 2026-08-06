# Android Development

The Android client uses Compose for UI, Kotlin for orchestration, native cosine/graph operations,
and LiteRT for model inference. Package: `com.powerampstartradio.v2`; minimum Android 8;
arm64 only. User workflows and controls are in [Setup](../SETUP.md) and
[Modes and controls](../MODES_AND_KNOBS.md).

## Build

Use JDK 17, SDK platform 36, NDK 27.2.12479018, and CMake 3.22.1.
With the SDK configured, run from this directory:

```bash
./gradlew :app:assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

For a fresh Linux/WSL toolchain:

```bash
./scripts/setup-wsl-android-env.sh
source .android-wsl-env
sdkmanager "platforms;android-36" "ndk;27.2.12479018" "cmake;3.22.1"
./gradlew :app:assembleDebug
```

The setup script installs the base tools; the extra SDK command supplies the versions used by
the current build. Model provisioning is covered once in [Setup](../SETUP.md#5-copy-the-models).

Host checks and instrumentation APK:

```bash
./gradlew :app:testDebugUnitTest :app:lintDebug :app:assembleDebugAndroidTest
```

An unsigned release APK can be built with `:app:assembleRelease`; sign it before installation.
Debug receivers and measurement activities are excluded from the release manifest.

## Runtime Map

Paths below are relative to `app/src/main/java/com/powerampstartradio/`.

| Area | Entry points |
| --- | --- |
| UI and requests | `MainActivity.kt`, `ui/MainViewModel.kt`, `ui/FindMusicQuerySpec.kt` |
| Radio algorithms | `similarity/RecommendationEngine.kt`, `similarity/algorithms/` |
| Find Music | `similarity/algorithms/GeoMeanSelector.kt`, `similarity/algorithms/FindMusicComposition.kt`, `similarity/FindMusic*QueuePlanner.kt` |
| Poweramp binding | `poweramp/PowerampHelper.kt`, `poweramp/TrackMatcher.kt` |
| Queue delivery and history | `services/RadioService.kt`, `services/RadioRequestStore.kt` |
| Per-seed radio memory | `ui/RadioSelectionMemory.kt`, `similarity/algorithms/DppConditioning.kt` |
| Library/identity | `indexing/V2ActiveLibraryCatalogLoader.kt`, `data/StableTrackIdentityCatalog.kt` |
| On-device indexing | `indexing/IndexingService.kt`, `indexing/v2/V2IndexingRuntime.kt` |
| Import and merge | `indexing/v2/V2BootstrapGenerationImporter.kt`, `indexing/v2/V2ServerBundleMerger.kt` |
| Atomic index activation | `indexing/v2/V2IndexGenerationActivation.kt` |
| Widget | `widget/` |

Native code is in `app/src/main/cpp/`; debug-only tools are in `app/src/debug/`.

### Index Storage

An index update prepares a private generation containing SQLite, packed vectors, graph, and
identity records. Validation completes before the active pointer changes atomically.
A durable indexing ledger retains selected tracks, completed stages, failures, and intermediate
work for recovery. Recommendation work and index publication are serialized to avoid stale reads
and competing updates.

On-device audio is decoded to mono, resampled to 24 kHz, and processed in five-second MERT windows,
then projected through CLaMP3 into one normalized 768d vector. Ordinary files use their decoded
span; CUE entries use their Poweramp offsets. See the
[desktop pipeline](../desktop-indexer/README.md#embedding-pipeline) for segmentation details.

Android's foreground-service time limits still apply. If the platform stops media processing,
the job must checkpoint and remain resumable rather than claiming to run indefinitely.

### Behavior to Preserve

- Relevance uses embeddings; metadata is limited to identity, dates, artist constraints, and display.
- Same index, active library, request, and remembered set yield the same ordered identities.
- A DPP working prefix must reproduce full-domain greedy selection before claiming All eligible.
- Find Music queues the displayed list, not a rerun; preview work never changes Poweramp's queue.
- Queue delivery binds exact Poweramp occurrences and reads back the result before recording success.
- Model/source-span compatibility is checked before accepting embeddings.
- Remembered radios are durable independently of the bounded saved-session list.
- Global rank evidence uses the full active library even when selection is date-filtered.

## Device Diagnostics

Avoid `connectedDebugAndroidTest` on an installation whose data you want to keep:
its install/uninstall lifecycle can remove the target app. Install both APKs explicitly and run
a selected instrumentation class instead.

```bash
adb install -r app/build/outputs/apk/androidTest/debug/app-debug-androidTest.apk
adb shell am instrument -w -r \
  -e class com.powerampstartradio.similarity.algorithms.RememberedRadioInstrumentedTest#nativeConditioningBeyond120StopsAtMathematicalExhaustion \
  com.powerampstartradio.v2.test/androidx.test.runner.AndroidJUnitRunner
```

That method checks conditioning/rank exhaustion using a synthetic native fixture without queueing
music. The class's other methods are corpus-specific: they require an installed index, cached
active catalog, and saved radio history with more than 120 remembered tracks for the test seed.

For broader device work, `scripts/snapshot_device_acceptance.sh` captures before/after state and
`scripts/compare_device_acceptance.sh` compares it. `run_feature_acceptance.sh` exercises selector
configurations; `run_production_indexing_acceptance.sh` operates on explicitly confirmed indexing
cohorts. Some debug tools enqueue music or alter index state: read their options before running.

Keep databases, music metadata, model files, screenshots, and device logs out of commits.
[Measurements](../EVALUATION.md) records aggregate results and their limits.
