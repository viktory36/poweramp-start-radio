# Poweramp Start Radio

An AI plugin/sidecar for Poweramp on Android. Lets you queue songs similar to what's currently playing, or accept textual description of music and find the closest matching results from your library. Powered by [CLaMP3](https://github.com/sanderwood/clamp3). Happy listening!

## What's in the box?

<video src="https://github.com/user-attachments/assets/28190d6c-c090-45a0-8658-b174e09c5301" width="180" controls muted playsinline></video>

*Start a radio against the current playing song on Poweramp.*

<img width="360" height="780" alt="Image" src="https://github.com/user-attachments/assets/6a1514de-be74-429b-9107-a21809e7eea1" />

*Letting users be able to choose how their rec algorithm works was important. A few modes are available to choose from, with knobs to fine-tune how they work. More details in the [MODES_AND_KNOBS.md](MODES_AND_KNOBS.md) doc!*

<video src="https://github.com/user-attachments/assets/e1e0c52a-b23a-4e61-aede-944bf996499d" width="180" controls muted playsinline></video>

*Some of these amazing models (like MuQ-MuLan, CLaMP3) also support mapping textual audio descriptions into the same search space. It struggles with the some of the more niche queries but is overall not too shabby.*

<video src="https://github.com/user-attachments/assets/1b4ac709-a643-4518-afe8-6c8ae35d5699" width="180" controls muted playsinline></video>

*Downloaded a new album? Incremental updates to the db can be done fully on-device with a [LiteRT](https://github.com/google-ai-edge/LiteRT)-converted copy of CLaMP3 running on GPU.*

## The workflow

- use `desktop-indexer` to build the initial embeddings.db from the bulk of your library
- use `desktop-indexer` to export the Android model files
- build and install the Android app, then import embeddings.db and copy the model files to the phone
- start radio!
- occasionally rerun the desktop or on-device indexer to update the db with your library

[SETUP.md](SETUP.md) has more information on how to get everything running. 


## Giving thanks

These people and their work directly helped this project take shape.
- [maxmpz](https://github.com/maxmpz/powerampapi)'s Poweramp
- [abhishekabhi789](https://github.com/abhishekabhi789/LyricsForPoweramp)'s LyricsForPoweramp for a reference implementation of a Poweramp plugin
- [marceljungle](https://github.com/marceljungle/mycelium)'s Mycelium for a reference implementation that does almost exactly the same things but on Plex
- The team behind [MuQ](https://github.com/tencent-ailab/MuQ), [Music Flamingo](https://github.com/NVIDIA/audio-flamingo), [CLaMP3](https://github.com/sanderwood/clamp3), and [LiteRT](https://github.com/google-ai-edge/LiteRT)
