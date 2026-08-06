# Third-Party Notices

Poweramp Start Radio is distributed under the root `LICENSE`. Third-party
software and model artifacts remain subject to their own licenses.

## Bundled Source

- **libsoxr** is included under `android-plugin/app/src/main/cpp/soxr/` and is
  licensed under LGPL-2.1-or-later. Its license and full LGPL text remain in
  that directory.
- **SentencePiece** is included under
  `android-plugin/app/src/main/cpp/third_party/sentencepiece/` and is licensed
  under Apache-2.0. Its vendored dependencies retain their own license files:
  Abseil (Apache-2.0), Darts-clone (BSD-3-Clause), esaxx (MIT), and
  protobuf-lite (BSD-3-Clause).

The Android and Python dependency graphs also contain third-party libraries
resolved by Gradle and `uv`; their upstream license terms apply. The lockfile
and Gradle build files are the authoritative dependency inventory for a given
build.

## External Model Artifacts

Model weights and tokenizer files are not distributed in this repository or
inside the APK:

- [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) is marked
  **CC-BY-NC-4.0**. Its non-commercial terms apply independently of this
  repository's MIT license.
- [CLaMP3](https://huggingface.co/sander-wood/clamp3) is marked **MIT**.
- [XLM-RoBERTa base](https://huggingface.co/FacebookAI/xlm-roberta-base) is
  marked **MIT**.

Anyone provisioning or redistributing model artifacts must review and comply
with the corresponding upstream license and attribution requirements.
