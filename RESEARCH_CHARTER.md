# Evidence-First Improvement Charter

This charter governs changes to the intelligence and embedding-to-queue behavior of Poweramp Start
Radio. It is deliberately stricter than ordinary feature ideation: a recommendation option earns
its place only when its exact promise, implementation, measured effect, and listening utility agree.

## Product North Star

The app should help a listener discover remarkable recordings already present in their library.
It should let them express the sound they want through:

- one current recording;
- one text description;
- several text or explicitly confirmed recording ingredients;
- a small set of controls with predictable musical effects.

Musical intelligence comes from the shared embedding space and declared algorithms. Poweramp
metadata may define identity, artist limits, library age, and display, but must not become hidden
taste intelligence.

## Acceptance Standard

For every serious recommendation idea:

1. State the musical request it is intended to satisfy.
2. Define exactly how embeddings become an ordered queue.
3. Bind the experiment to one immutable database and active-library generation.
4. Compare it only with existing behavior that serves the same request.
5. Measure relevance, redundancy, coverage, anchor satisfaction, determinism, domain correctness,
   duplicate exposure, and runtime.
6. Inspect real ordered results across mainstream, niche, non-English, electronic, ambient,
   acoustic, dense, and sparse neighborhoods.
7. Stress filters, queue sizes, artist constraints, duplicate versions, small domains, and
   incomplete graphs.
8. Reject it, refine it, use it to improve an existing promise, or promote it for device listening.

The ledger for a candidate records:

- database, embedding, graph, and active-library identities;
- hypothesis and exact formula;
- ordered track identities and aggregate metrics;
- timing and repeat fingerprints;
- qualitative observations;
- final retain, replace, merge, or reject decision.

## Mode Curation

Every visible option must complete this sentence:

> I would choose this when I want...

Apply these rules:

- If two algorithms answer the same request, keep the consistently stronger one.
- If a new algorithm improves an existing promise, improve that mode instead of adding a synonym.
- If behavior differs only in degree, prefer one measured control over two mode names.
- If an option mostly reorders the same recordings without changing utility, do not expose it.
- If a distinction needs a long mathematical explanation before it can be chosen, keep it
  internal unless listening establishes unusual value.
- Contextual result-set planners remain inside Find Music rather than becoming global radio modes.
- Do not suppress a genuinely distinct capability merely to meet a numerical mode cap.

The target is the smallest complete musical vocabulary, not the fewest possible algorithms.

## Control Standard

A visible knob must:

- be a stable semantic input to the stated algorithm;
- have at least two applicable values in the current context;
- cause a significant, predictable, and verified change in its declared tradeoff;
- preserve exact request persistence and replay.

Do not expose cache sizes, scheduler intervals, certificate prefixes, normalization constants,
mathematical aliases, or other implementation parameters as fake agency. A fixed measured
parameter may remain part of a versioned algorithm contract without becoming a user control.

## Correctness Invariants

- Same embedding generation plus the same complete request produces the same ordered identities.
- The displayed candidate domain is the domain the selector actually uses.
- Global nearest-rank evidence retains one consistent meaning across filters and modes.
- Preview and benchmark work never changes the Poweramp queue.
- Find Music queues the displayed result order, not a silent rerun.
- Seed and proven full-file copies are conditioned out where identity evidence justifies it.
- Similar embeddings alone never prove duplicate file identity.
- Greedy DPP claims equality only to the complete-domain greedy sequence, not global MAP
  optimality.
- Persisted requests fail closed when generation or evidence bindings no longer match.

## Performance And Reliability

Quality and correctness outrank startup or one-time cost. Performance work is accepted when it
preserves the exact output contract or makes an explicit domain tradeoff visible to the user.

On-device indexing additionally requires:

- complete-span audio policy;
- exact progress stage naming and evidence-based ETA;
- durable pause, resume, crash recovery, retry, and exclusion state;
- atomic generation publication;
- no repeated hashing of unchanged multi-gigabyte model files;
- measured device/host embedding parity;
- a graph update whose proof matches the published bytes.

Server indexing requires a durable private ledger, explicit baseline semantics, atomic cumulative
publication, readable status/ETA, and an overlap-safe phone merge.

## Evidence Versioning

`EVALUATION.md` is the public summary of the latest accepted evidence. A dated local report may
contain detailed queues, hashes, screenshots, or device logs, but public claims must be recomputed
from its machine-readable payloads and stripped of personal-library rows.

The July 30, 2026 V2 conclusion used 89,737 CLaMP3 embeddings and retained:

- All-of Ranked and contextual Varied;
- Refine with four measured primary-neighborhood widths;
- Like/Avoid;
- the visible single-seed mode set documented in `MODES_AND_KNOBS.md`;
- the identity-correct Graph Explorer topology.

That snapshot is a baseline, not a permanent benchmark. Future work must declare a new generation
identity and compare only the behavior it changes.

## Completion Rule

An investigation is complete when major plausible families have been considered and further work
no longer produces a distinct improvement in specificity, relevance, breadth, freshness,
expressive control, reliability, or listening utility.

The production UI receives only the winners. Rejected candidates, calibration machinery, raw
evidence, and experiment labels remain outside the listener-facing app.
