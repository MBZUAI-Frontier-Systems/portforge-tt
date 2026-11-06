# ResNet-50 benchmark

## Frozen corpus (`corpus.json`)

The corpus is private and not shipped with the pack (`port.json` sets
`corpus.file` to null; pass it with `--corpus`). `build_corpus.py` builds it from
the ImageNet-1k validation split (`ILSVRC/imagenet-1k`, revision
`49e2ee26f3810fb5a7536bbf732a7b07389a47b5`), which is gated under the ImageNet
terms of access for research use, so its images are never redistributed.

- Dev pool: 48 validation images chosen by a seeded ranking. Smoke, bringup and
  perf use them; agents see them as public inputs.
- Held-out pool: 1,024 validation images disjoint from the dev pool: one image
  from each of the 1,000 classes plus 24 more, in shuffled order under opaque
  names. Agents never see them before the full stage.
- Synthetic: five generated images without labels for flat, saturated and
  high-frequency inputs: all zeros, mid-gray (128), white (255), an 8-pixel
  checkerboard of 0 and 255, and uniform noise from a seeded hash stream.

Every image is decoded as RGB, resized so its shorter side is 256 pixels
(bilinear, as torchvision's `Resize`), center-cropped to 224x224 and stored as
lossless PNG with the SHA-256 of its uint8 pixels, which the evaluator checks.
The seed is secret and recorded in the corpus file itself. The corpus identity
(SHA-256 of its canonical JSON) is pinned in `evaluate.py` and `port.json`.

Sanity check after building: the FP32 oracle's held-out top-1 accuracy should be
close to the published 76.13%. A value far from it points to a label-order or
preprocessing error in the corpus, not to the model.

## Gates and why

- Logits NRMSE at most 0.30 per row: sanity only. It catches wrong weights, a
  missing BatchNorm fold, a wrong stride or layout. Block-floating-point
  weights and low-fidelity math legitimately move logits far more than BF16
  does, so a tight per-row bound would judge format rather than the task.
- Behaviour at most 0.05: a row must not depend on batch composition, order,
  repetition or streaming. The reference is the candidate's own output for the
  same image, with room for block-floating-point rounding.
- Top-1 agreement at least 14/16 per labelled dev batch: the task-level check
  during development. The synthetic batch is excluded because its flat and noise
  inputs produce near-tied logits.
- Full stage, decisive: held-out top-1 accuracy at most 1.0 percentage point
  below the oracle's (at most 10 of 1,024 images) and top-1 agreement with the
  oracle at least 0.95 (at most 51 disagreements).
- Determinism: repeated identical calls are bit-identical.

The bounds were fixed in an earlier pilot, before this campaign, and were not
tuned to its results.

## Throughput cases

`perf` adds one-call cases at batch 16 and 32 and two 256-image streams, so a
pipelined `classify_stream` (overlapping upload, compute and readback) counts
toward the objective. Streams cycle through the 48 dev images; consecutive
batches in a stream always differ.

## Held-out discipline

Held-out images reach the candidate only in the full stage. If an operator
shares a failing full-stage case with the agents, later full results are no
longer blind for that case.
