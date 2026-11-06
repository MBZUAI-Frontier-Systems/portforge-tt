#!/usr/bin/env python3
"""Build the private ResNet-50 ImageNet corpus (corpus.json) for ports/resnet50.

Not shipped with any pack: agents never see this file. Requires HF_TOKEN in the environment
(ILSVRC/imagenet-1k is gated; the token is passed to huggingface_hub and never written anywhere),
plus pyarrow and Pillow. Standard library otherwise.

    HF_TOKEN=... python ports/resnet50/build_corpus.py --out corpus.json --seed <int> [--cache-dir DIR]

A fresh held-out pool, disjoint from earlier corpora, keeps a later corpus blind:

    --exclude-corpus PATH   (repeatable) no held-out image may be an ImageNet image of PATH
    --reuse-dev-from PATH   copy PATH's dev and synthetic cases byte for byte; PATH's images are excluded
    --seed-file PATH        read the integer seed from a file (keeps it off the command line)

The corpus records no dataset row ids, so images are matched by the SHA-256 of their preprocessed
uint8 crop (the pixels_sha256 field). Every excluded image must be found in the validation split, or
the build aborts. Rows that repeat an excluded file under another label are excluded too and counted;
widespread label disagreement (misaligned rows) also aborts.

Output: canonical JSON (sorted keys, no spaces). Prints the corpus identity (SHA-256 of that
canonical JSON, the same value as portforge_tt.control.cluster.identity) to pin in evaluate.py and port.json.

Selection is deterministic: every candidate row is ranked by sha256(seed, role, row index), so the
choice needs no RNG library and reproduces exactly for the same seed and dataset revision.
"""
import argparse
import base64
import bisect
import hashlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import struct
import sys
import zlib

DATASET = "ILSVRC/imagenet-1k"
DATASET_REVISION = "49e2ee26f3810fb5a7536bbf732a7b07389a47b5"
PREPROCESSING = "PIL RGB; resize shorter side to 256 bilinear; center crop 224; uint8 HWC; PNG"
SHORT_SIDE, CROP = 256, 224
CLASSES, PER_CLASS = 1000, 50
DEV_COUNT, HELDOUT_COUNT = 48, 1024
EXTRA_HELDOUT = HELDOUT_COUNT - CLASSES
SYNTHETIC = ("syn-zeros", "syn-gray128", "syn-white255", "syn-checker", "syn-noise")
VALIDATION_SHARDS = 14
VALIDATION_ROWS = CLASSES * PER_CLASS
IMAGENET_SOURCE = "imagenet-1k/validation"
SYNTHETIC_SOURCE = "synthetic"
# Stamped into sources only by builds that exclude or reuse earlier corpora; absent means version 1.
CORPUS_VERSION = 2


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def corpus_identity(corpus):
    return hashlib.sha256(canonical_bytes(corpus)).hexdigest()


def ranked(seed, role, index):
    return hashlib.sha256(f"{seed}:{role}:{index}".encode()).hexdigest()


def select_pools(labels, seed, excluded=frozenset(), reuse_dev=False):
    """Dev and held-out row indices from the validation labels, in row order of the dataset.

    excluded: rows that must appear in neither pool (images of earlier corpora).
    reuse_dev: dev is not drawn, because its cases are copied from an earlier corpus.

    dev: the DEV_COUNT lowest-ranked rows outside excluded. Held-out: for every class, the lowest-ranked
    row of that class outside dev and excluded (one per class), plus the next EXTRA_HELDOUT lowest-ranked
    rows outside them. Held-out is returned in shuffled order (ranked by a second role).
    With nothing excluded and reuse_dev off, this is the original selection, unchanged.
    """
    if len(labels) != VALIDATION_ROWS:
        raise ValueError(f"validation split has {len(labels)} rows, expected {VALIDATION_ROWS}")
    excluded = set(excluded)
    order = sorted(range(len(labels)), key=lambda index: ranked(seed, "pool", index))
    dev = [] if reuse_dev else [index for index in order if index not in excluded][:DEV_COUNT]
    blocked = excluded | set(dev)
    first_of_class = {}
    for index in order:
        if index not in blocked and labels[index] not in first_of_class:
            first_of_class[labels[index]] = index
    if len(first_of_class) != CLASSES:
        raise ValueError("candidate pool misses some classes")
    chosen = set(first_of_class.values())
    extra = [index for index in order if index not in blocked and index not in chosen][:EXTRA_HELDOUT]
    heldout = sorted(chosen | set(extra), key=lambda index: ranked(seed, "order", index))
    if len(heldout) != HELDOUT_COUNT or len(set(labels[i] for i in heldout)) != CLASSES:
        raise ValueError("held-out selection violates the one-per-class rule")
    if set(heldout) & blocked:
        raise ValueError("held-out selection overlaps dev or excluded rows")
    return dev, heldout


def png_encode_rgb(pixels):
    """Lossless PNG (8-bit RGB, zlib level 9) from a uint8 array [H, W, 3]."""
    import numpy as np
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("PNG encoder expects [H, W, 3] uint8")
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def preprocess_image(image):
    """PIL image -> uint8 [224, 224, 3]; same arithmetic as torchvision Resize(256) + CenterCrop(224)."""
    import numpy as np
    from PIL import Image
    image = image.convert("RGB")
    width, height = image.size
    if width <= height:
        size = (SHORT_SIDE, int(SHORT_SIDE * height / width))
    else:
        size = (int(SHORT_SIDE * width / height), SHORT_SIDE)
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize(size, resample)
    left = round((size[0] - CROP) / 2.0)
    top = round((size[1] - CROP) / 2.0)
    image = image.crop((left, top, left + CROP, top + CROP))
    return np.asarray(image, dtype=np.uint8)


def synthetic_pixels(name, seed):
    import numpy as np
    shape = (CROP, CROP, 3)
    if name == "syn-zeros":
        return np.zeros(shape, np.uint8)
    if name == "syn-gray128":
        return np.full(shape, 128, np.uint8)
    if name == "syn-white255":
        return np.full(shape, 255, np.uint8)
    if name == "syn-checker":
        yy, xx = np.indices((CROP, CROP))
        cell = ((yy // 8) + (xx // 8)) % 2
        return np.repeat((cell * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    if name == "syn-noise":
        needed = CROP * CROP * 3
        stream = b"".join(hashlib.sha256(f"{seed}:{name}:{block}".encode()).digest()
                          for block in range((needed + 31) // 32))
        return np.frombuffer(stream[:needed], np.uint8).reshape(shape).copy()
    raise ValueError("unknown synthetic case " + name)


def pixels_sha256(pixels):
    """SHA-256 of a uint8 pixel array's bytes: a case's pixels_sha256, and the row identity used to
    exclude images of earlier corpora."""
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(pixels, dtype=np.uint8).tobytes()).hexdigest()


def case_entry(name, pixels, label, source):
    import numpy as np
    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    return {"name": name, "source": source, "label": label,
            "pixels_sha256": pixels_sha256(pixels),
            "png_b64": base64.b64encode(png_encode_rgb(pixels)).decode("ascii")}


def assemble_corpus(dev_entries, heldout_entries, synthetic_entries, seed, revision, extra_sources=None):
    """Schema shared with the test fixture in tests/test_evaluate.py. extra_sources adds provenance keys."""
    corpus = {
        "schema_version": 1,
        "sources": {"dataset": DATASET, "split": "validation", "dataset_revision": revision,
                    "preprocessing": PREPROCESSING, "seed": seed,
                    "selection": "dev and held-out rows ranked by sha256(seed, role, row); held-out presented in shuffled order",
                    "synthetic": "zeros, gray 128, white 255, 8-px checkerboard 0/255, sha256-counter uniform noise",
                    "license": "ImageNet terms of access (research use); synthetic cases are generated"},
        "cases": dev_entries + heldout_entries + synthetic_entries,
        "selection": {"dev": [e["name"] for e in dev_entries],
                      "heldout": [e["name"] for e in heldout_entries],
                      "synthetic": [e["name"] for e in synthetic_entries]},
    }
    corpus["sources"].update(extra_sources or {})
    return corpus


def label_names(schema_metadata):
    """ClassLabel names stored in the parquet metadata by Hugging Face, or None if absent."""
    try:
        info = json.loads(schema_metadata[b"huggingface"])
        return info["info"]["features"]["label"]["names"]
    except (KeyError, TypeError, ValueError):
        return None


def check_label_order(names):
    """Index 0 must be tench and index 1 goldfish. Names are WordNet-style ("tench, Tinca tinca")."""
    heads = [name.split(",")[0].strip().lower() for name in names[:2]]
    if len(names) != CLASSES or heads != ["tench", "goldfish"]:
        raise RuntimeError("label names do not match the ImageNet class order (index 0 tench, index 1 goldfish)")


def read_validation(root):
    """Return (shard paths, row offset of each shard, labels in dataset row order)."""
    import pyarrow.parquet as pq
    shards = sorted(Path(root, "data").glob("validation-*-of-*.parquet"))
    if len(shards) != VALIDATION_SHARDS:
        raise RuntimeError(f"expected {VALIDATION_SHARDS} validation shards, found {len(shards)}")
    offsets, labels, order_checked = [], [], False
    for path in shards:
        offsets.append(len(labels))
        if not order_checked:
            names = label_names(pq.read_schema(path).metadata or {})
            if names is None:
                print("warning: no ClassLabel names in the parquet metadata; class order not verified",
                      file=sys.stderr)
            else:
                check_label_order(names)
            order_checked = True
        labels.extend(pq.read_table(path, columns=["label"]).column("label").to_pylist())
    if len(labels) != VALIDATION_ROWS:
        raise RuntimeError(f"validation split has {len(labels)} rows, expected {VALIDATION_ROWS}")
    counts = [0] * CLASSES
    for label in labels:
        if type(label) is not int or not 0 <= label < CLASSES:
            raise RuntimeError("label outside 0..999")
        counts[label] += 1
    if any(count != PER_CLASS for count in counts):
        raise RuntimeError("validation split is not 50 images per class")
    return shards, offsets, labels


def decode_rows(shards, offsets, indices):
    """uint8 pixels for the chosen dataset rows; each shard is read once, only the image column."""
    import pyarrow.parquet as pq
    from PIL import Image
    by_shard = {}
    for index in indices:
        shard = bisect.bisect_right(offsets, index) - 1
        by_shard.setdefault(shard, []).append((index, index - offsets[shard]))
    pixels = {}
    for shard in sorted(by_shard):
        table = pq.read_table(shards[shard], columns=["image"])
        column = table.column("image")
        for index, local in by_shard[shard]:
            cell = column[local].as_py()
            pixels[index] = preprocess_image(Image.open(io.BytesIO(cell["bytes"])))
    return pixels


def load_corpus(path):
    """A corpus file, parsed. Only schema version 1 is understood."""
    corpus = json.loads(Path(path).read_bytes())
    if corpus.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported corpus schema")
    return corpus


def excluded_images(corpora):
    """pixels_sha256 -> set of labels, for every ImageNet case (dev or held-out) of the corpora, plus one
    record per distinct corpus. Synthetic cases are not validation rows and are not excluded."""
    images, records, seen = {}, [], set()
    for corpus in corpora:
        identity_value = corpus_identity(corpus)
        count = 0
        for case in corpus["cases"]:
            if case["source"] == SYNTHETIC_SOURCE:
                continue
            count += 1
            images.setdefault(case["pixels_sha256"], set()).add(case["label"])
        if identity_value not in seen:
            seen.add(identity_value)
            records.append({"identity": identity_value, "imagenet_cases": count})
    return images, records


def shard_pixel_hashes(path):
    """pixels_sha256 of every row of one validation shard, in row order (runs in a worker process)."""
    import pyarrow.parquet as pq
    from PIL import Image
    cells = pq.read_table(path, columns=["image"]).column("image").to_pylist()
    return [pixels_sha256(preprocess_image(Image.open(io.BytesIO(cell["bytes"])))) for cell in cells]


def row_pixel_hashes(shards, counts, workers):
    """pixels_sha256 of every validation row, in dataset row order. Shards decode in parallel."""
    parts = [None] * len(shards)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(shard_pixel_hashes, path): position for position, path in enumerate(shards)}
        for done, future in enumerate(as_completed(futures), start=1):
            position = futures[future]
            parts[position] = future.result()
            if len(parts[position]) != counts[position]:
                raise RuntimeError(f"shard {position} has {len(parts[position])} rows, labels say {counts[position]}")
            print(f"hashed shard {position} ({done}/{len(shards)})", file=sys.stderr, flush=True)
    return [digest for part in parts for digest in part]


MAX_LABEL_DISAGREEMENT = 0.01  # share of matched rows whose label is not one their image was recorded with


def excluded_rows(row_hashes, row_labels, images):
    """Row indices whose preprocessed pixels are one of the excluded images, and how many of those rows
    carry a label their image was not recorded with.

    Aborts unless every excluded image is found. A few disagreements are expected: ImageNet validation
    holds some files twice under two labels, and such a row is excluded with its twin, since the pixels
    are the same image. Widespread disagreement means row order and labels are misaligned, so it aborts.
    A preprocessing drift would leave images unfound, which also aborts.
    """
    matched, found, disagreeing = set(), set(), 0
    for index, digest in enumerate(row_hashes):
        labels_for = images.get(digest)
        if labels_for is None:
            continue
        if row_labels[index] not in labels_for:
            disagreeing += 1
        matched.add(index)
        found.add(digest)
    missing = len(images) - len(found)
    if missing:
        raise RuntimeError(f"exclusion check failed: {missing} excluded images not found in the validation split")
    if disagreeing > MAX_LABEL_DISAGREEMENT * len(matched):
        raise RuntimeError(f"exclusion check failed: {disagreeing} of {len(matched)} matched rows carry another "
                           "label; row order and labels disagree")
    return matched, disagreeing


def reused_cases(corpus):
    """The dev and synthetic cases of an earlier corpus, to copy byte for byte. The dataset revision is
    checked by build()."""
    dev_names = [f"dev-{position:02d}" for position in range(DEV_COUNT)]
    if corpus["selection"]["dev"] != dev_names or corpus["selection"]["synthetic"] != list(SYNTHETIC):
        raise ValueError("reused corpus does not hold the expected dev and synthetic cases")
    by_name = {case["name"]: case for case in corpus["cases"]}
    try:
        return [by_name[name] for name in dev_names], [by_name[name] for name in SYNTHETIC]
    except KeyError as missing:
        raise ValueError(f"reused corpus lacks case {missing}") from None


def build(out, seed, cache_dir, revision, exclude=(), reuse_dev_from=None, workers=None):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN must be set: ImageNet-1k is a gated dataset")
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=DATASET, repo_type="dataset", revision=revision,
                             allow_patterns=["data/validation-*.parquet"], cache_dir=cache_dir, token=token)
    shards, offsets, labels = read_validation(root)
    reuse = load_corpus(reuse_dev_from) if reuse_dev_from is not None else None
    if reuse is not None and (reuse["sources"].get("dataset") != DATASET
                              or reuse["sources"].get("dataset_revision") != revision):
        raise ValueError("reused corpus was built from a different dataset or revision")
    earlier = [load_corpus(path) for path in exclude] + ([reuse] if reuse is not None else [])
    images, extra_sources, excluded, disagreeing = {}, None, frozenset(), 0
    if earlier:
        images, records = excluded_images(earlier)
        counts = [end - start for start, end in zip(offsets, offsets[1:] + [len(labels)])]
        row_hashes = row_pixel_hashes(shards, counts, workers or os.cpu_count() or 1)
        matched, disagreeing = excluded_rows(row_hashes, labels, images)
        excluded = frozenset(matched)
        extra_sources = {"corpus_version": CORPUS_VERSION, "excluded_corpora": records,
                         "excluded_rows": len(excluded), "excluded_rows_with_other_label": disagreeing}
        print(f"excluded {len(images)} images as {len(excluded)} validation rows "
              f"({disagreeing} with another label)", file=sys.stderr, flush=True)
    dev_index, heldout_index = select_pools(labels, seed, excluded, reuse_dev=reuse is not None)
    if set(heldout_index) & excluded:
        raise RuntimeError("held-out pool overlaps the excluded rows")
    if reuse is not None:
        dev_entries, synthetic_entries = reused_cases(reuse)
        extra_sources["reused_from"] = {"identity": corpus_identity(reuse),
                                        "cases": "dev-* and syn-* copied byte for byte"}
        pixels = decode_rows(shards, offsets, heldout_index)
    else:
        pixels = decode_rows(shards, offsets, dev_index + heldout_index)
        dev_entries = [case_entry(f"dev-{position:02d}", pixels[index], labels[index], IMAGENET_SOURCE)
                       for position, index in enumerate(dev_index)]
        synthetic_entries = [case_entry(name, synthetic_pixels(name, seed), None, SYNTHETIC_SOURCE)
                             for name in SYNTHETIC]
    heldout_entries = [case_entry(f"ho-{position:04d}", pixels[index], labels[index], IMAGENET_SOURCE)
                       for position, index in enumerate(heldout_index)]
    corpus = assemble_corpus(dev_entries, heldout_entries, synthetic_entries, seed, revision, extra_sources)
    Path(out).write_bytes(canonical_bytes(corpus))
    summary = {"corpus": str(out), "identity": corpus_identity(corpus),
               "dev": len(dev_entries), "heldout": len(heldout_entries),
               "heldout_classes": len({e["label"] for e in heldout_entries}),
               "synthetic": len(synthetic_entries)}
    if earlier:
        summary.update(excluded_images=len(images), excluded_rows=len(excluded), excluded_rows_with_other_label=disagreeing)
    print(json.dumps(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    seed_source = parser.add_mutually_exclusive_group(required=True)
    seed_source.add_argument("--seed", type=int)
    seed_source.add_argument("--seed-file", type=Path, help="file holding the integer seed")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache directory (default: the hub default)")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--exclude-corpus", action="append", default=[], type=Path, metavar="PATH",
                        help="keep every ImageNet image of this corpus out of the pools (repeatable)")
    parser.add_argument("--reuse-dev-from", type=Path, default=None, metavar="PATH",
                        help="copy the dev and synthetic cases of this corpus byte for byte; its images are excluded")
    parser.add_argument("--workers", type=int, default=None,
                        help="processes that hash validation rows when excluding (default: CPU count)")
    args = parser.parse_args(argv)
    seed = args.seed if args.seed is not None else int(args.seed_file.read_text().strip())
    build(args.out, seed, args.cache_dir, args.dataset_revision, args.exclude_corpus,
          args.reuse_dev_from, args.workers)


if __name__ == "__main__":
    sys.exit(main())
