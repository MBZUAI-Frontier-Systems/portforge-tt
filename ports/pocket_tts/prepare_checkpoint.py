#!/usr/bin/env python3
"""Build the flat pinned checkpoint directory for the pocket_tts port pack.

The kyutai/pocket-tts repository is gated and stores this recipe under
languages/english_2026-09_2/ with subdirectories the models registry cannot
express, so the harness --download shortcut does not apply. This script
downloads the pinned subpaths (requires an authenticated Hugging Face token
from an account that accepted the repository terms), flattens them beside the
authored config files and verifies every byte against ports/pocket_tts/models.json.

Usage (from the repository root or the pack's frozen harness directory):

    python prepare_checkpoint.py /absolute/path/to/checkpoints/pocket_tts

Point the cluster file's porting.tt.weights_by_model entry for pocket_tts at
the same directory.
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import evaluate  # noqa: E402


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1])
    receipt = evaluate.prepare_checkpoint(target)
    print(receipt)


if __name__ == "__main__":
    main()
