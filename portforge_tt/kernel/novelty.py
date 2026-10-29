"""Duplicate detection before evaluation (zero LLM cost, ShinkaEvolve-style).

Two levels:
  source_key  the C source with comments, whitespace and the function name
              normalized: a hit is the same kernel, so the node becomes an alias
              of the existing one and reuses its receipts;
  loop_key    the hot loops' opcode sequences plus the dispatch shape from the
              static profile: a hit compiled to the same machine code, so it
              gets correctness gates but no board time unless a plan insists.
"""
import hashlib
import json
import re

COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
KERNEL_NAME = re.compile(r"\b(?:xnn|test)_\w+(?=\s*\()")


def normalize(source):
    text = COMMENT.sub("", source)
    text = KERNEL_NAME.sub("KERNEL", text, count=1)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("#include"))
    return re.sub(r"[ \t]+", " ", text)


def source_key(source):
    return hashlib.sha256(normalize(source).encode()).hexdigest()


def loop_key(profile):
    """Hash of the hot loops' instruction class sequences and vsetvl configs."""
    loops = profile.get("loops", []) if profile else []
    shape = [(sorted(h["counts"].items()), h["vsetvl"]) for h in loops]
    return hashlib.sha256(json.dumps(shape, sort_keys=True).encode()).hexdigest()


class Index:
    def __init__(self):
        self.sources, self.loops = {}, {}

    def add(self, node_id, source=None, profile=None):
        if source is not None:
            self.sources.setdefault(source_key(source), node_id)
        if profile is not None:
            self.loops.setdefault(loop_key(profile), node_id)

    def same_source(self, source):
        return self.sources.get(source_key(source))

    def same_loops(self, profile):
        return self.loops.get(loop_key(profile))
