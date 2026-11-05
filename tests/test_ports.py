"""Every kernel pack loads and freezes per family, TT packs are refused, and the tree is free of site detail."""
import json
from pathlib import Path
import re
import tempfile
import unittest

import support  # noqa: F401  (repository and pack setup)
from portforge_tt.control import port
from portforge_tt.ranking import search

PACKS = sorted(path for path in (support.ROOT / "ports").iterdir() if (path / "port.json").is_file())
KERNEL_PACKS = [path for path in PACKS if json.loads((path / "port.json").read_text()).get("kind") == "kernel"]
REAL = [path for path in KERNEL_PACKS if path.name != "template-kernel"]
# Archived receipts and scripts of the manual campaigns (October 2026), kept in place on request.
# They predate the site-detail rule; any other file with site detail fails.
LEGACY = {"ports/dav1d-rvv/kernels/itx16-wide/cmp_bench.py", "ports/dav1d-rvv/kernels/itx16-wide/REPORT.md",
          "ports/dav1d-rvv/kernels/itx16-wide/analyze_survey.py", "ports/xnnpack-rvv/kernels/qs8-vcvt/bench_table.py",
          "ports/xnnpack-rvv/kernels/qs8-vcvt/submission-submission-20261019T134340Z/RECEIPT.json",
          "ports/xnnpack-rvv/kernels/qs8-vmul-minmax-rndnu/submission-20261020T094536Z/RECEIPT.json",
          "ports/xnnpack-rvv/kernels/qs8-vlrelu/submission-20261019T182939Z/RECEIPT.json"}


def campaign_families(pack):
    """Families a campaign can target: those with a technique menu."""
    families = json.loads((pack / "families.json").read_text())["families"]
    return [f for f in sorted(families) if (pack / "kernels" / f / "TECHNIQUES.json").is_file()]


class PortPackTests(unittest.TestCase):
    def tearDown(self):
        support.configure_port()

    def test_every_campaign_family_loads_and_freezes(self):
        self.assertEqual({p.name for p in REAL}, {"dav1d-rvv", "xnnpack-rvv"})
        for pack in REAL:
            families = campaign_families(pack)
            self.assertTrue(families, pack.name)
            for family in families:
                with self.subTest(pack=pack.name, family=family), tempfile.TemporaryDirectory() as directory:
                    loaded = port.configure(pack, family)
                    for doc in ("CONTRACT.md", "TASK.md"):
                        self.assertIn(doc, loaded["docs"])
                    self.assertIn(family, loaded["title"])
                    self.assertIn(loaded["source_name"], port.SOURCE_NAMES)
                    self.assertTrue(set(loaded["verify"]) <= {"cbmc", "cvc5"})
                    documents = port.seed_documents(loaded, port.PACKAGE / "guides", search.known(loaded))
                    self.assertTrue(set(port.GUIDES + port.FAMILY_DOCS) <= set(documents))
                    self.assertNotIn('"heldout"', documents["SPEC.md"])
                    harness = Path(directory) / "harness"
                    port.freeze_harness(harness)
                    frozen = port.load_port(harness)
                    self.assertEqual((frozen["key"], frozen["family"]), (loaded["key"], family))
                    kernels = {p.name for p in (harness / "ports" / pack.name / "kernels").iterdir()}
                    self.assertEqual(kernels, {family})
                    files = port.host_files(harness)
                    self.assertIn("portforge_tt/remote/runner.py", files)
                    self.assertIn(f"ports/{pack.name}/evaluate.py", files)
                    self.assertFalse(any(name.startswith(("portforge_tt/control/", "portforge_tt/agents/"))
                                         for name in files))

    def test_tt_model_packs_are_refused_with_a_pointer(self):
        for pack in PACKS:
            if pack in KERNEL_PACKS:
                continue
            with self.subTest(pack=pack.name), self.assertRaisesRegex(ValueError, "tt branch"):
                port.load_port(pack)

    def test_technique_menus_name_catalogue_or_local_ids(self):
        known = search.catalogue()
        for pack in REAL:
            for family in campaign_families(pack):
                doc = json.loads((pack / "kernels" / family / "TECHNIQUES.json").read_text())
                with self.subTest(family=family):
                    self.assertTrue(doc.get("spec"))
                    self.assertTrue(all(c["id"] in known or c["id"].startswith("k:") for c in doc["candidates"]))

    def test_agent_documents_use_the_shared_tool_names(self):
        stale = re.compile(r"\bkernel_(?:files|read|write|compile|check)\b|python -m portforge_tt\.kernel")
        paths = [p for pack in KERNEL_PACKS for p in pack.glob("*.md")] + list((support.HARNESS / "guides").glob("*.md")) \
            + list((support.HARNESS / "prompts").glob("*.md"))
        for path in paths:
            with self.subTest(path=str(path.relative_to(support.ROOT))):
                self.assertIsNone(stale.search(path.read_text()), path)

    def test_tree_has_no_site_specific_hosts_users_or_paths(self):
        pattern = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b|/home/(?!<)[a-z]|/Users/[A-Za-z]|abror|murodbe|id_tt_box|"
                             r"google_compute|tt-blackhole-\d")
        # RFC 5737 documentation ranges are the intended placeholders in test fixtures.
        documentation = ("192.0.2.", "198.51.100.", "203.0.113.")
        # tt-metal is a vendored upstream checkout and paper/ is the manuscript;
        # this check governs portforge's own tree.
        excluded = {".git", ".claude", ".venv", "venv", "runs", "assets", "dist", "build",
                    "__pycache__", "portforge_tt.egg-info", "tt-metal", "paper"}
        skipped_names = {"corpus.json", "README.md", "portforge-cpu.json"}
        for path in support.ROOT.rglob("*"):
            if not path.is_file() or path.suffix not in (".py", ".md", ".json", ".yaml", ".sh"):
                continue
            relative = path.relative_to(support.ROOT).as_posix()
            if any(part in excluded for part in path.parts) or relative in LEGACY:
                continue
            if path.name in skipped_names or path == Path(__file__):
                continue
            for match in pattern.finditer(path.read_text(errors="replace")):
                if match.group(0) == "127.0.0.1" or match.group(0).startswith(documentation):
                    continue
                with self.subTest(path=relative, match=match.group(0)):
                    self.fail(f"site-specific detail {match.group(0)!r} in {relative}")


if __name__ == "__main__":
    unittest.main()
