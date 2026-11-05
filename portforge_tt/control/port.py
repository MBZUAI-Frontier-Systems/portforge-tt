"""Kernel packs: the project-specific half of a PortForge campaign.

A pack is a directory (see ports/template-kernel/) with:

  port.json      manifest validated here ("kind": "kernel")
  families.json  the kernel families: signature, upstream variants, verify methods
  evaluate.py    trusted adapter run on the x86 host (remote/runner.py calls it)
  CONTRACT.md    what every candidate must satisfy
  TASK.md        ordered engineering plan for the worker
  SOURCES.md     optional public sources (papers, upstream PRs, prior ports)
  kernels/<family>/{regimes,TECHNIQUES}.json  the family's matrix and technique menu

A campaign targets one family. Everything else (controller, agents, gates,
verification) is shared. When a campaign starts, the shared code and the pack
are frozen into the run's harness directory, so a run always replays the exact
gates it was graded by; the same harness is uploaded to the x86 host.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil

PACK_FILES = ("port.json", "families.json", "evaluate.py", "CONTRACT.md", "TASK.md")
OPTIONAL_DOCS = ("SOURCES.md",)
GUIDES = ("RVV_GUIDE.md", "RVV_TECHNIQUES.md", "LESSONS.md")
FAMILY_DOCS = ("SPEC.md", "TECHNIQUES.md")
CONTEXT_DIR = "context"
SOURCE_NAMES = ("kernel.c", "kernel.S")

DEFAULT_OBJECTIVE = (
    "Deliver ONE {family} kernel for RVV that is bit-exact with the project's reference at every VLEN from 128 "
    "to 1024 and as fast as possible on the SiFive X280: the geomean speedup over the family's regime x shape "
    "matrix against {baseline}, with no cell slower beyond the noise floor. Port first (a correct vector "
    "version), then optimize with measured perf receipts, trying the techniques in TECHNIQUES.md and "
    "RVV_TECHNIQUES.md and recording each in TRANSFORMS.json; submit the best source for certification.")


def load_port(directory, family=None):
    """Validate a kernel pack (or a frozen harness holding one) and return its resolved manifest.

    A frozen harness has campaign.json naming its pack and family; a pack directory needs family=.
    """
    root = Path(directory).resolve()
    if (root / "campaign.json").is_file():
        frozen = json.loads((root / "campaign.json").read_text())
        root, family = root / frozen["pack"], family or frozen["family"]
    missing = [name for name in PACK_FILES if not (root / name).is_file()]
    if not (root / "port.json").is_file():
        raise ValueError(f"{root} is not a port pack; missing port.json")
    manifest = json.loads((root / "port.json").read_text())
    if manifest.get("kind") != "kernel":
        raise ValueError(f"{root.name} is a TT model pack; this branch runs kernel packs (TT packs run from the tt branch)")
    if missing:
        raise ValueError(f"{root} is not a kernel pack; missing {', '.join(missing)}")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported port.json schema")
    if not isinstance(manifest.get("optimization", {}), dict):
        raise ValueError("port.json optimization must be an object (see ranking/objective.py)")
    families = json.loads((root / "families.json").read_text())["families"]
    key = manifest.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError("port.json key is required")
    port = {"dir": root, "key": key, "pack_title": manifest.get("title") or key, "task": manifest.get("task", ""),
            "repo_id": (manifest.get("upstream") or {}).get("repo", key), "families": sorted(families),
            "docs": [name for name in PACK_FILES + OPTIONAL_DOCS if name.endswith(".md") and (root / name).is_file()],
            "optimization": manifest.get("optimization") or {}, "family": None, "title": manifest.get("title") or key}
    if family is None:
        return port
    if family not in families:
        raise ValueError(f"unknown family {family!r}; {key} has: {', '.join(sorted(families))}")
    folder = root / "kernels" / family
    if not (folder / "TECHNIQUES.json").is_file():
        raise ValueError(f"{folder}/TECHNIQUES.json is required (the family's technique menu)")
    entry = families[family]
    source = entry.get("source_name", "kernel.c")
    if source not in SOURCE_NAMES:
        raise ValueError("source_name must be one of " + ", ".join(SOURCE_NAMES))
    regimes = json.loads((folder / "regimes.json").read_text()) if (folder / "regimes.json").is_file() else {}
    baseline = entry.get("baseline", "the fastest correct upstream RVV kernel per cell")
    return {**port, "family": family, "entry": entry, "source_name": source, "signature": entry.get("signature"),
            "title": entry.get("project", key.removesuffix("-rvv")) + " " + family, "baseline": baseline,
            "noise_floor": float(regimes.get("noise_floor", 0.005)),
            "verify": list(entry.get("verify", [])),
            "objective": manifest.get("objective") or DEFAULT_OBJECTIVE.format(family=family, baseline=baseline)}


def family_techniques(port):
    """The family's TECHNIQUES.json (spec, candidates, mined_from)."""
    return json.loads((port["dir"] / "kernels" / port["family"] / "TECHNIQUES.json").read_text())


def family_regimes(port):
    """The family's regimes.json without the held-out sampler (agents never see it)."""
    path = port["dir"] / "kernels" / port["family"] / "regimes.json"
    return {k: v for k, v in json.loads(path.read_text()).items() if k != "heldout"} if path.is_file() else {}


def render_techniques(port, catalogue):
    """TECHNIQUES.md: every technique mined for this family, with the catalogue entry it refers to."""
    doc = family_techniques(port)
    lines = [f"# Techniques for {port['family']}", "",
             "Mined for this family from every other ISA's implementation, upstream history and the RVV "
             "catalogue. All are optional ideas: none is known to win here until a perf receipt says so, and "
             "one that lost once may win under another parent or in combination. Ids without `k:` are "
             "catalogue entries (full text in RVV_TECHNIQUES.md); `k:` ids are local to this family. Record "
             "the ids you apply in TRANSFORMS.json.", ""]
    for item in doc.get("candidates", []):
        known = catalogue.get(item["id"], {})
        title = item.get("title") or known.get("title", "")
        lines.append(f"## `{item['id']}`" + (f": {title}" if title else ""))
        for field, label in (("application", "Here"), ("expected_gain", "Why it might win"),
                             ("preconditions", "Preconditions")):
            value = item.get(field)
            if value:
                lines.append(f"- **{label}.** " + ("; ".join(value) if isinstance(value, list) else str(value)))
        lines.append("")
    if doc.get("mined_from"):
        lines += ["## Mined from", "", json.dumps(doc["mined_from"], indent=1), ""]
    return "\n".join(lines)


def render_spec(port, context_names=()):
    """SPEC.md: the exact formula and domain, the check and benchmark matrix, and an index of context/."""
    doc = family_techniques(port)
    text = ("# Spec\n\n```json\n" + json.dumps(doc.get("spec", {}), indent=1) + "\n```\n\n"
            "# Benchmark and check matrix (cells are regime@shape)\n\n```json\n"
            + json.dumps(family_regimes(port), indent=1) + "\n```\n\n"
            f"Speedup is measured against {port['baseline']}.\n")
    if context_names:
        text += "\n# Reference and other implementations (context/)\n\n" + "".join(
            f"- context/{name}\n" for name in context_names)
    return text


def seed_documents(port, guides_dir, catalogue):
    """Agent-facing instruction files for a fresh workspace: pack docs, shared guides and family docs."""
    documents = {name: (port["dir"] / name).read_text() for name in port["docs"]}
    for name in GUIDES:
        documents[name] = (Path(guides_dir) / name).read_text()
    documents["SPEC.md"] = render_spec(port)
    documents["TECHNIQUES.md"] = render_techniques(port, catalogue)
    return documents


def freeze(port, harness):
    """Copy the selected pack (its files and this family's kernels/ folder) into a run's harness."""
    harness = Path(harness)
    target = harness / "ports" / port["dir"].name
    for path in sorted(port["dir"].rglob("*")):
        rel = path.relative_to(port["dir"])
        if not path.is_file() or "__pycache__" in rel.parts or path.name.startswith("."):
            continue
        if rel.parts[0] == "kernels" and (len(rel.parts) < 3 or rel.parts[1] != port["family"]):
            continue  # other families' material stays out of this campaign
        if rel.parts[0] == "tests":
            continue
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / rel)
    (harness / "campaign.json").write_text(json.dumps(
        {"pack": "ports/" + port["dir"].name, "family": port["family"]}, indent=1) + "\n")


# The selected pack. A frozen run harness carries its pack and family (campaign.json beside the
# package), so the controller and every job load it at import; campaign.start() selects one first.
HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent  # portforge_tt/ (guides/, prompts/ and the code directories)
CODE_DIRS = ("agents", "control", "ranking", "remote", "verify")
PORT = load_port(PACKAGE.parent) if (PACKAGE.parent / "campaign.json").is_file() else {}


def configure(directory, family):
    """Select the pack every controller-side module sees (PORT is shared and updated in place)."""
    PORT.clear()
    PORT.update(load_port(directory, family))
    return PORT


def seed_policy():
    return 'fresh-public-' + PORT['key'] + '/' + PORT['family']


def trusted_documents():
    """Instruction files agents may read but never overwrite (context/ is read-only as a whole)."""
    return tuple(PORT['docs']) + GUIDES + FAMILY_DOCS


def is_trusted(path):
    path = Path(path).as_posix()
    return path in trusted_documents() or path.split('/', 1)[0] == CONTEXT_DIR


def gated_stages():
    """Stages an agent may evaluate: check (G0-G2 under QEMU) and perf (check, then the X280 bench)."""
    return ['check', 'perf']


def harness_identity(folder):
    """Hash of every frozen source, prompt, guide, tool and pack file in a run's harness."""
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(folder.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def frozen_environment(run, manifest):
    frozen = Path(run).resolve() / 'harness'
    if harness_identity(frozen) != manifest['harness_sha256']:
        raise ValueError('frozen harness identity mismatch')
    repository = Path(manifest['repository_root'])
    if not repository.is_absolute() or repository == Path('/'):
        raise ValueError('invalid repository isolation root')
    return {**os.environ, 'PYTHONPATH': str(frozen), 'PORTFORGE_REPOSITORY_ROOT': str(repository)}


def agent_environment(manifest):
    """Role backend/model selection recorded in the manifest."""
    backends = manifest.get('agent_backends', {})
    models = manifest.get('agent_models', {})
    environment = {}
    for role in ('worker', 'supervisor'):
        environment['PORTFORGE_' + role.upper() + '_BACKEND'] = backends.get(role, 'claude')
        if models.get(role):
            environment['PORTFORGE_' + role.upper() + '_MODEL'] = models[role]
    return environment


def freeze_harness(harness):
    """Copy the shared package (code, prompts, guides) and the selected pack into a run's harness.

    The harness is a package root: the controller runs it as ``python -m portforge_tt.control.controller``
    with PYTHONPATH=harness, and the x86 host runs ``python3 -m portforge_tt.remote.runner`` from its copy.
    """
    harness = Path(harness)
    harness.mkdir()
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    target = harness / 'portforge_tt'
    target.mkdir()
    shutil.copyfile(PACKAGE / '__init__.py', target / '__init__.py')
    for folder in CODE_DIRS + ('prompts', 'guides'):
        shutil.copytree(PACKAGE / folder, target / folder, ignore=ignore)
    freeze(PORT, harness)


def host_files(harness):
    """What the x86 host needs from a harness: the package minus the controller side, plus the pack."""
    harness = Path(harness)
    keep = ('portforge_tt/__init__.py', 'portforge_tt/remote/', 'portforge_tt/verify/', 'ports/', 'campaign.json')
    return {p.relative_to(harness).as_posix(): p.read_bytes() for p in sorted(harness.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts
            and p.relative_to(harness).as_posix().startswith(keep)}
