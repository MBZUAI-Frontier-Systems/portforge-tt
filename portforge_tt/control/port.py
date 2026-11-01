"""Port packs: the model-specific half of a PortForge campaign.

A pack is a directory (see ports/template/) with:

  port.json     manifest validated here
  models.json   pinned checkpoint registry read by portforge_models.py
  evaluate.py   trusted evaluator: make_suite, reference, candidate, assess
  CONTRACT.md   interface and acceptance envelope the agents must meet
  TASK.md       ordered engineering plan for the worker
  BENCHMARK.md  optional corpus provenance and gate rationale
  SOURCES.md    optional model-specific public sources (papers, reference code, prior ports)
  corpus.json   optional frozen benchmark (otherwise pass --corpus)

Everything else (controller, agents, TT runner, reference jobs, guides) is shared.
When a campaign starts, the pack is frozen into the run's harness directory next
to the shared code, so a run always replays the exact evaluator it was graded by.
"""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

if __package__:
    from ..remote.portforge_eval import TT_PRECISIONS
else:
    from portforge_eval import TT_PRECISIONS

PACK_FILES = ("port.json", "models.json", "evaluate.py", "CONTRACT.md", "TASK.md")
OPTIONAL_DOCS = ("BENCHMARK.md", "SOURCES.md")
GUIDES = ("PORTING_RULES.md", "TT_GUIDE.md", "LESSONS.md", "TT_OPTIMIZATIONS.md", "CORRECTNESS_PITFALLS.md", "UPSTREAM_PR.md")

DEFAULT_OBJECTIVE = (
    "Deliver ONE production-level reusable {title} TTNN source packaged in tt-metal style (tt/, demo/, "
    "reference/, tests/, benchmarks/, docs/ with SPDX headers) suitable for an upstream tt-metal PR. Establish "
    "measured baselines FIRST and preserve them (PyTorch on the TT host; CUDA reference timings are the NVIDIA "
    "comparison context); pass smoke, then bringup against the FP32 oracle; profile and optimize against the "
    "recorded baseline with matched same-precision comparisons; assemble the tt-metal-style package and "
    "portable tests; then freeze for full validation. A first bringup pass alone is not the goal, and "
    "optimization claims without a recorded baseline are invalid. Advance independent implementation when "
    "runtime diagnosis blocks final evaluation. Follow LESSONS.md.")
DEFAULT_PRECISION_POLICY = (
    "BF16 primary; investigate supported BFP8_B with honest quality, latency and memory tradeoffs. Missing "
    "cells are not support or passes; BFP8_B is block float, not IEEE FP8. Do not inherit CUDA precision results.")


def _strings(value, field):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError(f"port.json {field} must be a nonempty list of strings")
    return list(value)


def load_port(directory):
    """Validate a pack (or a frozen harness holding one) and return its resolved manifest."""
    root = Path(directory).resolve()
    missing = [name for name in PACK_FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"{root} is not a port pack; missing {', '.join(missing)}")
    manifest = json.loads((root / "port.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported port.json schema")
    registry = json.loads((root / "models.json").read_text())["models"]
    key = manifest.get("key")
    delivery = _strings(manifest.get("delivery_models", [key]), "delivery_models")
    if key not in registry or key not in delivery or any(name not in registry for name in delivery):
        raise ValueError("port.json key must be listed in delivery_models, and every delivery model must be a models.json key")
    precisions = _strings(manifest.get("precisions", list(TT_PRECISIONS)), "precisions")
    if not isinstance(manifest.get("optimization", {}), dict):
        raise ValueError("port.json optimization must be an object (see objective.py)")
    delivery_precisions = _strings(manifest.get("delivery_precisions", ["bf16"]), "delivery_precisions")
    if any(p not in TT_PRECISIONS for p in precisions) or any(p not in precisions for p in delivery_precisions):
        raise ValueError("precisions must be TT formats and include every delivery precision")
    corpus = manifest.get("corpus") or {}
    digest = corpus.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("port.json corpus.sha256 must pin the frozen benchmark")
    corpus_file = corpus.get("file")
    if corpus_file is not None and (not isinstance(corpus_file, str) or Path(corpus_file).name != corpus_file):
        raise ValueError("corpus.file must be a file name inside the pack")
    mesh_cards = manifest.get("mesh_cards")
    if mesh_cards is not None and (type(mesh_cards) is not int or not 2 <= mesh_cards <= 8):
        raise ValueError("port.json mesh_cards (cards of the data-parallel mesh stage) must be an integer 2..8")
    title = manifest.get("title") or key
    reference = manifest.get("reference") or {}
    return {
        "dir": root, "key": key, "title": title, "task": manifest.get("task", ""),
        "repo_id": registry[key]["repo_id"], "delivery_models": delivery,
        "precisions": precisions, "delivery_precisions": delivery_precisions,
        "corpus_sha256": digest, "corpus_path": root / corpus_file if corpus_file else None,
        "objective": manifest.get("objective") or DEFAULT_OBJECTIVE.format(title=title),
        "precision_policy": manifest.get("precision_policy") or DEFAULT_PRECISION_POLICY,
        "docs": [name for name in PACK_FILES + OPTIONAL_DOCS if name.endswith(".md") and (root / name).is_file()],
        "optimization": manifest.get("optimization") or {},
        "mesh_cards": mesh_cards,
        "reference_packages": list(reference.get("packages", [])),
        "reference_imports": list(reference.get("imports", [])),
    }


def seed_documents(port, guides_dir):
    """Agent-facing instruction files for a fresh workspace: pack docs plus shared guides."""
    documents = {name: (port["dir"] / name).read_text() for name in port["docs"]}
    for name in GUIDES:
        documents[name] = (Path(guides_dir) / name).read_text()
    return documents


def freeze(port, harness):
    """Copy the pack into a run's harness directory, beside the shared code."""
    harness = Path(harness)
    for name in PACK_FILES + OPTIONAL_DOCS:
        if (port["dir"] / name).is_file():
            shutil.copyfile(port["dir"] / name, harness / name)


# The selected pack. A frozen run harness carries its pack (port.json beside this file), so the
# controller and every job load it at import; campaign.start() selects one before creating a run.
HERE = Path(__file__).resolve().parent
# portforge_tt/ (guides/, prompts/, agents/ and the code directories), or the flat frozen harness itself.
PACKAGE = HERE if (HERE / "guides").is_dir() else HERE.parent
CODE_DIRS = ("control", "ranking", "remote")
PORT = load_port(HERE) if (HERE / "port.json").is_file() else {}


def configure(directory):
    """Select the pack every controller-side module sees (PORT is shared and updated in place)."""
    PORT.clear()
    PORT.update(load_port(directory))
    return PORT


def seed_policy():
    return 'fresh-public-' + PORT['key']



def trusted_documents():
    """Instruction files agents may read but never overwrite."""
    return tuple(PORT['docs']) + GUIDES



def gated_stages():
    """Stages an agent may evaluate: the mesh stage only when the pack declares mesh_cards."""
    return ['smoke', 'bringup', 'perf'] + (['mesh'] if PORT.get('mesh_cards') else [])



def same_evaluator(run):
    """True when an earlier run's references came from the evaluator this harness ships."""
    frozen = Path(run) / 'harness' / 'evaluate.py'
    return frozen.is_file() and frozen.read_bytes() == (PORT['dir'] / 'evaluate.py').read_bytes()



def pack_evaluator():
    """The pack's evaluator: the importable ``evaluate`` module when present
    (the frozen harness or a pack on sys.path, so operator patches apply),
    else loaded by path so no other module can shadow it."""
    if 'evaluate' in sys.modules:
        return sys.modules['evaluate']
    try:
        return importlib.import_module('evaluate')
    except ImportError:
        pass
    return load_evaluator('portforge_pack_evaluate')


def load_evaluator(name):
    """Execute the pack's evaluate.py by path. Packs import the kit flat (as on remote hosts), so in
    package mode the flat names point at this package's modules and share the registry."""
    if __package__:
        from ..remote import portforge_eval, portforge_models
        sys.modules.setdefault('portforge_eval', portforge_eval)
        sys.modules.setdefault('portforge_models', portforge_models)
    spec = importlib.util.spec_from_file_location(name, PORT['dir'] / 'evaluate.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    return evaluator


def harness_identity(folder):
    """Hash of every frozen source, prompt, guide and pack file in a run's harness."""
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): p.read_text() for p in sorted(folder.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.md', '.json') and '__pycache__' not in p.parts}
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
    """Role backend/model selection recorded in the manifest (opencode for older runs)."""
    backends = manifest.get('agent_backends', {})
    models = manifest.get('agent_models', {})
    environment = {}
    for role in ('worker', 'supervisor'):
        environment['PORTFORGE_' + role.upper() + '_BACKEND'] = backends.get(role, 'opencode')
        if models.get(role):
            environment['PORTFORGE_' + role.upper() + '_MODEL'] = models[role]
    return environment



def freeze_harness(harness):
    """Copy the shared code (flat: every module beside the others), prompts, guides and the selected pack into a run."""
    harness.mkdir()
    for folder in CODE_DIRS:
        for path in (PACKAGE / folder).iterdir():
            if path.is_file() and path.suffix == '.py' and path.name != '__init__.py':
                shutil.copyfile(path, harness / path.name)
    # The agents package travels flat inside the harness (its own relative
    # imports keep working; only its parent-package imports switch).
    (harness / 'agents').mkdir()
    for path in (PACKAGE / 'agents').iterdir():
        if path.is_file() and path.suffix == '.py':
            shutil.copyfile(path, harness / 'agents' / path.name)
    for folder in ('prompts', 'guides'):
        shutil.copytree(PACKAGE / folder, harness / folder)
    freeze(PORT, harness)
