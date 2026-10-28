"""Static upstream-readiness check of a deliverable tt-metal-style package.

Derived from the review of the hackathon PRs to tenstorrent/tt-metal
(models/experimental/{nllb,esm2,parakeet}): reviewers asked for real pytest
tests with asserts, the repository ``device`` fixture instead of private
open_device calls, no hard-coded sandbox paths, an end-to-end test with the real
checkpoint, a demo, and performance numbers in the README in the style of
models/model_targets.yaml. This is a static gate on files, not a substitute for
running the tests on tt-metal CI.
"""
import re
from pathlib import PurePosixPath

PACKAGE_DIRS = ('tt', 'tests', 'demo')
SANDBOX_PATHS = re.compile(r'''["'](/weights|/input|/output|/work)(/[^"']*)?["']''')
PERF_HEADER = re.compile(r'^\|.*\b(latency|throughput|t/s|tokens/s|fps|rtf|ms|samples/s|sentences/s|series/s)\b.*\|',
                         re.IGNORECASE | re.MULTILINE)
TEST_DEF = re.compile(r'^\s*def (test_\w+)\(([^)]*)\)', re.MULTILINE)


def package_root(files):
    """Directory holding tt/ and tests/ (e.g. models/experimental/<name>), or None."""
    roots = {}
    for name in files:
        parts = PurePosixPath(name).parts
        for i, part in enumerate(parts[:-1]):
            if part in PACKAGE_DIRS:
                roots.setdefault(PurePosixPath(*parts[:i]) if i else PurePosixPath('.'), set()).add(part)
    complete = [root for root, found in roots.items() if {'tt', 'tests'} <= found]
    return min(complete, key=lambda p: len(p.parts)) if complete else None


def check(files):
    """files: {relative path: text}. Returns {'passed', 'issues', 'package_root', 'checked'}."""
    issues, root = [], package_root(files)
    if root is None:
        return dict(passed=False, package_root=None, checked={},
                    issues=['no tt-metal-style package: need a directory with tt/ and tests/ (plus demo/, README.md)'])
    prefix = '' if str(root) == '.' else str(root) + '/'
    inside = {name[len(prefix):]: text for name, text in files.items() if name.startswith(prefix)}
    tests = {n: t for n, t in inside.items() if n.startswith('tests/') and PurePosixPath(n).name.startswith('test_')
             and n.endswith('.py')}
    python = {n: t for n, t in inside.items() if n.endswith('.py')}
    definitions = {n: TEST_DEF.findall(t) for n, t in tests.items()}
    if not any(definitions.values()):
        issues.append('tests/ has no pytest test functions (def test_...)')
    for name, text in tests.items():
        if definitions[name] and 'assert' not in text:
            issues.append(name + ': test functions never assert')
        if re.search(r'^\s*(from|import)\s+backend\b', text, re.MULTILINE):
            issues.append(name + ': imports the sandbox-only "backend" module')
        if 'open_device(' in text:
            issues.append(name + ': opens its own device; use the repository "device"/"mesh_device" fixture')
        if definitions[name] and not any(re.search(r'\b(device|mesh_device)\b', args) for _, args in definitions[name]):
            issues.append(name + ': no test takes the device fixture')
    if not any(re.search(r'e2e|end_to_end|demo', n) for n in tests):
        issues.append('no end-to-end test (tests/*e2e* or tests/*demo*) with the real checkpoint')
    for name, text in python.items():
        if SANDBOX_PATHS.search(text):
            issues.append(name + ': hard-coded sandbox path ' + SANDBOX_PATHS.search(text).group(1))
        if 'SPDX-License-Identifier' not in '\n'.join(text.splitlines()[:5]):
            issues.append(name + ': missing SPDX header')
    if not any(n.startswith('demo/') and n.endswith('.py') for n in inside):
        issues.append('no demo/ script')
    readme = inside.get('README.md', '')
    if not readme:
        issues.append('no README.md')
    elif not PERF_HEADER.search(readme):
        issues.append('README.md has no performance table (latency/throughput/fps/rtf/t/s with batch and shape)')
    return dict(passed=not issues, package_root=str(root), issues=issues[:40],
                checked=dict(tests=len(tests), test_functions=sum(len(v) for v in definitions.values()),
                             python_files=len(python)))
