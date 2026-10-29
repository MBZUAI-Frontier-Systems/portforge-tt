"""Normalizing kernel sources before they enter a harness.

Kernels optimized outside XNNPACK often carry their own copies of XNNPACK's
param structs (`struct xnn_f32_minmax_params { ... };`, or a bare forward
declaration) and shims such as `#define XNN_OOB_READS`. Inside a harness that
already has the real definitions these clash: a redefinition error in XNNPACK,
or, in the verifier's namespace, a second distinct type. Only those file-scope
declarations are removed; kernel code is never touched, and every removal is
reported so a verdict can say what it was computed on.
"""
import re

XNN_STRUCT_HEAD = re.compile(r"^(?:typedef\s+)?(?:struct|union)\s+(xnn_\w+)\s*(;|\{)", re.M)
# Attribute shims (XNN_OOB_READS, XNN_UNPREDICTABLE, ...); configuration such as
# XNN_ARCH_ARM64 or XNN_ENABLE_* selects code paths and is kept.
XNN_MACRO_SHIM = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(XNN_(?!ARCH_|ENABLE_)\w+)\b[^\n]*\n", re.M)


def _block_end(text, brace):
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                semi = text.find(";", i)
                return semi + 1 if semi != -1 else i + 1
    return len(text)


def strip_local_xnn_types(text):
    """(text without file-scope xnn_* struct declarations and XNN_* macro shims, removed names)."""
    removed = []
    out, pos = [], 0
    for m in XNN_STRUCT_HEAD.finditer(text):
        if m.start() < pos:
            continue
        end = m.end() if m.group(2) == ";" else _block_end(text, m.end() - 1)
        out.append(text[pos:m.start()])
        removed.append(f"struct {m.group(1)}")
        pos = end
    text = "".join(out) + text[pos:]
    removed += [f"#define {m.group(1)}" for m in XNN_MACRO_SHIM.finditer(text)]
    text = XNN_MACRO_SHIM.sub("", text)
    return text, removed


FUNCTION_START = re.compile(r"^(?:static\s+)?void\s+(xnn_\w+)\s*\(", re.M)


def extract_function(text, name):
    """One ukernel's definition out of a source that defines several (upstream
    generators emit u4/u8/u16 variants into one file); the file's preamble
    (includes, helpers) is kept so the definition still compiles."""
    starts = [(m.start(), m.group(1)) for m in FUNCTION_START.finditer(text)]
    for i, (pos, fn) in enumerate(starts):
        if fn == name:
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            return text[:starts[0][0]] + text[pos:end]
    raise ValueError(f"{name} is not defined in this source")
