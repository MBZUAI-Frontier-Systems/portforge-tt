# portforge_tt/verify: formal verification methods

Each method proves a candidate equal to a trusted reference on a bounded domain and lives in its own
directory behind `contract.py`: `certify(cfg, family, entry, candidate, out, budget)` returns one
result with a verdict from `ALL_PASSED`, `PARTIAL`, `COUNTEREXAMPLE`, `UNSUPPORTED` or `ERROR`, and
only `ALL_PASSED` passes. A family lists its methods in `families.json` (`"verify": [...]`); the
pack adapter supplies each method's inputs at certification.

| method | checks | reference | inputs |
|---|---|---|---|
| `cbmc/` | the built machine code, lifted to C over an RVV model | the project's C reference | the project build holding the candidate |
| `cvc5/` | the RVV C source, through SALTyRN's intrinsic shims | the Neon kernel it was ported from | the source, the Neon text, params from the library's init |

A new method is a package here with a `certify` that returns `contract.result(...)`, plus an entry
in `contract.METHODS`.
