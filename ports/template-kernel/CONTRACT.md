# Contract: <library> kernels on RVV

## Interface
A candidate is one C file defining one kernel with the upstream signature and params
struct. The harness renames and registers it next to the kernel it was ported from.

## Correctness (hard gates, run by the harness on every candidate)
- G0: compiles for rv64gcv as the upstream build compiles RVV.
- G1: the library's own tests for the family pass at VLEN 128/256/512/1024 (QEMU),
  running exactly the listed tests: a crash, timeout or empty selection fails.
- G2: bit-exact with the reference kernel over every regime in `regimes.json`, every
  check batch (tails included), five input patterns, misaligned pointers, with guard
  bytes after the output; floats additionally against special values (NaN, ±0, ±inf,
  subnormals).
- G5 (certification only): held-out regimes the workers never see, bounded SMT
  equivalence (the family's verify methods, verify/contract.py) where one applies,
  the tests on the X280, and a second timing session.

## Speed (objective after the gates)
Per cell (regime x shape) the candidate is timed against the fastest *correct* upstream
kernel in the same process, in paired rounds; cells where no upstream kernel is correct
are excluded from the speed geomean and reported as correctness wins. Primary:
geomean speedup over cells. Constraint: no cell slower than the baseline beyond the
measured noise floor (confirmed by a re-measurement).

## Rules
- The full parameter envelope the library's init can produce, not the test values.
- Dispatch on parameters only by predicates proven to make each path exact on its
  whole domain; never special-case particular values.
- No reading tests, fixtures or oracle files to special-case inputs.
- The hot loop is vector code; no scalar fallback for the bulk of the work.
