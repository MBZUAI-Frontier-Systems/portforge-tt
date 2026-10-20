#!/usr/bin/env bash
# Bounded SMT equivalence of RVV kernels against the XNNPACK Neon kernel, using
# SALTyRN's verifier (cvc5). Runs inside portforge-rvv-host.
# Usage: verify.sh <src-dir> <out-dir> [vlen] [timeout-seconds] [params.json]
# Modes per kernel:
#   concrete  SALTyRN's stock edge-value configs
#   symbolic  SALTyRN's symbolic params (one proof for all params in its ranges)
#   xnnpack   whole param structs from params.json (difftest --dump-params),
#             i.e. what XNNPACK's init function actually produces
set -euo pipefail
src=$1 out=$2 vlen=${3:-512} budget=${4:-120} params=${5:-}
# The sweep grows the batch of one config until the budget is gone, so later
# configs never run. In xnnpack mode cap the batch so every config is checked.
xnn_max_batch=${XNN_MAX_BATCH:-8}
wrapper=$(cd "$(dirname "$0")/../../tools" 2>/dev/null && pwd)/saltyrn_configs.py
mkdir -p "$out/kernels"
neon=$SALTYRN/kernels/source/qs8-vadd-minmax.c   # SALTyRN's prepped copy of neon-ld64-u16

# Upstream XNNPACK RVV kernel, prepped the way SALTyRN preps its inputs:
# drop #include lines and rename the ukernel to test_rvv.
sed -e '/^#include/d' \
    -e 's/xnn_qs8_vadd_minmax_ukernel__rvv_u2v/test_rvv/' \
    "$src/xnnpack/src/qs8-vadd/gen/qs8-vadd-minmax-rvv-u2v.c" > "$out/kernels/xnn_rvv_u2v.c"
sed -e '/^#include/d' "$src/optimized-rvv-kernels/qs8-vadd-minmax.c" > "$out/kernels/saltyrn_rvv_opt.c"

cd "$SALTYRN"
for kernel in xnn_rvv_u2v saltyrn_rvv_opt; do
  modes=(${VERIFY_MODES:-concrete symbolic})
  [[ -n $params ]] && modes+=(xnnpack)
  for mode in "${modes[@]}"; do
    flags=() entry=(-m src.workflow.verification.orchestrator)
    [[ $mode == symbolic ]] && flags+=(--symbolic-params)
    [[ $mode == xnnpack ]] && entry=("$wrapper" "$params") flags+=(--max-batch "$xnn_max_batch")
    log=$out/$kernel.$mode.vlen$vlen.log
    echo "== $kernel vs neon, $mode params, VLEN=$vlen, budget ${budget}s"
    set +e
    python "${entry[@]}" \
      --neon "$neon" --rvv "$out/kernels/$kernel.c" \
      --vlen "$vlen" --timeout "$budget" --backend cvc5 "${flags[@]}" >"$log" 2>&1
    status=$?
    set -e
    tail -n 25 "$log" | grep -iE 'verdict|VERIFIED|COUNTEREXAMPLE|PARTIAL|TIMEOUT|batch|error' | tail -n 8
    echo "   exit=$status log=$log"
  done
done
