#!/usr/bin/env bash
# End-to-end smoke for one kernel family on the two machines, run on the x86 host:
#   1. build    static difftest for x86-64 / aarch64 / riscv64   (container)
#   2. x86      native run: scalar, SSE4.1, AVX2                   (host CPU)
#   3. qemu     aarch64 (Neon) and riscv64 at VLEN 128/256/512/1024 (container)
#   4. x280     native run + timing on the Blackhole X280 cores    (ssh)
#   5. verify   SMT equivalence of both RVV kernels vs Neon        (container)
#   6. digests  every kernel on every platform must agree
# Usage: run.sh <workdir>     (workdir holds src/ with the pinned checkouts)
set -euo pipefail
work=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
kernel=$(basename "$here")
image=${PORTFORGE_RVV_IMAGE:-portforge-rvv-host:0}
x280_port=${X280_SSH_PORT:-2222}
x280_cpu=${X280_CPU:-1}
skip_verify=${SKIP_VERIFY:-0}
# A failing kernel exits 1; record it and keep going so every platform reports.
res=$work/results/qs8_vadd/$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$res"
in_box() { docker run --rm -v "$work/src:/src:ro" -v "$here/../..:/rvv:ro" -v "$res:/res" "$image" "$@"; }
x280() { ssh -F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
             -o LogLevel=ERROR -p "$x280_port" debian@localhost "$@"; }

echo "## 1. build"
in_box bash "/rvv/smoke/$kernel/build.sh" /src /res/bin | tee "$res/build.log"

echo "## 2. x86-64 native ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs))"
"$res/bin/x86_64/difftest" --bench --cpu 2 | tee "$res/x86_64.jsonl" || true

echo "## 3. qemu"
in_box qemu-aarch64 /res/bin/aarch64/difftest | tee "$res/qemu-aarch64.jsonl" || true
for vlen in 128 256 512 1024; do
  in_box qemu-riscv64 -cpu "rv64,v=true,vlen=$vlen,elen=64" /res/bin/riscv64/difftest \
    | sed "s/^{/{\"vlen\":$vlen,/" | tee "$res/qemu-riscv64-vlen$vlen.jsonl" || true
done

echo "## 4. x280 (Blackhole L2CPU, real RVV 1.0)"
x280 "mkdir -p /tmp/portforge && cat /proc/cpuinfo | grep -m1 uarch"
scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -P "$x280_port" \
  "$res/bin/riscv64/difftest" debian@localhost:/tmp/portforge/difftest
x280 "/tmp/portforge/difftest --bench --cpu $x280_cpu" | tee "$res/x280.jsonl" || true

if [[ $skip_verify != 1 ]]; then
  echo "## 5. verify (SALTyRN + cvc5)"
  "$res/bin/x86_64/difftest" --dump-params > "$res/params.json"
  in_box bash "/rvv/smoke/$kernel/verify.sh" /src /res/verify "${VERIFY_VLEN:-512}" "${VERIFY_BUDGET:-120}" \
    /res/params.json | tee "$res/verify.log"
fi

echo "## 6. digests"
python3 - "$res"/x86_64.jsonl "$res"/qemu-*.jsonl "$res"/x280.jsonl <<'EOF' | tee "$res/digests.txt"
import json, os, sys
digests = set()
for path in sys.argv[1:]:
    platform = os.path.basename(path).removesuffix(".jsonl")
    for line in open(path):
        r = json.loads(line)
        if "digest" not in r:
            continue
        digests.add(r["digest"])
        print(f"{platform:<22} {r['kernel']:<26} {r['status']:<5} {r['digest']} {r['first_failure']}")
print("cross-platform:", "IDENTICAL" if len(digests) == 1 else f"{len(digests)} distinct digests")
EOF
echo "results: $res"
