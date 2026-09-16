#!/usr/bin/env bash
# Run the scattered-gather bandwidth gate on the remote A100 host.
#
# Why a script and not ad-hoc ssh: the server answer differs from the laptop
# answer in exactly the ways that matter (real DDR instead of unified memory,
# and NUMA, which the laptop cannot have). So the run has to record CPU/NUMA/ISA
# provenance alongside the numbers, and has to sweep the NUMA binding, or the
# result is not interpretable.
#
# Three bindings are measured on purpose:
#   unbound        - what the engine gets today if nobody thinks about placement
#   membind=0      - best case, KV pinned to one socket
#   interleave=all - worst realistic case, KV spread across sockets
# The gap between the first and second is the size of the prize for adding NUMA
# awareness to the allocator; the third bounds the downside of ignoring it.
#
# Usage: bash tools/run_cpu_gather_bench_remote.sh
set -euo pipefail

HOST="${HOST:-10.232.195.203}"
REMOTE_DIR="${REMOTE_DIR:-/data00/home/sitian/tllm/cpu_gather_bench}"
RUN_TAG="${RUN_TAG:-cpu-gather-$(date +%Y%m%d-%H%M%S)}"
OUT="experiments/cpu_gather_bandwidth/${RUN_TAG}"

mkdir -p "$OUT"
echo ">>> run tag: $RUN_TAG"
echo ">>> output : $OUT"

echo ">>> staging source"
ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
scp -q tools/cpu_kv_gather_bandwidth.c "$HOST:$REMOTE_DIR/"

echo ">>> collecting machine provenance"
ssh "$HOST" 'lscpu' > "$OUT/lscpu.txt" 2>&1 || true
ssh "$HOST" 'numactl -H 2>/dev/null || echo NUMACTL_UNAVAILABLE' > "$OUT/numactl.txt" 2>&1 || true
ssh "$HOST" 'free -g' > "$OUT/free.txt" 2>&1 || true
ssh "$HOST" 'nproc' > "$OUT/nproc.txt" 2>&1 || true
# ISA support decides whether the int8 branch can ever pay off: without VNNI the
# dequant has to go through float math, so quantisation saves DRAM traffic but
# not the FLOPs that this path is actually bound by.
ssh "$HOST" 'grep -m1 ^flags /proc/cpuinfo | tr " " "\n" | grep -Ei "avx512|vnni|amx|avx2" | sort -u' \
    > "$OUT/isa_flags.txt" 2>&1 || true
ssh "$HOST" 'cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || echo THP_UNKNOWN' \
    > "$OUT/thp.txt" 2>&1 || true

NPROC="$(tr -d '[:space:]' < "$OUT/nproc.txt" 2>/dev/null || echo 8)"
MEM_GB="$(awk '/^Mem:/{print $2}' "$OUT/free.txt" 2>/dev/null || echo 32)"
# Buffer must comfortably exceed last-level cache but must not push the shared
# box into swap; a quarter of RAM capped at 32 GiB satisfies both.
BUF_GIB=$(( MEM_GB / 4 ))
[ "$BUF_GIB" -gt 32 ] && BUF_GIB=32
[ "$BUF_GIB" -lt 4 ] && BUF_GIB=4
echo ">>> cores=$NPROC  ram=${MEM_GB}GB  buffer=${BUF_GIB}GiB"

echo ">>> building"
ssh "$HOST" "cd '$REMOTE_DIR' && cc -O3 -o cpu_kv_gather_bandwidth cpu_kv_gather_bandwidth.c -lpthread" \
    2>&1 | tee "$OUT/build.log"

run_binding() {
    local name="$1" prefix="$2"
    echo ">>> measuring binding: $name"
    # shellcheck disable=SC2029
    ssh "$HOST" "cd '$REMOTE_DIR' && $prefix ./cpu_kv_gather_bandwidth $BUF_GIB $NPROC" \
        > "$OUT/gather-${name}.json" 2> "$OUT/gather-${name}.txt" || {
            echo "!!! binding $name failed, see $OUT/gather-${name}.txt"
            return 0
        }
    sed -n '1,60p' "$OUT/gather-${name}.txt"
}

run_binding "unbound" ""
if grep -q available "$OUT/numactl.txt" 2>/dev/null; then
    run_binding "membind0" "numactl --cpunodebind=0 --membind=0"
    run_binding "interleave" "numactl --interleave=all"
else
    echo ">>> numactl unavailable or single node; skipping NUMA bindings"
fi

echo
echo ">>> ISA flags found:"
cat "$OUT/isa_flags.txt" 2>/dev/null || true
echo ">>> done. artifacts in $OUT"
