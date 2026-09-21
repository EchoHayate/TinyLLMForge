#!/usr/bin/env bash
# Verify-then-delete cleanup for /data00/home/sitian.
# Default is DRY RUN. Pass --apply to actually delete.
# Trusted originals: /tmp/trusted_originals.txt (sha256 + basename), verified locally
# against the Git LFS pointer OIDs in TinyLLMForge@feat/kv-sparse-attention.
set -uo pipefail

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

TRUSTED=/tmp/trusted_originals.txt
LOG=/tmp/cleanup_report.txt
: > "$LOG"

log() { echo "$@" | tee -a "$LOG"; }

log "=== cleanup start: $(date) ; apply=$APPLY ==="
log ""
log "--- free space before ---"
df -h /data00/home/sitian | tee -a "$LOG"
log ""

# ---------- Target 1: artifacts.failed-resume (18G, duplicate of sibling artifacts/) ----------
FAILED="/data00/home/sitian/tllm/speculation-router-runs/qwen3-06b-router-controlled-canonical-20260717-154410/artifacts.failed-resume"
GOOD="/data00/home/sitian/tllm/speculation-router-runs/qwen3-06b-router-controlled-canonical-20260717-154410/artifacts"

log "--- target 1: artifacts.failed-resume ---"
if [ -d "$FAILED" ] && [ -d "$GOOD" ]; then
  log "failed-resume: $(du -sh "$FAILED" | cut -f1)   artifacts: $(du -sh "$GOOD" | cut -f1)"
  nf=$(find "$FAILED" -type f | wc -l); ng=$(find "$GOOD" -type f | wc -l)
  log "file counts: failed-resume=$nf  artifacts=$ng"
  # only delete if the good sibling is non-empty and at least as populated
  if [ "$ng" -gt 0 ] && [ "$ng" -ge "$nf" ]; then
    log "DECISION: safe to delete (sibling artifacts/ is intact and >= as complete)"
    if [ "$APPLY" = "1" ]; then rm -rf "$FAILED" && log "DELETED $FAILED"; else log "DRY RUN: would delete $FAILED"; fi
  else
    log "DECISION: SKIP - sibling artifacts/ is not clearly complete, needs eyes"
  fi
else
  log "SKIP: one of the two directories is missing (already cleaned?)"
fi
log ""

# ---------- Target 2: duplicated needle_sq_results/*.pt with tracked originals ----------
log "--- target 2: duplicated needle_sq_results .pt files ---"
log "verifying every remote copy against the trusted sha256 list before touching it"

reclaim=0
matched=0
unmatched=0

while IFS= read -r dir; do
  # never touch the canonical repo checkout itself
  case "$dir" in
    */TinyLLMForge/needle_sq_results) log "SKIP canonical checkout: $dir"; continue ;;
  esac
  while IFS= read -r f; do
    base=$(basename "$f")
    want=$(awk -v b="$base" '$2==b {print $1}' "$TRUSTED")
    if [ -z "$want" ]; then
      log "  KEEP (not in trusted list, unique file): $f"
      unmatched=$((unmatched+1))
      continue
    fi
    got=$(sha256sum "$f" | awk '{print $1}')
    if [ "$got" = "$want" ]; then
      sz=$(stat -c %s "$f")
      reclaim=$((reclaim+sz))
      matched=$((matched+1))
      if [ "$APPLY" = "1" ]; then rm -f "$f" && log "  DELETED (hash matches original): $f"; else log "  DRY RUN would delete (hash OK): $f"; fi
    else
      log "  KEEP (HASH DIFFERS from original - NOT a duplicate): $f"
      log "      want=$want"
      log "      got =$got"
      unmatched=$((unmatched+1))
    fi
  done < <(find "$dir" -maxdepth 1 -type f -name '*.pt' 2>/dev/null)
done < <(find /data00/home/sitian -maxdepth 8 -type d -name needle_sq_results 2>/dev/null)

log ""
log "verified duplicates: $matched   kept (unique or mismatched): $unmatched"
log "reclaimable from target 2: $(awk -v b="$reclaim" 'BEGIN{printf "%.1f GB", b/1073741824}')"
log ""
log "--- free space after ---"
df -h /data00/home/sitian | tee -a "$LOG"
log "=== cleanup end: $(date) ==="
