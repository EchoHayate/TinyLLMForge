#!/usr/bin/env bash
set -uo pipefail

root=/Users/bytedance/dev/TinyLLMForge-adaptive-ngram/experiments/cuda_graph
state="$root/archive-manifest-20260807"
bundle="$root/archive-bundle-20260807"
destination=tinyllmforge-gdrive:TinyLLMForge-archives/cuda_graph-forensic-packs-20260807
log="$state/upload.log"
status="$state/upload.status"
verification="$state/verification.status"
report="$state/archive-report.md"

rm -f "$status" "$status.tmp" "$verification" "$verification.tmp"
mkdir -p "$bundle"
rc=0

while IFS= read -r name; do
  [[ -n "$name" ]] || continue
  archive="$bundle/$name.tar"
  if [[ ! -f "$archive" ]]; then
    printf '[%s] PACK %s\n' "$(date -u +%FT%TZ)" "$name" >>"$log"
    tar -C "$root" -cf "$archive.tmp" "$name" || {
      rc=$?
      rm -f "$archive.tmp"
      break
    }
    mv "$archive.tmp" "$archive"
  fi
done <"$state/archive_roots.txt"

if [[ "$rc" -eq 0 ]]; then
  cp "$state/archive_roots.txt" "$bundle/"
  cp "$state/manifest.jsonl" "$bundle/"
  cp "$state/summary.json" "$bundle/"
  cp "$state/SHA256SUMS" "$bundle/source-SHA256SUMS"
  cp "$state/MD5SUMS" "$bundle/source-MD5SUMS"
  (
    cd "$bundle" || exit
    shasum -a 256 ./*.tar >archive-SHA256SUMS.tmp &&
      mv archive-SHA256SUMS.tmp archive-SHA256SUMS
    md5 -r ./*.tar >archive-MD5SUMS.tmp &&
      mv archive-MD5SUMS.tmp archive-MD5SUMS
  ) || rc=$?
fi

if [[ "$rc" -eq 0 ]]; then
  printf '[%s] UPLOAD bundle\n' "$(date -u +%FT%TZ)" >>"$log"
  rclone copy "$bundle" "$destination" \
    --immutable \
    --checksum \
    --transfers 2 \
    --checkers 4 \
    --drive-chunk-size 128M \
    --retries 10 \
    --low-level-retries 20 \
    --timeout 10m \
    --contimeout 30s \
    --stats 60s \
    --stats-one-line \
    --log-file "$log" \
    --log-level INFO || rc=$?
fi

if [[ "$rc" -eq 0 ]]; then
  printf '[%s] CHECK bundle\n' "$(date -u +%FT%TZ)" >>"$log"
  rclone check "$bundle" "$destination" \
    --checksum \
    --one-way \
    --checkers 8 \
    --retries 10 \
    --low-level-retries 20 \
    --log-file "$log" \
    --log-level INFO || rc=$?
fi

if [[ "$rc" -eq 0 ]]; then
  printf '0\n' >"$verification.tmp"
  mv "$verification.tmp" "$verification"
  {
    printf '# CUDA Graph Forensic Archive\n\n'
    printf -- '- Destination: `%s`\n' "$destination"
    printf -- '- Source roots: `4`\n'
    printf -- '- Source files: `13,934`\n'
    printf -- '- Source bytes: `69,169,127,166`\n'
    printf -- '- Package format: `uncompressed tar`\n'
    printf -- '- Package count: `4`\n'
    printf -- '- Source integrity: `SHA256SUMS`\n'
    printf -- '- Package integrity: `archive-SHA256SUMS`, `archive-MD5SUMS`\n'
    printf -- '- Local/remote verification: `rclone check --checksum --one-way`\n'
    printf -- '- Verification result: `PASS`\n'
    printf -- '- Completed at: `%s`\n' "$(date -u +%FT%TZ)"
    printf -- '- Original local data deleted: `no`\n'
    printf -- '- Temporary local tar packages deleted: `no`\n'
    printf -- '- Earlier partial object-tree trial retained remotely: `yes (12-32 small files)`\n'
  } >"$report.tmp"
  mv "$report.tmp" "$report"
  rclone copyto "$report" "$destination/archive-report.md" \
    --immutable \
    --checksum \
    --log-file "$log" \
    --log-level INFO || rc=$?
fi

if [[ "$rc" -ne 0 ]]; then
  printf '%s\n' "$rc" >"$verification.tmp"
  mv "$verification.tmp" "$verification"
fi
printf '%s\n' "$rc" >"$status.tmp"
mv "$status.tmp" "$status"
exit "$rc"
