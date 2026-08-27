#!/bin/bash
cd /mnt/agents/output/analysis/method_fix/v3/v35_gating || exit 1
run_line() {
  cmd="$1"
  tag=$(echo "$cmd" | sed -n 's/.*--out rows\/\(.*\)\.csv/\1/p')
  timeout 1800 $cmd > "logs/${tag}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[driver] FAIL $tag rc=$rc $(date +%H:%M:%S)"; fi
}
export -f run_line
xargs -P 2 -I {} bash -c 'run_line "$@"' _ {} < scripts/queue_diag.txt
echo "[driver] ALL DONE $(date +%H:%M:%S)"
