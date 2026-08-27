#!/bin/bash
cd /mnt/agents/output/analysis/method_fix/v3/v34_full || exit 1
run_line() {
  cmd="$1"
  tag=$(echo "$cmd" | sed -n 's/.*--out npz\/\(.*\)\.npz/\1/p')
  echo "[driver] start $tag $(date +%H:%M:%S)"
  timeout 1800 $cmd > "logs/${tag}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then echo "[driver] FAIL  $tag rc=$rc $(date +%H:%M:%S)"; else echo "[driver] done  $tag $(date +%H:%M:%S)"; fi
}
export -f run_line
xargs -P 2 -I {} bash -c 'run_line "$@"' _ {} < scripts/queue.txt
echo "[driver] ALL DONE $(date +%H:%M:%S)"
