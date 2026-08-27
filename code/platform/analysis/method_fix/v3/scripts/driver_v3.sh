#!/bin/bash
cd /mnt/agents/output/analysis/method_fix/v3 || exit 1
cat > /tmp/v3_queue.txt <<'Q'
python3 scripts/run_one_v3.py --cell 6 --rep 1 --out npz/cell06_rep1_v3.npz
python3 scripts/run_one_v3.py --cell 6 --rep 2 --out npz/cell06_rep2_v3.npz
python3 scripts/run_one_v3.py --cell 2 --rep 0 --out npz/cell02_rep0_v3.npz
python3 scripts/run_one_v3.py --cell 2 --rep 1 --out npz/cell02_rep1_v3.npz
python3 scripts/run_one_v3.py --cell 2 --rep 2 --out npz/cell02_rep2_v3.npz
python3 scripts/run_one_v3.py --cell 11 --rep 0 --out npz/cell11_rep0_v3.npz
python3 scripts/run_one_v3.py --cell 11 --rep 1 --out npz/cell11_rep1_v3.npz
python3 scripts/run_one_v3.py --cell 11 --rep 2 --out npz/cell11_rep2_v3.npz
python3 scripts/run_one_v3.py --cell 22 --rep 0 --out npz/cell22_rep0_v3.npz
python3 scripts/run_one_v3.py --cell 22 --rep 1 --out npz/cell22_rep1_v3.npz
python3 scripts/run_one_v3.py --cell 22 --rep 2 --out npz/cell22_rep2_v3.npz
Q
run_line() {
  cmd="$1"
  tag=$(echo "$cmd" | sed -n 's/.*--out npz\/\(.*\)\.npz/\1/p')
  echo "[driver] start $tag $(date +%H:%M:%S)"
  timeout 1500 $cmd > "logs/${tag}.log" 2>&1
  echo "[driver] done  $tag rc=$? $(date +%H:%M:%S)"
}
export -f run_line
xargs -P 2 -I {} bash -c 'run_line "$@"' _ {} < /tmp/v3_queue.txt
echo "[driver] ALL DONE $(date +%H:%M:%S)"
