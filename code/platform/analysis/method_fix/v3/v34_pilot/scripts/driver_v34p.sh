#!/bin/bash
cd /mnt/agents/output/analysis/method_fix/v3/v34_pilot || exit 1
cat > /tmp/v34p_queue.txt <<'Q'
python3 scripts/run_one_v34p.py --cell 1002 --rep 1 --out npz/cell1002_rep1.npz
python3 scripts/run_one_v34p.py --cell 1002 --rep 2 --out npz/cell1002_rep2.npz
python3 scripts/run_one_v34p.py --cell 1002 --rep 3 --out npz/cell1002_rep3.npz
python3 scripts/run_one_v34p.py --cell 1002 --rep 4 --out npz/cell1002_rep4.npz
python3 scripts/run_one_v34p.py --cell 1005 --rep 0 --out npz/cell1005_rep0.npz
python3 scripts/run_one_v34p.py --cell 1005 --rep 1 --out npz/cell1005_rep1.npz
python3 scripts/run_one_v34p.py --cell 1005 --rep 2 --out npz/cell1005_rep2.npz
python3 scripts/run_one_v34p.py --cell 1005 --rep 3 --out npz/cell1005_rep3.npz
python3 scripts/run_one_v34p.py --cell 1005 --rep 4 --out npz/cell1005_rep4.npz
python3 scripts/run_one_v34p.py --cell 1009 --rep 0 --out npz/cell1009_rep0.npz
python3 scripts/run_one_v34p.py --cell 1009 --rep 1 --out npz/cell1009_rep1.npz
python3 scripts/run_one_v34p.py --cell 1009 --rep 2 --out npz/cell1009_rep2.npz
python3 scripts/run_one_v34p.py --cell 1009 --rep 3 --out npz/cell1009_rep3.npz
python3 scripts/run_one_v34p.py --cell 1009 --rep 4 --out npz/cell1009_rep4.npz
Q
run_line() {
  cmd="$1"
  tag=$(echo "$cmd" | sed -n 's/.*--out npz\/\(.*\)\.npz/\1/p')
  echo "[driver] start $tag $(date +%H:%M:%S)"
  timeout 1800 $cmd > "logs/${tag}.log" 2>&1
  echo "[driver] done  $tag rc=$? $(date +%H:%M:%S)"
}
export -f run_line
xargs -P 2 -I {} bash -c 'run_line "$@"' _ {} < /tmp/v34p_queue.txt
echo "[driver] ALL DONE $(date +%H:%M:%S)"
