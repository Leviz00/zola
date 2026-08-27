#!/bin/bash
cd /mnt/agents/output/analysis/method_fix || exit 1
cat > /tmp/mf_queue2.txt <<'Q'
python3 scripts/run_one.py --cell 2 --rep 1 --arm v2 --lam 1.0 --out npz/cell02_rep1_v2_l1.0.npz
python3 scripts/run_one.py --cell 2 --rep 2 --arm v2 --lam 1.0 --out npz/cell02_rep2_v2_l1.0.npz
python3 scripts/run_one.py --cell 6 --rep 1 --arm v2 --lam 1.0 --out npz/cell06_rep1_v2_l1.0.npz
python3 scripts/run_one.py --cell 6 --rep 2 --arm v2 --lam 1.0 --out npz/cell06_rep2_v2_l1.0.npz
python3 scripts/run_one.py --cell 11 --rep 1 --arm v2 --lam 1.0 --out npz/cell11_rep1_v2_l1.0.npz
python3 scripts/run_one.py --cell 11 --rep 2 --arm v2 --lam 1.0 --out npz/cell11_rep2_v2_l1.0.npz
python3 scripts/run_one.py --cell 22 --rep 1 --arm v2 --lam 0.1 --out npz/cell22_rep1_v2_l0.1.npz
python3 scripts/run_one.py --cell 22 --rep 2 --arm v2 --lam 0.1 --out npz/cell22_rep2_v2_l0.1.npz
Q
run_line() {
  cmd="$1"
  tag=$(echo "$cmd" | sed -n 's/.*--out npz\/\(.*\)\.npz/\1/p')
  echo "[driver] start $tag $(date +%H:%M:%S)"
  timeout 1800 $cmd > "logs/${tag}.log" 2>&1
  echo "[driver] done  $tag rc=$? $(date +%H:%M:%S)"
}
export -f run_line
xargs -P 2 -I {} bash -c 'run_line "$@"' _ {} < /tmp/mf_queue2.txt
echo "[driver] ALL DONE $(date +%H:%M:%S)"
