#!/bin/bash
# driver.sh — method_fix 运行队列（xargs -P 2，2 核预算；timeout 保护单次运行）
cd /mnt/agents/output/analysis/method_fix || exit 1
cat > /tmp/mf_queue.txt <<'EOF'
python3 scripts/run_one.py --cell 2 --rep 0 --arm v1 --out npz/cell02_rep0_v1.npz
python3 scripts/run_one.py --cell 2 --rep 1 --arm v1 --out npz/cell02_rep1_v1.npz
python3 scripts/run_one.py --cell 2 --rep 2 --arm v1 --out npz/cell02_rep2_v1.npz
python3 scripts/run_one.py --cell 6 --rep 1 --arm v1 --out npz/cell06_rep1_v1.npz
python3 scripts/run_one.py --cell 6 --rep 2 --arm v1 --out npz/cell06_rep2_v1.npz
python3 scripts/run_one.py --cell 11 --rep 0 --arm v1 --out npz/cell11_rep0_v1.npz
python3 scripts/run_one.py --cell 11 --rep 1 --arm v1 --out npz/cell11_rep1_v1.npz
python3 scripts/run_one.py --cell 11 --rep 2 --arm v1 --out npz/cell11_rep2_v1.npz
python3 scripts/run_one.py --cell 22 --rep 0 --arm v1 --out npz/cell22_rep0_v1.npz
python3 scripts/run_one.py --cell 22 --rep 1 --arm v1 --out npz/cell22_rep1_v1.npz
python3 scripts/run_one.py --cell 22 --rep 2 --arm v1 --out npz/cell22_rep2_v1.npz
python3 scripts/run_one.py --cell 2 --rep 0 --arm v2 --lam 0.1 --out npz/cell02_rep0_v2_l0.1.npz
python3 scripts/run_one.py --cell 2 --rep 0 --arm v2 --lam 1.0 --out npz/cell02_rep0_v2_l1.0.npz
python3 scripts/run_one.py --cell 6 --rep 0 --arm v2 --lam 0.1 --out npz/cell06_rep0_v2_l0.1.npz
python3 scripts/run_one.py --cell 6 --rep 0 --arm v2 --lam 1.0 --out npz/cell06_rep0_v2_l1.0.npz
python3 scripts/run_one.py --cell 11 --rep 0 --arm v2 --lam 0.1 --out npz/cell11_rep0_v2_l0.1.npz
python3 scripts/run_one.py --cell 11 --rep 0 --arm v2 --lam 0.44 --out npz/cell11_rep0_v2_l0.44.npz
python3 scripts/run_one.py --cell 11 --rep 0 --arm v2 --lam 1.0 --out npz/cell11_rep0_v2_l1.0.npz
python3 scripts/run_one.py --cell 22 --rep 0 --arm v2 --lam 0.1 --out npz/cell22_rep0_v2_l0.1.npz
python3 scripts/run_one.py --cell 22 --rep 0 --arm v2 --lam 0.44 --out npz/cell22_rep0_v2_l0.44.npz
python3 scripts/run_one.py --cell 22 --rep 0 --arm v2 --lam 1.0 --out npz/cell22_rep0_v2_l1.0.npz
EOF
run_line() {
  cmd="$1"
  tag=$(echo "$cmd" | sed -n 's/.*--out npz\/\(.*\)\.npz/\1/p')
  echo "[driver] start $tag $(date +%H:%M:%S)"
  timeout 1500 $cmd > "logs/${tag}.log" 2>&1
  rc=$?
  echo "[driver] done  $tag rc=$rc $(date +%H:%M:%S)"
}
export -f run_line
xargs -P 2 -I {} bash -c 'run_line "$@"' _ {} < /tmp/mf_queue.txt
echo "[driver] ALL DONE $(date +%H:%M:%S)"
