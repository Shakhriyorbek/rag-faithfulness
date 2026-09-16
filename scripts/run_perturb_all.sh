#!/bin/bash
# Re-run the falsification probe under every evaluator after the F1/F3/F4/F6
# fixes. Deterministic case construction, so all scorers see the same cases.
set -u
cd ~/rag-faithfulness
mkdir -p ~/rag_faithfulness/outputs/logs
LOG=~/rag_faithfulness/outputs/logs/perturb_2026-09-01
for S in numeric nli nli_concat claim; do
  echo "=== scorer $S  $(date) ==="
  python3 -u src/perturbation_check.py --scope-n 1000 --scorer $S --limit 200 \
      > ${LOG}_$S.log 2>&1
  echo "exit $? for $S"
  tail -3 ${LOG}_$S.log
done
echo "=== scorer align  $(date) ==="
PYTHONPATH=$HOME/align_env python3 -u src/perturbation_check.py --scope-n 1000 \
    --scorer align --limit 200 > ${LOG}_align.log 2>&1
echo "exit $? for align"
echo "ALL DONE $(date)"
