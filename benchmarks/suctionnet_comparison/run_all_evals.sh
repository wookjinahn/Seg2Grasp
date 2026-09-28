#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate suctionnet_eval
cd ~/Desktop/Codes/s2g_suctionnet_bench
for dump in normal_std seg2grasp_committed seg2grasp_bestscore; do
  echo "=================== EVAL $dump ==================="
  python full_eval.py --dump ./preds_full/$dump --proc 15 --res-out ./preds_full/${dump}_res.npy 2>&1 \
    | grep -E "SUMMARY|AP_top|per-thr|raw res|Error|Traceback"
  echo "=================== DONE $dump ==================="
done
echo "ALL_EVALS_COMPLETE"
