#!/usr/bin/env bash
# O4a _off5 CNN top-up RESUME launcher (prepared 2026-10-01 19:20, NOT auto-run).
# Use only if 1777918 times out at its 10 h limit (2026-10-02 04:08). Submits a 24 h copy with per-H1-block
# checkpoints (SM_TOPUP_CKPT) and repoints the -2.05 merge 1777919 to the new job.
# Usage: bash launchers/m2_o4a_off5_topup_resume.sh            (immediate)
#        bash launchers/m2_o4a_off5_topup_resume.sh afternotok  (conditional: runs only if 1777918 fails)
set -euo pipefail
ROOT=${MADGRAV_ROOT}; RUN=o4a; LOGS=${MADGRAV_ROOT}/m2/prep/logs
JIDS=$ROOT/launchers/logs_m2/${RUN}_off5_m2_gx_jids.env; OLD=1777918; MERGE=1777919
GPU=(--partition=${PARTITION_GPU} --account=${SLURM_ACCOUNT} --qos=${QOS_GPU} --requeue)
DEP=(); [ "${1:-}" = afternotok ] && DEP=(--dependency=afternotok:$OLD)
J=$(sbatch --parsable --job-name=m2-${RUN}-off5-cnntopup-resume "${GPU[@]}" --gres=gpu:2 --time=24:00:00 \
   --output=$LOGS/off5_${RUN}_cnntopup_resume_%j.log "${DEP[@]}" \
   --export="ALL,SM_RUN_SEL=${RUN}_off5,SM_ARM_TOPUP=-1e9,SM_TOPUP_IDX=901,OFF5_RUN=$RUN" \
   $ROOT/launchers/cnn_topup_resume.sbatch)
echo "TOPUP_RESUME=$J" | tee -a $JIDS
# merge fires on whichever top-up succeeds ('?' = OR); both jobs must still be pending/running when this runs
scontrol update job $MERGE Dependency="afterok:$OLD?afterok:$J" && scontrol show job $MERGE | grep -oE 'Dependency=\S+'
