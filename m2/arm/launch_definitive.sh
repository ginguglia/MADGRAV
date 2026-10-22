#!/bin/bash
# waits for the large positive generation, assembles data_v3, launches the definitive arm training, then the cross-fit scorer
export PATH=/opt/slurm/rocky-9.6/25-11-3-1/bin:$PATH
P=${MADGRAV_SCRATCH}/madgrav2/positives; D=${MADGRAV_SCRATCH}/madgrav2/arm/data_v3; M=${MADGRAV_SCRATCH}/madgrav2/arm/models
cd ${MADGRAV_ROOT}/m2/arm
until [ $(grep -lE 'END rc=0' $P/run_pos_1741369.log $P/run_pos_1741370.log $P/run_pos_1741371.log 2>/dev/null | wc -l) -ge 3 ]; do sleep 120; done
echo "[def] positives done $(date)"; mkdir -p $D
JA=$(sbatch --parsable --export=ALL,ARM_DATA=$D,ARM_SIG_RULE=pair4 run_assemble_v2.sbatch | tail -1); echo "[def] assemble job $JA"
until sacct -j $JA -X -o State -n 2>/dev/null | grep -qE 'COMPLETED|FAILED|CANCELLED'; do sleep 60; done
sacct -j $JA -X -o State -n; head -2 $D/ASSEMBLY.txt
JT=$(sbatch --parsable --export=ALL,ARM_TAG=_def,ARM_EPOCHS=60,ARM_DATA=$D,ARM_GLITCH_BALANCE=0.5 --array=0-19 run_train_arm.sbatch | tail -1); echo "[def] training array $JT (2 folds x 10 seeds)"
until [ -f $M/arm_def_f0_s0.json ] && [ -f $M/arm_def_f1_s0.json ]; do sleep 120; done
ARM_DATA=$D ${PY:-python} score_families.py _def 2>&1 | grep -vE 'arn' | tee $M/score_families_def_first.md | grep -E 'models per|families:|summed|WATCHLIST|FAIL|lowest'
echo "[def] first cross-fit scoring done $(date); rerun score_families.py _def when all 20 models exist"
