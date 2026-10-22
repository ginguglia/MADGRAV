#!/usr/bin/env bash
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=4 --mem=32G --time=00:20:00 --job-name=m2-spec-evalsingle
#SBATCH --output=${MADGRAV_ROOT}/m2/spec/logs/evalsingle_%j.log
cd ${MADGRAV_ROOT}/m2/spec; MD=${MADGRAV_SCRATCH}/madgrav2_spec/models_single
SPEC_SPLIT=single ${PY:-python} -W ignore eval_pilot.py
${PY:-python} plot_losses.py $MD/loss_single_s0.png $MD/spec_hm_single_s0.json $MD/spec_lm_single_s0.json $MD/spec_s2_single_s0.json
