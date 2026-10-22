#!/usr/bin/env bash
# GlitchGen bank v1 TOP-UP to 10k per class (2026-09-30, design decision): GAN stage --resume adds chunks 001-009 (chunk 000 kept, per-(class,chunk) seeds);
# parametric scripts have no --resume, so the 1k param dir is moved aside to param_1k_20260930 and regenerated (chunk 000 byte-identical by seed).
# Original header: GlitchGen bank v1 for the contaminated-positives / glitch-augmentation line (2026-09-29). 14 GAN classes (the 3 above 1 kHz dropped:
# Violin_Mode, 1080Lines, 1400Ripples), gs converter, 1000 per class, chunk 1000 (one model load per class), + 10 parametric classes
# (param_glitches.py incl. the low-frequency-burst confuser, whistle_param.py). Raw strain, 2 s @ 4096 Hz, SNR realistic vs aLIGO design.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=32 --mem=64G --time=03:00:00
#SBATCH --job-name=m2-glitchgen-bank-10k --output=${MADGRAV_ROOT}/m2/prep/logs/glitchgen_bank10k_%j.log
set -uo pipefail; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ${MADGRAV_SCRATCH}/glitchgen/scripts; T0=$SECONDS; echo "START $(date '+%F %T') $(hostname)"
${MADGRAV_SCRATCH}/glitchgen-venv/bin/python generate_glitches.py --out ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1 --converter gs --per-class 10000 --chunk-size 1000 --workers 30 --resume \
  --classes Scattered_Light Low_Frequency_Burst Low_Frequency_Lines Blip Repeating_Blips Power_Line Koi_Fish Tomte Scratchy Helix Air_Compressor Light_Modulation Paired_Doves Wandering_Line; echo "GAN rc=$? $((SECONDS-T0))s"
[ -d ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1/param ] && mv ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1/param ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1/param_1k_20260930
${MADGRAV_SCRATCH}/glitchgen-venv/bin/python param_glitches.py --out ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1/param --n 10000 --chunk-size 1000; echo "PARAM rc=$? $((SECONDS-T0))s"
${MADGRAV_SCRATCH}/glitchgen-venv/bin/python whistle_param.py --out ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1/param --n 10000 --chunk-size 1000; echo "WHISTLE rc=$? $((SECONDS-T0))s"
${MADGRAV_SCRATCH}/glitchgen-venv/bin/python pilot_checks.py ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1 ${MADGRAV_SCRATCH}/madgrav2/arm/synth/glitchgen_v1/param 2>&1 | tail -60
echo "END $(date '+%F %T') WALL=$((SECONDS-T0))s"
