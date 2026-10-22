#!/usr/bin/env bash
# vt96p success mail (CPU, after vt96p-final): body = dated notebook entry (VT totals and ratios vs vt96/2.2/M1, placement losses by run, CPU-h,
# 0 GPU-h); attachments = table .md, compare .md, paper figure, recovery plane. Writes mail/body.txt + attachments.txt + READY; tries mailer.py
# itself, else the login-node mail_watcher.sh (m2/vt96p copy) sends it.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=1 --mem=4G --time=00:30:00
#SBATCH --job-name=vt96p-mail --output=${MADGRAV_ROOT}/m2/vt96p/logs/%x_%A.log
set -uo pipefail
PY=${PY:-python}; P=${MADGRAV_ROOT}/m2/vt96p; SC=${MADGRAV_SCRATCH}/madgrav2; M=$P/mail
MAILER=${MADGRAV_SCRATCH}/MADGRAV1/search_mode/pastro_final/campaign/mailer.py; SUBJ="[MADGRAV2] vt96p: VT of the frozen 96 recipe at the search's own placement - table, figures, paired comparison with vt96"
TAB=$(ls -t $P/VT_TABLE_M96P_*.md | head -1); CMP=$(ls -t $P/COMPARE_M96P_vs_M96_*.md | head -1)
IDS=$(grep -o "=[0-9]\{7\}" $P/JOBS.txt | tr -d '=' | sort -u | paste -sd,); CPUH=$(sacct -j $IDS -X -n -o CPUTimeRAW -P 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s/3600}'); GPUS=$(sacct -j $IDS -X -n -o AllocTRES -P 2>/dev/null | grep -c "gres/gpu" || true)
{
  echo "$(date '+%F') -- vt96p: VT of the frozen MADGRAV 2 version-2 recipe (96) with the injections placed as the search sees them (1 s grid, merger uniform in the tile, two scored windows), automatic chain."
  echo; echo "Why: vt96/vt22 scored three 0.25 s-spaced windows around each merger and kept the best, which flatters the fixed 1 s search near the net floors (VT_PLACEMENT_CONSISTENCY.md). vt96p is the same field (seed 20261004, same manifests, nper 450, paired row by row) at SM_INJ_PLACE=prod; scoring and relabel unchanged. Gate before the campaign: patched inject.py at grid placement byte-identical to git HEAD on a 40-row smoke."
  echo; echo "VT (comoving, source frame, Gpc^3 yr), from $TAB:"; grep "^Total VT\|^Per run" $TAB
  echo; echo "Sum over the four runs, MADGRAV1 bins:"; awk '/^## Sum over the four runs/{f=1} f&&/^\|/{print} /^Total VT/{f=0}' $TAB
  echo; echo "Paired comparison with vt96 (same injections), from $CMP:"; grep "^## " $CMP; grep "net SNR shift\|losses by cause\|gains by cause" $CMP
  echo; echo "Resources: ${CPUH:-?} CPU-h over the chain job IDs in JOBS.txt (sacct CPUTimeRAW); GPU allocations in these jobs: ${GPUS:-0} -> 0 GPU-h."
  echo; echo "Files: $TAB; $CMP; $SC/figures_m2/vt96p/vt96_paper.{pdf,png}; $SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96p.{pdf,png}; fields $SC/vt96/inj_scored_<run>_m96p.npz, relabel_inj_<run>_m96p.npz, vt_relabel_comoving_m96p.json; raw $SC/inj_<run>_m2_96p; record $P/DECISIONS_VT96P.md; job IDs $P/JOBS.txt."
} > $M/body.txt
printf "%s\n" "$TAB" "$CMP" "$SC/figures_m2/vt96p/vt96_paper.png" "$SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96p.png" > $M/attachments.txt
echo "$SUBJ" > $M/subject.txt; touch $M/READY; echo "[vt96p-mail] body + attachments written, READY marker set $(date '+%F %T')"; cat $M/body.txt
if [ ! -e $M/SENT ]; then for a in 1 2 3; do out=$($PY $MAILER "$SUBJ" $M/body.txt $(cat $M/attachments.txt) 2>&1 | tail -1); echo "try $a: $out"; echo "$out" | grep -q "SENT" && { touch $M/SENT; echo "$(date '+%F %T') sent from $(hostname)" >> $M/SENT; break; }; sleep 60; done; fi
[ -e $M/SENT ] && echo "[vt96p-mail] sent" || echo "[vt96p-mail] not sent from the compute node -- the login-node mail_watcher.sh will send it"; exit 0
