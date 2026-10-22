#!/usr/bin/env bash
# vt96 success mail (CPU, after vt96-final): body = dated notebook entry (validation PASS lines, VT totals and ratios, paired flips, CPU-h,
# 0 GPU-h); attachments = table .md, compare .md, vt96_paper.png, recovery plane png. Writes mail/body.txt + mail/attachments.txt + mail/READY;
# tries mailer.py itself (compute nodes usually have no SMTP route) and touches mail/SENT on success; otherwise the login-node
# mail_watcher.sh sends it.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=1 --mem=4G --time=00:30:00
#SBATCH --job-name=vt96-mail --output=${MADGRAV_ROOT}/m2/vt96/logs/%x_%A.log
set -uo pipefail
PY=${PY:-python}; W=${MADGRAV_ROOT}/m2/vt96; SC=${MADGRAV_SCRATCH}/madgrav2; M=$W/mail
MAILER=${MADGRAV_SCRATCH}/MADGRAV1/search_mode/pastro_final/campaign/mailer.py; SUBJ="[MADGRAV2] VT of the frozen 96 recipe (vt96) - table, figures, paired comparison with 2.2"
TAB=$(ls -t $W/VT_TABLE_M96_*.md | head -1); CMP=$(ls -t $W/COMPARE_M96_vs_M22_*.md | head -1); VAL=$(ls -t $W/logs/VALIDATION_2*.txt | head -1)
IDS=$(grep -o "=[0-9]\{7\}" $W/JOBS.txt | tr -d '=' | sort -u | paste -sd,); CPUH=$(sacct -j $IDS -X -n -o CPUTimeRAW -P 2>/dev/null | awk '{s+=$1} END{printf "%.0f", s/3600}'); GPUS=$(sacct -j $IDS -X -n -o AllocTRES -P 2>/dev/null | grep -c "gres/gpu" || true)
{
  echo "$(date '+%F') -- vt96: injection efficiency / VT of the frozen MADGRAV 2 version-2 recipe (96 catalogue detections), automatic chain."
  echo; echo "Campaign: seed 20261004, nper 450, same manifests as the 2.2 field of 2026-10-04 -> paired injection-by-injection; SM_CROP=coinc (inject96.py crop_start override), local-ASD columns (SM_INJ_ASDVETO=1; LOCAL_ALL on O3), Grad-CAM-crop CNN columns for the guard. Scoring far96.py: floor 2.75 + zero-lag gates, arm max:-2.05, S3 0.5357 + HGB ranker, LM rule 0.0382, local-ASD veto (net clause everywhere, feature clause ON in O3 / OFF in O4 by the plateau metric), net-sigma guard in its production form, loglr cap bypass. CPU partition only."
  echo; echo "Validation against the production _v2 merge (per-trigger FAR fields, local-ASD decisions, detections_v2 / detections_v2_nsguard membership, cap bypass, the 96 set):"; cat $VAL
  echo; echo "VT (comoving, source frame, Gpc^3 yr), from $TAB:"; grep -A0 "^Total VT\|^Per run" $TAB
  echo; echo "Sum over the four runs, MADGRAV1 bins:"; awk '/^## Sum over the four runs/{f=1} f&&/^\|/{print} /^Total VT/{f=0}' $TAB
  echo; echo "Paired comparison with the 2.2 field (same injections), from $CMP:"; grep "^## " $CMP; grep "losses by cause\|gains by cause" $CMP
  echo; echo "Resources: ${CPUH:-?} CPU-h over the chain job IDs in JOBS.txt (sacct CPUTimeRAW, incl. the 104 injection tasks, exports, validation, scores, relabel, final); GPU allocations in these jobs: ${GPUS:-0} -> 0 GPU-h."
  echo; echo "Files: $TAB; $CMP; $SC/figures_m2/vt96_paper.{pdf,png}; $SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96.{pdf,png}; fields $SC/vt96/inj_scored_<run>_m96.npz, relabel_inj_<run>_m96.npz, vt_relabel_comoving_m96.json; record $W/DECISIONS_VT96.md; job IDs $W/JOBS.txt."
} > $M/body.txt
printf "%s\n" "$TAB" "$CMP" "$SC/figures_m2/vt96_paper.png" "$SC/figures_m2/mtot_snr_smooth_m2_v2_96_vt96.png" > $M/attachments.txt
echo "$SUBJ" > $M/subject.txt; touch $M/READY; echo "[vt96-mail] body + attachments written, READY marker set $(date '+%F %T')"; cat $M/body.txt
if [ ! -e $M/SENT ]; then for a in 1 2 3; do out=$($PY $MAILER "$SUBJ" $M/body.txt $(cat $M/attachments.txt) 2>&1 | tail -1); echo "try $a: $out"; echo "$out" | grep -q "SENT" && { touch $M/SENT; echo "$(date '+%F %T') sent from $(hostname)" >> $M/SENT; break; }; sleep 60; done; fi
[ -e $M/SENT ] && echo "[vt96-mail] sent" || echo "[vt96-mail] not sent from the compute node -- the login-node mail_watcher.sh will send it"; exit 0
