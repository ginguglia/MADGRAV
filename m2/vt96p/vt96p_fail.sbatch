#!/usr/bin/env bash
# vt96p failure mail (copy of m2/vt96/vt96_fail.sbatch, 2026-10-22) (CPU; one per upstream job, --dependency=afternotok): sacct line + log tail of the failed job -> mail/FAIL_<id>.txt +
# marker mail/FAIL_READY_<id>; tries mailer.py itself, else the login-node mail_watcher.sh sends it. Env: FAILED_JOB, FAILED_NAME.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --cpus-per-task=1 --mem=4G --time=00:20:00
#SBATCH --job-name=vt96p-fail --output=${MADGRAV_ROOT}/m2/vt96p/logs/%x_%A.log
set -uo pipefail; : "${FAILED_JOB:?}" "${FAILED_NAME:?}"
PY=${PY:-python}; W=${MADGRAV_ROOT}/m2/vt96p; M=$W/mail
MAILER=${MADGRAV_SCRATCH}/MADGRAV1/search_mode/pastro_final/campaign/mailer.py; SUBJ="[MADGRAV2] vt96p chain: FAILURE in $FAILED_NAME (job $FAILED_JOB)"
B=$M/FAIL_$FAILED_JOB.txt
{
  echo "$(date '+%F %T') -- vt96p chain: job $FAILED_JOB ($FAILED_NAME) ended in a failed state; downstream afterok jobs are killed (kill-on-invalid-dep)."
  echo; echo "sacct:"; sacct -j $FAILED_JOB -o JobID%18,JobName%22,State%14,ExitCode,Elapsed,MaxRSS,NodeList%14 -P
  echo; for f in $(ls -t $W/logs/*_${FAILED_JOB}.log $W/logs/*_${FAILED_JOB}_*.log 2>/dev/null | head -4); do echo "== tail $f"; grep -v "^\[inj96\] inject.py" $f | tail -25 | cut -c1-240; done
  echo; echo "Record: $W/DECISIONS_VT96P.md; job IDs: $W/JOBS.txt. Nothing is resubmitted automatically."
} > $B
echo "$SUBJ" > $M/FAIL_SUBJECT_$FAILED_JOB.txt; touch $M/FAIL_READY_$FAILED_JOB; cat $B
for a in 1 2 3; do out=$($PY $MAILER "$SUBJ" $B 2>&1 | tail -1); echo "try $a: $out"; echo "$out" | grep -q "SENT" && { echo "$(date '+%F %T') sent from $(hostname)" > $M/FAIL_SENT_$FAILED_JOB; break; }; sleep 60; done; exit 0
