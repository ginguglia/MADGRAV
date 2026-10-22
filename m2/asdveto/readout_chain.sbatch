#!/usr/bin/env bash
# Read-out of the local-ASD veto check, chained afterany on the injection arrays and background jobs; writes READOUT_ASDVETO_<date>.md and mails it.
#SBATCH --partition=${PARTITION_CPU} --account=${SLURM_ACCOUNT} --qos=${QOS_CPU} --time=01:00:00 --mem=16G --cpus-per-task=2
#SBATCH --output=${MADGRAV_ROOT}/m2/asdveto/logs/%x_%j.log
set -uo pipefail; W=${MADGRAV_ROOT}/m2/asdveto; PY=${PY:-python}; F=$W/READOUT_ASDVETO_$(date '+%Y%m%d_%H%M').md
cd $W && $PY -I readout_asdveto.py $F; rc=$?
{ echo "Local-ASD veto check read-out ($(date '+%F %T')), rc=$rc. Injection arrays 1804604-07, background 1804636-39. Record: $F"; echo; cat $F; } > $W/logs/mail_readout.txt
python3 ${MADGRAV_ROOT}/search_mode/pastro_final/campaign/mailer.py "MADGRAV2 ASD-veto check: read-out" $W/logs/mail_readout.txt 2>&1 | tail -3
exit $rc
