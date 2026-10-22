#!/usr/bin/env bash
# Submit the _off5 -> _s3rank chain for one run (usage: submit_chain.sh o4a|o4b). CPU partition only. Appends the ids to logs/chain_<run>_<date>.txt.
set -uo pipefail
RUN=${1:?run}; W=${MADGRAV_ROOT}/m2/arm/s3off5; E="SM_RUN_SEL=$RUN"
[ -s ${MADGRAV_ROOT}/launchers/logs_m2/${RUN}_off5_m2_gx_jids.env ] || { echo "no _off5 jids env for $RUN"; exit 1; }
sub(){ sbatch --parsable "$@" $W/stage.sbatch | cut -d';' -f1; }
J1=$(sub --job-name=off5s3-bgfeats-$RUN   --cpus-per-task=16 --mem=240G --time=10:00:00 --export=ALL,STAGE=bgfeats,$E)
J3=$(sub --job-name=off5s3-feats-$RUN     --cpus-per-task=16 --mem=120G --time=02:00:00 --export=ALL,STAGE=feats,$E)
J2=$(sub --job-name=off5s3-missbank-$RUN  --cpus-per-task=8  --mem=120G --time=03:00:00 --dependency=afterok:$J1 --export=ALL,STAGE=missbank,$E)
J4=$(sub --job-name=off5s3-s3score-$RUN   --cpus-per-task=16 --mem=120G --time=14:00:00 --array=0-15%8 --dependency=afterok:$J2 --export=ALL,STAGE=s3score,$E)
J5=$(sub --job-name=off5s3-concat-$RUN    --cpus-per-task=2  --mem=32G  --time=00:30:00 --dependency=afterok:$J4 --export=ALL,STAGE=concat,$E)
J6=$(sub --job-name=off5s3-rankscore-$RUN --cpus-per-task=8  --mem=120G --time=01:00:00 --dependency=afterok:$J2:$J3 --export=ALL,STAGE=rankscore,$E)
J7=$(sub --job-name=off5s3-merge-$RUN     --cpus-per-task=44 --mem=300G --time=10:00:00 --dependency=afterok:$J5:$J6 --export=ALL,STAGE=merge,$E)
echo "$RUN chain $(date '+%F %T'): bgfeats=$J1 feats=$J3 missbank=$J2 s3score=$J4 concat=$J5 rankscore=$J6 merge=$J7" | tee -a $W/logs/chain_${RUN}_$(date +%Y%m%d).txt
