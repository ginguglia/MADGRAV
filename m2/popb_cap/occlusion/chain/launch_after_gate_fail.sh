#!/usr/bin/env bash
# 2026-10-10 16:40: manual launch after the gate FAIL on GW250109_010541 (DECISIONS_PLACEMENT.md). Same steps as crop_controller.sh after PASS.
C=${MADGRAV_ROOT}/m2/popb_cap/occlusion/chain; PY=${PY:-python}; MAILER=${MADGRAV_ROOT}/search_mode/pastro_final/campaign/mailer.py
LOG=$C/logs/crop_controller.log; echo "[ctl] $(date '+%F %T') manual launch after gate FAIL (decision 16:40 in DECISIONS_PLACEMENT.md)" >> $LOG
for r in o3a o3b o4a o4b; do DRY=0 RUN=$r $C/crop_chain.sh >> $LOG 2>&1 || echo "[ctl] chain $r launch FAILED" >> $LOG; done
{ echo "Gate FAIL on GW250109_010541 only (0.463 -> 1.039/yr); chains launched by decision 2026-10-10 16:40."; echo; cat $C/GATE_RESULT.txt; echo; echo "chains launched:"; grep -h "^CNN_SHARDS\|^READOUT" ${MADGRAV_ROOT}/launchers/logs_m2/*_off5_crop_m2_gx_jids.env; } > $C/logs/launch_mail.txt
$PY $MAILER "[MADGRAV2 crop] 4 crop chains launched after gate FAIL (GW250109 only)" $C/logs/launch_mail.txt
nohup $PY -I $C/crop_mailer.py > $C/logs/crop_mailer.out 2>&1 &
echo "[ctl] $(date '+%F %T') mailer pid $! ; manual launch done" >> $LOG
grep -E "crop-chain|FAILED" $LOG | tail -8; squeue -u USER -o "%.10i %.28j %.9T" | head -60
