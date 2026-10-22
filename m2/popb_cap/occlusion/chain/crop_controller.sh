#!/usr/bin/env bash
# Unattended controller (2026-10-10): wait for the background jobs 1811274 (loglr bank) and 1811300 (net-sigma), run the gate; on PASS launch the
# four crop chains (DRY=0) and start the mailer; on FAIL mail and stop. Light login-node process (nohup). Log: chain/logs/crop_controller.log
C=${MADGRAV_ROOT}/m2/popb_cap/occlusion/chain; PY=${PY:-python}; MAILER=${MADGRAV_ROOT}/search_mode/pastro_final/campaign/mailer.py
LOG=$C/logs/crop_controller.log; echo "$$" > $C/logs/crop_controller.pid; echo "[ctl] $(date '+%F %T') start" >> $LOG
while [ -n "$(squeue -h -j 1811274,1811300 2>/dev/null)" ]; do sleep 300; done
echo "[ctl] $(date '+%F %T') background jobs finished: $(sacct -j 1811274,1811300 -X -n -o JobID,State | tr '\n' ' ')" >> $LOG
cd $C/.. && $PY -I $C/crop_gate.py > $C/logs/gate_stdout.txt 2>&1; rc=$?; echo "[ctl] gate rc=$rc" >> $LOG
if [ $rc -ne 0 ]; then $PY $MAILER "[MADGRAV2 crop] GATE FAILED -- chains NOT launched" $C/GATE_RESULT.txt; echo "[ctl] stop" >> $LOG; exit 1; fi
for r in o3a o3b o4a o4b; do DRY=0 RUN=$r $C/crop_chain.sh >> $LOG 2>&1 || echo "[ctl] chain $r launch FAILED" >> $LOG; done
{ cat $C/GATE_RESULT.txt; echo; echo "chains launched:"; grep -h "^CNN_SHARDS\|^READOUT" ${MADGRAV_ROOT}/launchers/logs_m2/*_off5_crop_m2_gx_jids.env; } > $C/logs/launch_mail.txt
$PY $MAILER "[MADGRAV2 crop] GATE PASS -- 4 crop chains launched" $C/logs/launch_mail.txt
nohup $PY -I $C/crop_mailer.py > $C/logs/crop_mailer.out 2>&1 &
echo "[ctl] $(date '+%F %T') mailer pid $! ; controller exit" >> $LOG
