#!/usr/bin/env bash
# Score the unified inj_full campaign and produce the mass-SNR recovery figure.
# Tag _x1cnnfullveto. Nothing adopted is touched: the adopted artifacts carry the
# _x1cnnadopt48fveto tag and both code gates (SM_SNR_GRID_EXT, SM_VETO_NOREF) default off.
# The p_astro / VT side outputs are marked INVALID on the way out: inj_full spans 13 SNR
# levels, so the rho^-4 population weights differ from the adopted grid and every WEIGHTED
# quantity is invalid. The figure bins det_frac with no weights and is unaffected.
set -uo pipefail
export MADGRAV_ROOT=${MADGRAV_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}; cd "$MADGRAV_ROOT"
PY=${PY:-python}
MAILER=search_mode/pastro_final/campaign/mailer.py
PF=search_mode/pastro_final
D=${MADGRAV_SCRATCH:-$MADGRAV_ROOT/scratch}/mtot_snr_sim_experiment
LOG=$MADGRAV_ROOT/launchers/injfull_score.log
export OMP_NUM_THREADS=4 MPLBACKEND=Agg
KE=$MADGRAV_ROOT/details/successor_statistic/ke_adopted.json
export SM_BANK_LM="$MADGRAV_ROOT/data/lowmass_bank_v1/projectedbank__mtot_10.0_22.0__qmax_6.0__fs_4096__minFreq_10.0__apx_IMRPhenomPv2"
TAG=x1cnnfullveto

fail () {
  echo "[full] FAILED at: $1"
  { echo "The unified inj_full scoring chain failed at step: $1"
    echo "No figure attached -- a partial or mis-scored campaign must not look like a result."
    echo; echo "The injection stage DID succeed: 110/110 tasks, 321,750 injections,"
    echo "0 bad files, strata 33.9/33.1/33.0%, Mtot 10-400, SNR 5-80."
    echo; echo "Last 60 lines:"; echo "----------------------------------------"
    tail -60 "$LOG" 2>/dev/null || echo "(log unavailable)"
  } > "$D/full_fail.txt"
  $PY "$MAILER" "MADGRAV inj_full scoring FAILED at: $1" "$D/full_fail.txt" || true
  exit 1
}

echo "[full] start $(date '+%F %T') host=$(hostname)"
echo "[full] 1/5 pastro_final on inj_full -> _x1cnnfull"
SM_SNR_GRID_EXT=30,40,60,80 \
SM_TRIALS_OFF=1 SM_INJ_CNN_GATE=1 SM_LR_ONLY=1 SM_NETMAX=10.6 SM_KE=$KE \
SM_INJ_DIR=inj_full SM_SUF_EXTRA=full \
SM_ADOPTED_AXIS=1 SM_AXIS_COL=far_excl SM_ADMIT=all SM_DET_RULE=far \
  nice -n 5 $PY $PF/pastro_final.py || fail "pastro_final"

echo "[full] 2/5 build_inj_veto"
SM_VETO_SRC=inj_full SM_VETO_SUF=_x1cnnfull SM_VETO_NOREF=1 \
  nice -n 5 $PY $PF/build_inj_veto.py || fail "build_inj_veto"

echo "[full] 3/5 vt_relabel_comoving (source-frame masses; lm bank registered)"
SM_VT_SUF=_${TAG} SM_INJ_CAMPAIGN=inj_full SM_VT_FIXED_ASD=1 SM_ASD_TAG=fixed2 \
  nice -n 5 $PY $PF/vt_relabel_comoving.py || fail "vt_relabel_comoving"

echo "[full] 4/5 mark the weighted side outputs INVALID (before anything can read them)"
for f in $PF/pastro_final_x1cnnfull.json $PF/pastro_final_x1cnnfull.csv \
         $PF/vt_relabel_comoving_${TAG}.json; do
  [ -f "$f" ] && mv "$f" "${f%.*}_INVALID_popweights_figonly.${f##*.}" && echo "  marked $(basename $f)"
done

echo "[full] 5/5 figure + email"
MG_FIG_TAG=$TAG nice -n 5 $PY $D/plot_mtot_snr_smooth.py || fail "figure"
MG_FIG_TAG=$TAG $PY $D/make_body_snrext.py > "$D/body_full.txt" || fail "body"
$PY "$MAILER" "MADGRAV mass vs SNR -- unified campaign, Mtot 10-400 and SNR 5-80 measured" \
     "$D/body_full.txt" "$D/mtot_snr_smooth.pdf" "$D/mtot_snr_smooth.png" || fail "mail"
echo "[full] DONE $(date '+%F %T')"
