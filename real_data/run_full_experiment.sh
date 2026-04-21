#!/usr/bin/env bash
set -e

SECONDS=0

PYTHON_BIN="${PYTHON_BIN:-python}"

current_pid=""

cleanup() {
  trap - EXIT INT TERM
  if [ -n "$current_pid" ]; then
    kill -INT "$current_pid" 2>/dev/null || true
    wait "$current_pid" 2>/dev/null || true
  fi
}

interrupt() {
  echo
  echo "Interrupted. Stopping current experiment step..." >&2
  cleanup
  exit 130
}

run_command() {
  echo ">>> $*"
  "$@" &
  current_pid=$!
  if wait "$current_pid"; then
    current_pid=""
  else
    status=$?
    current_pid=""
    exit "$status"
  fi
}

trap cleanup EXIT
trap interrupt INT TERM

PART_FEATURES=(
  # Single feature groups
  MI
  PP
  TS

  # Two-group combinations
  MI_PP
  MI_TS
  PP_TS

  # Three-group combination
  MI_PP_TS
)


echo
echo "============================================================"
echo "Step 1/4: Training all models"
echo "============================================================"
run_command bash run_train.sh

echo
echo "============================================================"
echo "Step 2/4: Aggregate optimal metrics"
echo "============================================================"
run_command "${PYTHON_BIN}" main_optimal_metrics.py --rep_start 1 --rep_end 30

echo
echo "============================================================"
echo "Step 3/4: Fig6 Radar plots"
echo "============================================================"
run_command "${PYTHON_BIN}" fig6_radar_plot_enhanced.py --style style2

echo
echo "============================================================"
echo "Step 4/4: Fig7 Single-model visualization"
echo "============================================================"
run_command "${PYTHON_BIN}" fig7_visualize_single_model.py \
  --feature_name MI_PP_TS_dim66 \
  --model_type Deep \
  --rep_id 1 \
  --num_reps 1

total=$SECONDS
printf "\nFull experiment completed. Total elapsed: %02d:%02d:%02d\n" \
  $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
