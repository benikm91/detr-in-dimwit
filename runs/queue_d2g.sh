#!/bin/bash
#
# Queues a d2g training job per corpus and model size. Every job notes itself in
# `jobs.txt` as it starts, and every training job queues its own scoring job when it finishes.
#
#   runs/queue_d2g.sh              # every configuration below, sharing one OUTPUT_DIR
#   runs/queue_d2g.sh l-shape xs   # one configuration, in an OUTPUT_DIR of its own
#
#   CHECKPOINT_DIR=$HOME/.detr-cache/output/<run>/checkpoints \
#     runs/queue_d2g.sh rectilinear s-deep 128000   # that run taken up again at step 128000
#

set -euo pipefail

MODEL=d2g
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.detr-cache/output/$(date +%Y%m%d_%H%M%S)}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.detr-cache/corpora}"

if [[ $# -eq 2 || $# -eq 3 ]]; then
  CORPUS="$1"
  SIZE="$2"
  export CORPUS SIZE
  mkdir -p "$OUTPUT_DIR"

  # Enough GPUs that a device's share of the batch fits in memory, and enough time for the steps.
  case $SIZE in
    s-deep) gpus=4; hours=12 ;;
    *) gpus=2; hours=6 ;;
  esac
  stage=train
  if [[ $# -eq 3 ]]; then
    : "${CHECKPOINT_DIR:?set CHECKPOINT_DIR to the checkpoints of the run to continue}"
    export CHECKPOINT_DIR FROM_STEP="$3"
    stage=continue
    hours=$((hours * 3 / 2))
  fi
  jobId="$(sbatch --parsable --gres=gpu:$gpus --time=$hours:00:00 --job-name="$MODEL-$CORPUS-$SIZE-$stage" "runs/$MODEL.sh" $stage)"
  echo "queued $MODEL $stage on $CORPUS at size $SIZE as $jobId: metrics in $OUTPUT_DIR"
  exit 0
fi

[[ $# -eq 0 ]] || { echo "usage: runs/queue_$MODEL.sh [<corpus> <size> [<step to continue from>]]" >&2; exit 2; }

read -r -p "queue every $MODEL configuration listed below, as a job each? [y/N] " answer || true
[[ $answer == y ]] || { echo "nothing queued"; exit 1; }

# Paste a line to queue that one run.
runs/queue_d2g.sh l-shape xs
runs/queue_d2g.sh l-shape s
runs/queue_d2g.sh rectilinear xs
runs/queue_d2g.sh rectilinear s
runs/queue_d2g.sh sketch xs
runs/queue_d2g.sh sketch s
