#!/bin/bash
#
# Queues a detr training job per corpus and model size. Every job notes itself in
# `jobs.txt` as it starts, and every training job queues its own scoring job when it finishes.
#
#   runs/queue_detr.sh              # every configuration below, sharing one OUTPUT_DIR
#   runs/queue_detr.sh l-shape xs   # one configuration, in an OUTPUT_DIR of its own
#

set -euo pipefail

MODEL=detr
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.detr-cache/output/$(date +%Y%m%d_%H%M%S)}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.detr-cache/corpora}"

if [[ $# -eq 2 ]]; then
  CORPUS="$1"
  SIZE="$2"
  export CORPUS SIZE
  mkdir -p "$OUTPUT_DIR"

  jobId="$(sbatch --parsable --job-name="$MODEL-$CORPUS-$SIZE-train" "runs/$MODEL.sh" train)"
  echo "queued $MODEL on $CORPUS at size $SIZE as $jobId: metrics in $OUTPUT_DIR"
  exit 0
fi

[[ $# -eq 0 ]] || { echo "usage: runs/queue_$MODEL.sh [<corpus> <size>]" >&2; exit 2; }

read -r -p "queue every $MODEL configuration listed below, as a job each? [y/N] " answer || true
[[ $answer == y ]] || { echo "nothing queued"; exit 1; }

# Paste a line to queue that one run.
runs/queue_detr.sh l-shape xs
runs/queue_detr.sh l-shape s
runs/queue_detr.sh rectilinear xs
runs/queue_detr.sh rectilinear s
runs/queue_detr.sh sketch xs
runs/queue_detr.sh sketch s
