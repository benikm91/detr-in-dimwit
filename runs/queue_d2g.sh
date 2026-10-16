#!/bin/bash
#
# Queues a d2g training job on the synthetic floor plans per model size. Every job notes itself in
# `jobs.txt` as it starts. Every training job queues its scoring on the validation split when it finishes, and that
# queues the scoring of the last checkpoint on the test split.
#
#   runs/queue_d2g.sh      # every configuration below, sharing one OUTPUT_DIR
#   runs/queue_d2g.sh s    # one configuration, in an OUTPUT_DIR of its own
#

set -euo pipefail

MODEL=d2g
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.detr-cache/output/$(date +%Y%m%d_%H%M%S)}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.detr-cache/corpora}"

if [[ $# -eq 1 ]]; then
  CORPUS=synthetic-floorplans
  SIZE="$1"
  export SIZE
  mkdir -p "$OUTPUT_DIR"

  # Enough GPUs that a device's share of the batch fits in memory, and enough time for the steps.
  gpus=2; time=10:00:00
  jobId="$(sbatch --parsable --gres=gpu:$gpus --time=$time --job-name="$MODEL-$CORPUS-$SIZE-train" "runs/$MODEL.sh" train)"
  echo "queued $MODEL on $CORPUS at size $SIZE as $jobId: metrics in $OUTPUT_DIR"
  exit 0
fi

[[ $# -eq 0 ]] || { echo "usage: runs/queue_$MODEL.sh [<size>]" >&2; exit 2; }

read -r -p "queue every $MODEL configuration listed below, as a job each? [y/N] " answer || true
[[ $answer == y ]] || { echo "nothing queued"; exit 1; }

# Paste a line to queue that one run.
runs/queue_d2g.sh s
runs/queue_d2g.sh s-deep
