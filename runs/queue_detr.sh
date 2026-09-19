#!/bin/bash
#
# Queues a detr job per corpus and model size, noting each SLURM id in `jobs.txt`.
#
#   runs/queue_detr.sh              # every configuration below, sharing one OUTPUT_DIR
#   runs/queue_detr.sh l-shape xs   # one configuration, in an OUTPUT_DIR of its own
#

set -euo pipefail

MODEL=detr
export OUTPUT_DIR="${OUTPUT_DIR:-$HOME/.detr-cache/output/$(date +%Y%m%d_%H%M%S)}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.detr-cache/corpora}"
JOBS_FILE=jobs.txt

if [[ $# -eq 2 ]]; then
  CORPUS="$1"
  SIZE="$2"
  export CORPUS SIZE
  mkdir -p "$OUTPUT_DIR"
  jobId="$(sbatch --parsable --job-name="$MODEL-$CORPUS-$SIZE" "runs/$MODEL.sh")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$MODEL" "$CORPUS" "$SIZE" "$OUTPUT_DIR" "$jobId" >>"$JOBS_FILE"
  echo "queued $MODEL on $CORPUS at size $SIZE as $jobId: metrics in $OUTPUT_DIR, noted in $JOBS_FILE"
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
