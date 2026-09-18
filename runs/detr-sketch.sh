#!/bin/bash
#SBATCH --export=ALL,SARUS_HOME=TRUE
#SBATCH --job-name=detr-sketch
#SBATCH --partition=gpu
#SBATCH --account=cai_cv
#SBATCH --gres=gpu:a100:2
#SBATCH --exclude=sanjose,irvine
#SBATCH --time=4:00:00
#SBATCH --output=/cluster/home/%u/.logs/slurm/%j/detr-sketch_%j.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/%j/detr-sketch_%j.err
#
# Trains the detector on a corpus, scores every checkpoint, and leaves the metrics as one CSV.
#
#   source runs/corpus_lshape.env; source runs/model_s.env
#   CACHE_DIR=/cluster/scratch/$USER/corpora \
#   OUTPUT_DIR=/cluster/scratch/$USER/metrics \
#     sbatch runs/detr-sketch.sh
#
# The `.env` files in `runs/` set CORPUS and SIZE; `sbatch` hands the environment on to the job.
# CACHE_DIR is where the corpora land, so that the next job does not download them again.
# OUTPUT_DIR is where `detr-<corpus>-<size>.csv` ends up: what is left of the job once the
# instance is wiped. CHECKPOINT_DIR is where the checkpoints go meanwhile, which need not
# outlive the job.

set -euo pipefail

: "${CACHE_DIR:?set CACHE_DIR to a directory that outlives the job, where the corpora are cached}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a directory that outlives the job, where the metrics are written}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch}"

# Which corpus to train on and how big a model — the names `Corpus` and `DETR.Size` know.
CORPUS="${CORPUS:-sketch}"
SIZE="${SIZE:-s}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
echo "running detr on $CORPUS at size $SIZE: corpora in $CACHE_DIR, checkpoints in $CHECKPOINT_DIR, metrics in $OUTPUT_DIR"

module load sarus/1.6.4

IMAGE="benikm91/dimwit-gpu:snapshot"
sarus pull "$IMAGE"

sarus run \
  --mount=type=bind,source="$CACHE_DIR",destination=/cache \
  --mount=type=bind,source="$OUTPUT_DIR",destination=/output \
  --mount=type=bind,source="$CHECKPOINT_DIR",destination=/checkpoints \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    corpus="$1"
    size="$2"

    export TMPDIR=/tmp

    # The corpora download themselves from the Hub on first use; this is where they land, so
    # that the next job finds them instead of fetching them again.
    export HF_HUB_CACHE=/cache/huggingface-cache
    # Where a run keeps its checkpoints, and where scoring them writes the CSV.
    export CHECKPOINT_DIR=/checkpoints
    export OUTPUT_DIR=/output

    # DimWit and DeepWit from source. The sharding the training script splits its batch over is on
    # a branch of DimWit rather than in a release.
    cd /usr/src
    git clone https://github.com/dimwit-dev/dimwit
    git clone https://github.com/dimwit-dev/deepwit
    git clone https://github.com/benikm91/dimwit-sharding
    git clone https://github.com/benikm91/detr-in-dimwit

    cd dimwit
    sbt publishLocal
    cd ..

    cd deepwit
    sbt publishLocal
    cd ..

    cd dimwit-sharding
    sbt publishLocal
    cd ..

    cd detr-in-dimwit

    # The image points DimWit at its own Python. Ours comes from pyproject.toml instead: DimWit runs uv sync and uses the venv that gives.
    unset DIMWIT_SKIP_SYNC DIMWIT_PYTHON_PATH DIMWIT_PYTHON_LIBRARY

    sbt "detr/runMain detrTrain $corpus $size"
    sbt "detr/runMain detrEval $corpus $size"
  ' detr-sketch "$CORPUS" "$SIZE"

echo "job finished, metrics in $OUTPUT_DIR:"
ls -la "$OUTPUT_DIR"/detr-"$CORPUS"-"$SIZE".csv
