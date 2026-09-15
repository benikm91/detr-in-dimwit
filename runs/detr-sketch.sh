#!/bin/bash
#SBATCH --job-name=detr-sketch
#SBATCH --partition=gpu_top_ia
#SBATCH --account=cai_cv
#SBATCH --gres=gpu:2
#SBATCH --exclude=sanjose,irvine
#SBATCH --time=2:00:00
#SBATCH --output=/cluster/home/%u/.logs/slurm/%j/detr-sketch_%j.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/%j/detr-sketch_%j.err
#
# Trains the detector on a corpus and scores it, on however many GPUs the job was given.
#
#   INPUT_DIR=/cluster/scratch/$USER/corpora \
#   OUTPUT_DIR=/cluster/scratch/$USER/runs \
#     sbatch runs/detr-sketch.sh
#
# The instance this runs on is wiped when the job ends, so both directories have to outlive it:
# INPUT_DIR is what the next run does not have to download again, and OUTPUT_DIR is what is left
# of this one. Everything under them is the code's own doing — a corpus makes itself a folder to
# cache in, and a run makes itself a folder named for when it started.

set -euo pipefail

: "${INPUT_DIR:?set INPUT_DIR to a directory that outlives the run, where the corpora are cached}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a directory that outlives the run, where runs are kept}"

# Which setup to run: the mains are `detr${SETUP}Train` and `detr${SETUP}Eval`, and CORPUS is what
# `dataset/prepare.sh` calls the corpus that setup reads.
SETUP="${SETUP:-Sketches}"
CORPUS="${CORPUS:-sketches}"

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
echo "running $SETUP: corpora in $INPUT_DIR, runs in $OUTPUT_DIR"

module load sarus/1.6.4

IMAGE="benikm91/dimwit-gpu:snapshot"
sarus pull "$IMAGE"

srun sarus run \
  --mount=type=bind,source="$INPUT_DIR",destination=/input \
  --mount=type=bind,source="$OUTPUT_DIR",destination=/output \
  "$IMAGE" \
  bash -c '
    set -euo pipefail
    setup="$1"
    corpus="$2"

    export TMPDIR=/tmp
    export INPUT_DIR=/input
    export OUTPUT_DIR=/output

    # DimWit and DeepWit from source. The sharding the training script splits its batch over is on
    # a branch of DimWit rather than in a release.
    cd /usr/src
    git clone https://github.com/dimwit-dev/dimwit
    git clone https://github.com/dimwit-dev/deepwit
    git clone https://github.com/dimwit-dev/dimwit-sharding
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
    ./dataset/prepare.sh "$corpus"
    sbt "detr/runMain detr${setup}Train"
    sbt "detr/runMain detr${setup}Eval"
  ' detr-sketch "$SETUP" "$CORPUS"

echo "run finished, newest run in $OUTPUT_DIR:"
ls -lat "$OUTPUT_DIR" | head -3
