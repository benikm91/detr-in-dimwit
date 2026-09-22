#!/bin/bash
#SBATCH --export=ALL,SARUS_HOME=TRUE
#SBATCH --job-name=egtr
#SBATCH --partition=gpu
#SBATCH --account=cai_cv
#SBATCH --gres=gpu:4
#SBATCH --exclude=sanjose,irvine,salinas
#SBATCH --time=6:00:00
#SBATCH --output=/cluster/home/%u/.logs/slurm/%j/%x_%j.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/%j/%x_%j.err
#
# Trains the scene graph model on a corpus, scores every checkpoint, and leaves the metrics as one
# CSV. The detector underneath is trained with it, from scratch. The batch is split over the GPUs
# the job gets, so `--gres=gpu:N` sets how much of it each GPU holds, not how big it is.
# `runs/queue_egtr.sh` asks for as many as the model size needs.
#
#   CORPUS=l-shape SIZE=s \
#   CACHE_DIR=/cluster/scratch/$USER/corpora \
#   OUTPUT_DIR=/cluster/scratch/$USER/metrics \
#     sbatch runs/egtr.sh train
#
# `runs/queue_egtr.sh` is the usual way in; `sbatch` hands the environment on to the job. The job
# builds the branch this repository has checked out, as pushed to GitHub — a change that is not
# pushed is not run.
# CACHE_DIR is where the corpora land, so that the next job does not download them again.
# OUTPUT_DIR is where `egtr-<corpus>-<size>.csv` ends up: what is left of the job once the
# instance is wiped. CHECKPOINT_DIR is where the checkpoints go, under OUTPUT_DIR unless it is set
# otherwise: the scoring job reads them back, and they are yours to delete once it has.

set -euo pipefail

STAGE="${1:?say which stage to run: train or eval}"
[[ $STAGE == train || $STAGE == eval ]] || { echo "no stage named '$STAGE': train or eval" >&2; exit 2; }

: "${CACHE_DIR:?set CACHE_DIR to a directory that outlives the job, where the corpora are cached}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a directory that outlives the job, where the metrics are written}"
export CACHE_DIR OUTPUT_DIR
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"

# Which corpus to train on and how big a model — the names `Corpus` and `EGTR.Size` know.
export CORPUS="${CORPUS:-sketch}"
export SIZE="${SIZE:-s}"

# Inherited by the scoring job this one queues, so both build the same branch.
export BRANCH="${BRANCH:-$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --abbrev-ref HEAD)}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
echo "running egtr $STAGE on $CORPUS at size $SIZE from branch $BRANCH: corpora in $CACHE_DIR, checkpoints in $CHECKPOINT_DIR, metrics in $OUTPUT_DIR"

# Which run this job is, for looking its id up later by what it ran and where it wrote.
printf '%s\tegtr\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CORPUS" "$SIZE" "$STAGE" "$OUTPUT_DIR" "${SLURM_JOB_ID:-none}" \
  >>"${SLURM_SUBMIT_DIR:-$PWD}/jobs.txt"

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
    model="$0"
    corpus="$1"
    size="$2"
    stage="$3"
    slurmJobId="$4"
    node="$5"
    branch="$6"

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
    git clone --branch "$branch" https://github.com/benikm91/detr-in-dimwit

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

    # What produced the metrics beside it: the commits every part of the stack was built from, the
    # JAX that ran them, and the GPUs they ran on. Written before training, so that a run cut short
    # still says what it was.
    if [[ $stage == train ]]; then
      jaxVersion="$(uv run python -c "import jax; print(jax.__version__)" 2>/dev/null || echo unknown)"
      gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ";" - || true)"
      [[ -n $gpus ]] || gpus=unknown
      cat >"/output/$model-$corpus-$size.environment.json" <<JSON
{
  "model": "$model",
  "corpus": "$corpus",
  "size": "$size",
  "slurmJobId": "$slurmJobId",
  "node": "$node",
  "gpus": "$gpus",
  "jax": "$jaxVersion",
  "branch": "$branch",
  "commits": {
    "detr-in-dimwit": "$(git -C /usr/src/detr-in-dimwit rev-parse HEAD)",
    "dimwit": "$(git -C /usr/src/dimwit rev-parse HEAD)",
    "deepwit": "$(git -C /usr/src/deepwit rev-parse HEAD)",
    "dimwit-sharding": "$(git -C /usr/src/dimwit-sharding rev-parse HEAD)"
  }
}
JSON
      cat "/output/$model-$corpus-$size.environment.json"
    fi

    case "$stage" in
      train) sbt "egtr/runMain egtrTrain $corpus $size" ;;
      eval) sbt "egtr/runMain egtrEval $corpus $size" ;;
    esac
  ' egtr "$CORPUS" "$SIZE" "$STAGE" "${SLURM_JOB_ID:-none}" "${SLURMD_NODENAME:-$(hostname)}" "$BRANCH"

if [[ $STAGE == train ]]; then
  echo "job finished, checkpoints in $CHECKPOINT_DIR"
  # Scoring is queued from here, so that it reads the checkpoints this run just wrote and runs only
  # if there are any. One GPU is enough: it scores one checkpoint at a time.
  evalId="$(sbatch --parsable --gres=gpu:1 --time=8:00:00 --job-name="egtr-$CORPUS-$SIZE-eval" runs/egtr.sh eval)"
  echo "queued scoring as $evalId"
else
  echo "job finished, metrics in $OUTPUT_DIR:"
  ls -la "$OUTPUT_DIR"/egtr-"$CORPUS"-"$SIZE".csv
fi
