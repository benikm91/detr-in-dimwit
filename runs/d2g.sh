#!/bin/bash
#SBATCH --export=ALL,SARUS_HOME=TRUE
#SBATCH --job-name=d2g
#SBATCH --partition=gpu
#SBATCH --account=cai_cv
#SBATCH --gres=gpu:4
#SBATCH --exclude=irvine,salinas
#SBATCH --time=24:00:00
#SBATCH --output=/cluster/home/%u/.logs/slurm/%j/%x_%j.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/%j/%x_%j.err
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --open-mode=append
#
# Trains the graph transcriber on the synthetic floor plans, scores every checkpoint, and leaves the
# metrics as one CSV.
# The batch is split over the GPUs the job gets, so `--gres=gpu:N` sets how much of it each GPU
# holds, not how big it is. `runs/queue_d2g.sh` asks for as many as the model size needs.
#
#   SIZE=s \
#   CACHE_DIR=$HOME/.detr-cache/corpora \
#   NODES_DIR=$HOME/.detr-cache/output/<d2s run name>/checkpoints \
#   OUTPUT_DIR=$HOME/.detr-cache/output/<run name> \
#     sbatch runs/d2g.sh train
#
# A run of `train` is fine-tuned, every part of it, by the stage of its own, from one of its
# checkpoints; it is scored by eval and test as any other:
#
#   FINETUNE_FROM=$HOME/.detr-cache/output/<run name>/checkpoints/d2g/<...>/<step>.pkl \
#   SIZE=s CACHE_DIR=... OUTPUT_DIR=$HOME/.detr-cache/output/<new run name> \
#     sbatch runs/d2g.sh finetune
#
# `runs/queue_d2g.sh` is the usual way in; `sbatch` hands the environment on to the job.
# CACHE_DIR is where the floor plans land, so that the next job does not download them again.
# OUTPUT_DIR is where `d2g-synthetic-floorplans-<size>-<split>.csv` ends up: what is left of the job once the
# instance is wiped. CHECKPOINT_DIR is where the checkpoints go, under OUTPUT_DIR unless it is set
# otherwise: the scoring job reads them back, and they are yours to delete once it has.
# NODES_DIR holds the checkpoints of the d2s run, of the same size and order, whose nodes the
# relationships are trained on; its weights stay as they are.

set -euo pipefail

STAGE="${1:?say which stage to run: train, finetune, eval, test or draw}"
[[ $STAGE == train || $STAGE == finetune || $STAGE == eval || $STAGE == test || $STAGE == draw ]] || { echo "no stage named '$STAGE': train, finetune, eval, test or draw" >&2; exit 2; }

: "${CACHE_DIR:?set CACHE_DIR to a directory that outlives the job, where the corpora are cached}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a directory that outlives the job, where the metrics are written}"
export CACHE_DIR OUTPUT_DIR
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
# Each training stage reads a run that is already there, mounted into the container.
mounts=()
if [[ $STAGE == train ]]; then
  : "${NODES_DIR:?set NODES_DIR to the checkpoints of the d2s run that writes the nodes}"
  mounts+=(--mount=type=bind,source="$NODES_DIR",destination=/nodes,readonly)
fi
if [[ $STAGE == finetune ]]; then
  : "${FINETUNE_FROM:?set FINETUNE_FROM to the checkpoint of the d2g run to fine-tune}"
  mounts+=(--mount=type=bind,source="$(dirname "$FINETUNE_FROM")",destination=/finetuned,readonly)
fi

# The one corpus d2g reads, and how big a model — the names `D2SModelConfiguration` knows.
export CORPUS=synthetic-floorplans
export SIZE="${SIZE:-s}"

# Which seed a run trains from, so that a setting can be run again: SEED=1 for a repeat.
export SEED="${SEED:-0}"

# Which order a run writes the nodes of a record in — the names `d2s.floorplan.NodeOrder.named`
# knows: free, walls-first or walls-last.
export ORDER="${ORDER:-free}"

# Which branch the container builds, as pushed to GitHub — a change that is not pushed is not
# run. Inherited by the scoring job this one queues, so both build the same one.
export BRANCH="${BRANCH:-main}"

mkdir -p "$CACHE_DIR" "$OUTPUT_DIR" "$CHECKPOINT_DIR"
echo "running d2g $STAGE on $CORPUS at size $SIZE, order $ORDER, seed $SEED, from branch $BRANCH: corpora in $CACHE_DIR, checkpoints in $CHECKPOINT_DIR, metrics in $OUTPUT_DIR"

# Which run this job is, for looking its id up later by what it ran and where it wrote.
printf '%s\td2g\t%s\t%s\t%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$CORPUS" "$SIZE" "$STAGE" "$OUTPUT_DIR" "${SLURM_JOB_ID:-none}" \
  >>"${SLURM_SUBMIT_DIR:-$PWD}/jobs.txt"

# A run longer than one job puts itself back in the queue shortly before Slurm would kill it.
# What starts again reads the newest checkpoint of the same folder and takes the steps that
# are left, so how long a run is is set by its step count and not by what fits in a job.
requeued=no
continueLater() {
  requeued=yes
  echo "time limit approaching: putting ${SLURM_JOB_ID:-this job} back in the queue to carry on"
  scontrol requeue "$SLURM_JOB_ID"
}
# Any other stage has nothing to carry on from, so it ignores the warning and runs to the limit.
if [[ $STAGE == train || $STAGE == finetune ]]; then trap continueLater USR1; else trap '' USR1; fi

module load sarus/1.6.4

# DimWit, DeepWit, PlotWit and dimwit-sharding come published inside the image, at the versions
# build.sbt asks for. A new release of any of them needs a new image.
IMAGE="benikm91/dimwit-gpu:kv-cache-control-flow"
sarus pull "$IMAGE"

sarus run \
  --mount=type=bind,source="$CACHE_DIR",destination=/cache \
  --mount=type=bind,source="$OUTPUT_DIR",destination=/output \
  --mount=type=bind,source="$CHECKPOINT_DIR",destination=/checkpoints \
  "${mounts[@]}" \
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
    seed="$7"
    order="$8"
    finetuneFrom="$9"

    export TMPDIR=/tmp

    # The corpora download themselves from the Hub on first use; this is where they land, so
    # that the next job finds them instead of fetching them again.
    export HF_HUB_CACHE=/cache/huggingface-cache
    # Where a run keeps its checkpoints, and where scoring them writes the CSV.
    export CHECKPOINT_DIR=/checkpoints
    export NODES_DIR=/nodes
    export OUTPUT_DIR=/output

    cd /usr/src
    git clone --branch "$branch" https://github.com/benikm91/detr-in-dimwit
    cd detr-in-dimwit

    # The image points DimWit at its own Python. Ours comes from pyproject.toml instead: DimWit runs uv sync and uses the venv that gives.
    unset DIMWIT_SKIP_SYNC DIMWIT_PYTHON_PATH DIMWIT_PYTHON_LIBRARY

    # What produced the metrics beside it: the commits every part of the stack was built from, the
    # JAX that ran them, and the GPUs they ran on. Written before training, so that a run cut short
    # still says what it was.
    if [[ $stage == train || $stage == finetune ]]; then
      jaxVersion="$(uv run python -c "import jax; print(jax.__version__)" 2>/dev/null || echo unknown)"
      gpus="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ";" - || true)"
      [[ -n $gpus ]] || gpus=unknown
      cat >"/output/$model-$corpus-$size.environment.json" <<JSON
{
  "model": "$model",
  "corpus": "$corpus",
  "size": "$size",
  "seed": "$seed",
  "order": "$order",
  "finetuneFrom": "$finetuneFrom",
  "slurmJobId": "$slurmJobId",
  "node": "$node",
  "gpus": "$gpus",
  "jax": "$jaxVersion",
  "branch": "$branch",
  "commits": {
    "detr-in-dimwit": "$(git -C /usr/src/detr-in-dimwit rev-parse HEAD)",
    "dimwit": "$(git -C /opt/wit/src/dimwit rev-parse HEAD)",
    "deepwit": "$(git -C /opt/wit/src/deepwit rev-parse HEAD)",
    "plotwit": "$(git -C /opt/wit/src/plotwit rev-parse HEAD)",
    "dimwit-sharding": "$(git -C /opt/wit/src/dimwit-sharding rev-parse HEAD)"
  }
}
JSON
      cat "/output/$model-$corpus-$size.environment.json"
    fi

    case "$stage" in
      train) sbt "d2g/runMain d2gTrain $size $order $seed" ;;
      finetune) sbt "d2g/runMain d2gFineTune $size $order $seed /finetuned/$(basename "$finetuneFrom")" ;;
      eval) sbt "d2g/runMain d2gEval $size" ;;
      test) sbt "d2g/runMain d2gTest $size" ;;
      draw) sbt "d2g/runMain d2gDraw $size" ;;
    esac
  ' d2g "$CORPUS" "$SIZE" "$STAGE" "${SLURM_JOB_ID:-none}" "${SLURMD_NODENAME:-$(hostname)}" "$BRANCH" "$SEED" "$ORDER" "${FINETUNE_FROM:-}" &

wait $! || trained=$?

if [[ $requeued == yes ]]; then
  echo "stopped at the time limit; the rest of this run is queued"
  exit 0
fi
[[ ${trained:-0} -eq 0 ]] || exit "${trained:-0}"

# A run goes train, eval, test: each stage queues the next once it has finished, so that the next
# reads what this one wrote, and runs only if it got that far. One GPU is enough for scoring: it
# scores one checkpoint at a time. The memory this job was given is not handed on, since the next
# stage may land on a node that has less.
queueNext() {
  nextId="$(env -u SLURM_MEM_PER_NODE sbatch --parsable --gres=gpu:1 --time=24:00:00 --job-name="d2g-$CORPUS-$SIZE-$1" runs/d2g.sh "$1")"
  echo "queued $1 as $nextId"
}
case "$STAGE" in
  train | finetune)
    echo "job finished, checkpoints in $CHECKPOINT_DIR"
    queueNext eval
    ;;
  eval)
    echo "job finished, metrics in $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR"/d2g-"$CORPUS"-"$SIZE"-*.csv
    queueNext test
    ;;
  *)
    echo "job finished, metrics in $OUTPUT_DIR:"
    ls -la "$OUTPUT_DIR"/d2g-"$CORPUS"-"$SIZE"-*.csv
    ;;
esac
