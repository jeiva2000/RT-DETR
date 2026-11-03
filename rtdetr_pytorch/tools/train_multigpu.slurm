#!/bin/bash
#SBATCH --job-name=rtdetr_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a40:1             # 1x A40 en gpu04 (libre ahora mismo)
#SBATCH --nodelist=gpu04             # comprueba antes con: scontrol show node gpu04
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --export=NONE

set -euo pipefail

# No confiar en variables heredadas de la shell que lanzó el sbatch (login node
# vs sesión interactiva pueden tener MODULEPATH distinto); reinicializar aquí,
# ya en el propio nodo de cómputo.
source /etc/profile

mkdir -p logs

# --- Módulos y entorno CUDA/GCC ---
module purge
module load GCC/12.2.0
module load CUDA/11.8.0

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc no está en PATH tras 'module load CUDA/11.8.0'. Revisa 'module avail CUDA' en este nodo ($(hostname))." >&2
    module avail CUDA 2>&1 >&2
    exit 1
fi

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH=$CUDA_HOME
export CC=$(which gcc)
export CXX=$(which g++)
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="8.0;8.6"   # cubre A30/A100 (8.0) y A40 (8.6)

# --- Entorno conda ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rtdetrv2                 # TODO: nombre de tu entorno conda

# --- Diagnóstico (queda en el log) ---
echo "===== ENV ====="
hostname
date
which python; python --version
which nvcc; nvcc --version
echo "CUDA_HOME=$CUDA_HOME"
nvidia-smi
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.version.cuda, '| available:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
echo "================"

# --- Config de entrenamiento ---
CONFIG=configs/rtdetr/rtdetr_r101vd_6x_coco.yml
CHECKPOINT=checkpoints/rtdetr_r101vd_6x_coco_from_paddle.pth
NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
MASTER_PORT=$((10000 + RANDOM % 20000))

cd "$SLURM_SUBMIT_DIR"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: no se encuentra el checkpoint $CHECKPOINT. Descárgalo antes desde el login node:" >&2
    echo "  wget -P checkpoints/ https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth" >&2
    exit 1
fi

export OMP_NUM_THREADS=1

echo "Job $SLURM_JOB_ID en $(hostname), usando $NPROC_PER_NODE GPUs"

torchrun \
    --master_port="$MASTER_PORT" \
    --nproc_per_node="$NPROC_PER_NODE" \
    tools/train.py -c "$CONFIG" -t "$CHECKPOINT" --amp --seed=0
