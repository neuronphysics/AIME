#!/bin/bash
#SBATCH --job-name=DPGMM
#SBATCH --nodes=1
#SBATCH --gpus=h100:1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=180G
#SBATCH --time=01-03:59:59
#SBATCH --account=def-irina
#SBATCH --output=/home/memole/links/projects/def-irina/memole/AIME/logs/dpgmm-transit-run-seed-1_%N-%j.out
#SBATCH --error=/home/memole/links/projects/def-irina/memole/AIME/logs/dpgmm-transit-run-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
set -euo pipefail

#Start clean
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load python/3.11
module load scipy-stack/2024a   # required by opencv/4.10.0
module load opencv/4.10.0
module load mujoco/3.3.0
module load mpi4py/3.1.6
module load arrow
export TF_CPP_MIN_LOG_LEVEL=2

#virtualenv --no-download --clear /home/memole/links/D2E
source /home/memole/links/D2E/bin/activate
### FORCE SINGLE-THREAD BLAS / OMP / TORCH
# ---- VRAM allocator / fragmentation fixes ----
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8
export CUDA_MODULE_LOADING=LAZY
#pip install --no-index --no-cache-dir --no-deps "mamba-ssm==2.2.4+computecanada"
#pip install --no-index --no-cache-dir --no-deps "torch==2.5.1+computecanada" "sympy==1.13.1+computecanada" "torchvision==0.20.1+computecanada" "torchaudio==2.5.1+computecanada"
# If TensorFlow is imported anywhere in the SAME training process:
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=4

# keep Python from seeing ~/.local
export PYTHONNOUSERSITE=1

#python -m pip install --no-deps "sentry-sdk>=2.0.0" "gitpython!=3.1.29,>=1.0.0"
#python -m pip install --no-deps "pydantic>=2,<3"
#python -m pip install --no-deps fairscale
#python -m pip install --no-deps "tensorflow"
#pthonn -m pip install --no-deps setuptools
#python -m pip install --no-deps einx
#python -m pip install --no-deps "timm<1.0.0" 
#python -m pip install --no-deps umap-learn
#python -m pip install --no-deps seaborn
#python -m pip install git+https://github.com/richzhang/PerceptualSimilarity.git
#python -m pip install PyWavelets


#pip install -r requirements.txt --no-deps
#gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/medium /home/memole/links/scratch/transition_data/dmc_vb/humanoid_walk
#gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/expert /home/memole/links/scratch/transition_data/dmc_vb/humanoid_walk
#gsutil -m cp -r gs://dmc_vision_benchmark/dmc_vision_benchmark/locomotion/humanoid_walk/mixed /home/memole/links/scratch/transition_data/dmc_vb/humanoid_walk
#pip install -U huggingface_hub

python -X faulthandler - <<'PY'
import os, torch
print("python ok")
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
x = torch.randn(1024,1024, device="cuda")
y = x @ x
print("matmul ok:", y.mean().item())
PY
cd /home/memole/links/projects/def-irina/memole/AIME
#mkdir -p results/pretrained_weights
#wget https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth -O results/pretrained_weights/lpips_alex.pth

#wget https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth -O results/pretrained_weights/lpips_vgg.pth
#wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth -O results/pretrained_weights/hub/checkpoints/alexnet-owt-7be5be79.pth

# Download VGG16 (alternative, often better)
#wget https://download.pytorch.org/models/vgg16-397923af.pth -O results/pretrained_weights/hub/checkpoints/vgg16-397923af.pth

echo "pretrain VQVAE ....."
#CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.pretrain_vqvae
echo "finished pretraining and start training world model dpgmm vrnn model... " 
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
#python -m pip install -U "huggingface_hub[cli]"
#hf download ml-jku/meta-world --local-dir /home/memole/links/scratch/transition_data/LiRE/meta-world --repo-type dataset
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.dmc_vb_transition_dynamics_trainer --data_dir /home/memole/links/scratch/transition_data/

