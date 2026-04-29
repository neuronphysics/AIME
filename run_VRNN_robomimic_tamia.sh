#!/bin/bash
#SBATCH --job-name=DPGMM-robomimic
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=23:59:59
#SBATCH --account=aip-irina
#SBATCH --output=/home/m/memole/links/scratch/AIME/logs/dpgmm-robomimic-transit-run-seed-1_%N-%j.out
#SBATCH --error=/home/m/memole/links/scratch//AIME/logs/dpgmm-robomimic-transit-run-seed-1_%N-%j.err
#SBATCH --mail-user=sheikhbahaee@gmail.com              # notification for job conditions
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

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

#virtualenv --no-download --clear /home/memole/D2E
source /home/m/memole/D2E/bin/activate
### FORCE SINGLE-THREAD BLAS / OMP / TORCH
# ---- VRAM allocator / fragmentation fixes ----
export PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:256

export CUDA_MODULE_LOADING=LAZY

# If TensorFlow is imported anywhere in the SAME training process:
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_NUM_THREADS=1

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

#python download_datasets.py --tasks lift can square --dataset_types mh --hdf5_types image --download_dir $DATA_ROOT

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
##Dowload dataset
#action space: can (118, 7), lift (48, 7), square (127, 7)
#export DATA_ROOT=/home/m/memole/links/scratch/transition_data/robomimic/datasets
#python download_datasets.py --tasks can --dataset_types mh --hdf5_types raw --download_dir "$DATA_ROOT"
#python /home/m/memole/links/scratch/AIME/VRNN/dataset_states_to_obs.py --dataset "$DATA_ROOT/can/mh/demo_v15.hdf5" --output_name image64_can_mh.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64 --compress --exclude-next-obs
#python /home/m/memole/links/scratch/AIME/VRNN/dataset_states_to_obs.py --dataset "$DATA_ROOT/lift/mh/demo_v15.hdf5" --output_name image64_lift_mh.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64 --compress --exclude-next-obs
#python /home/m/memole/links/scratch/AIME/VRNN/dataset_states_to_obs.py --dataset "$DATA_ROOT/square/mh/demo_v15.hdf5" --output_name image64_square_mh.hdf5 --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 64 --camera_width 64 --compress --exclude-next-obs
#python -c 'import os, h5py; p=os.path.expandvars("$DATA_ROOT/square/mh/demo_v15.hdf5"); f=h5py.File(p,"r"); d=next(iter(f["data"])); print(f["data"][d]["actions"].shape)'
#python -c 'import os,h5py,numpy as np; p=os.path.expandvars("$DATA_ROOT/square/mh/demo_v15.hdf5"); f=h5py.File(p,"r"); L=[f["data"][d]["actions"].shape[0] for d in f["data"]]; print(len(L), "demos, T min/mean/max:", min(L), np.mean(L), max(L))'
CUDA_VISIBLE_DEVICES=0 python3 -m VRNN.robomimic_transition_dynamics_trainer --hdf5 /home/memole/links/scratch/transition_data/robomimic_v1.5/v1.5/can/mg/image64_v15.hdf5 --obs_keys agentview_image --run_name robomimic_can_agentview --logger tensorboard --sequence_length 12
