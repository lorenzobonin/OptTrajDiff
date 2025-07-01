#!/bin/bash
#SBATCH -A uTS25_Bonin
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --job-name=qcnet_validation
#SBATCH --output=job-%x.out

source ~/.bashrc
conda activate scengen
module load gcc/12.2.0
export LD_LIBRARY_PATH=/leonardo/home/userexternal/lbonin00/micromamba/envs/scengen/lib:$LD_LIBRARY_PATH

srun python val_qcnet.py --model QCNet --root /leonardo_scratch/large/userexternal/lbonin00/argoverse/argoverse_data/ --ckpt_path /leonardo/home/userexternal/lbonin00/repos/OptTrajDiff/lightning_logs/version_2/checkpoints/epoch=40-step=256168.ckpt
