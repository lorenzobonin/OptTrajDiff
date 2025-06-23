#!/bin/bash
#SBATCH -A uTS25_Bonin
#SBATCH -p boost_usr_prod
#SBATCH -N 1
#SBATCH --time 24:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --job-name=qcnet_training
#SBATCH --output=job-%x.out

source ~/.bashrc
conda activate scengen
module load gcc/12.2.0
export LD_LIBRARY_PATH=/leonardo/home/userexternal/lbonin00/micromamba/envs/scengen/lib:$LD_LIBRARY_PATH

python train_qcnet.py --root /leonardo_scratch/large/userexternal/lbonin00/argoverse/argoverse_data/ --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --devices 4 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150

