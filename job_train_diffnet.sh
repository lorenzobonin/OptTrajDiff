#!/bin/bash
#SBATCH -A uTS25_Bonin
#SBATCH -p boost_usr_prod
#SBATCH -N 2
#SBATCH --time 24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --job-name=diffnet_training
#SBATCH --output=job-%x.out

source ~/.bashrc
conda activate scengen
module load gcc/12.2.0
export LD_LIBRARY_PATH=/leonardo/home/userexternal/lbonin00/micromamba/envs/scengen/lib:$LD_LIBRARY_PATH

srun python train_diffnet_tb.py --root /leonardo_scratch/large/userexternal/lbonin00/argoverse/argoverse_data/ --train_batch_size 4 --val_batch_size 4 --test_batch_size 4 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 150 --time_span 10 --pl2a_radius 50 --a2a_radius 50 --num_t2m_steps 30 --pl2m_radius 150 --a2m_radius 150 --num_nodes 2 --devices 4 --qcnet_ckpt_path ./lightning_logs/QCNet_AV2.ckpt --num_workers 8 --num_denoiser_layers 3 --num_diffusion_steps 100 --T_max 30 --max_epochs 30 --lr 0.005 --beta_1 0.0001 --beta_T 0.05 --diff_type opd --sampling ddim --sampling_stride 10 --num_eval_samples 6 --choose_best_mode FDE --std_reg 0.3 --check_val_every_n_epoch 3 --path_pca_s_mean 'pca/imp_org/s_mean_10.npy' --path_pca_VT_k 'pca/imp_org/VT_k_10.npy' --path_pca_V_k 'pca/imp_org/V_k_10.npy' --path_pca_latent_mean 'pca/imp_org/latent_mean_10.npy' --path_pca_latent_std 'pca/imp_org/latent_std_10.npy'