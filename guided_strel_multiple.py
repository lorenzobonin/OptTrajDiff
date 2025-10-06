#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pickle
import torch
import pytorch_lightning as pl
from argparse import ArgumentParser
from torch_geometric.data import Batch

from datasets import ArgoverseV2Dataset
from predictors.guided_diffnet import GuidedDiffNet
from transforms import TargetBuilder

import strel.strel_utils as su
import strel.strel_properties as sp


# ============================================================
# --- Generator wrapper for property evaluation
# ============================================================

class GenFromLatent(pl.LightningModule):
    def __init__(self, model, scen_id, node_types, property_name="reach"):
        super().__init__()
        self.model = model
        self.scen_id = scen_id
        self.node_types = node_types
        self.property_name = property_name

    def forward(self, z):
        out = self.model.latent_generator(
            z,
            self.scen_id,
            plot=False,
            enable_grads=True,
            return_pred_only=False

        )
        full_world, pred_eval_local, mask_eval, eval_mask = out

        # Choose STREL property
        if self.property_name == "reach":
            robustness = sp.evaluate_eg_reach_adaptive(
                full_world, mask_eval, eval_mask, self.node_types,
                left_label=[0, 1], right_label=[0, 1, 2]
            )

        elif self.property_name == "reach_adapt":
            robustness = sp.evaluate_eg_reach_adaptive(
                full_world, mask_eval, eval_mask, self.node_types,
                left_label=[0, 1], right_label=[0, 1, 2]
            )

        elif self.property_name == "safe_lane":
            robustness = sp.evaluate_safe_lane_keeping(
                full_world, mask_eval, eval_mask, self.node_types
            )

        elif self.property_name == "cyclist_yield":
            robustness = sp.evaluate_cyclist_yield(
                full_world, mask_eval, eval_mask, self.node_types
            )

        elif self.property_name == "heading_stable":
            robustness = sp.evaluate_heading_stability(
                full_world, mask_eval, eval_mask, self.node_types
            )

        elif self.property_name == "safe_lane":
            robustness = sp.evaluate_safe_lane_keeping_adapt(
                full_world, mask_eval, eval_mask, self.node_types,mode='adaptive'
            )

        elif self.property_name == "ped_some":
            robustness = sp.evaluate_ped_somewhere_safe_adaptive(
                full_world, mask_eval, eval_mask, self.node_types
            )
        elif self.property_name == "ped_yield":
            robustness = sp.evaluate_ped_front_yield_adaptive(
                full_world, mask_eval, eval_mask, self.node_types
            )

        elif self.property_name == "veh_space":
            robustness = sp.evaluate_vehicle_spacing(
                full_world, mask_eval, eval_mask, self.node_types
            )
        elif self.property_name == "ped_clear":
            robustness = sp.evaluate_pedestrian_clearance(
                full_world, mask_eval, eval_mask, self.node_types
            )

        else:
            raise ValueError(f"Unknown property type '{self.property_name}'")

        return robustness


# ============================================================
# --- Main
# ============================================================

if __name__ == '__main__':
    seed_value = 2025
    pl.seed_everything(seed_value, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=1) 
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=str, default="4,")
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--sampling', choices=['ddpm','ddim'],default='ddpm')
    parser.add_argument('--sampling_stride', type = int, default = 20)
    parser.add_argument('--num_eval_samples', type = int, default = 6)
    parser.add_argument('--eval_mode_error_2', type = int, default = 1)
    
    parser.add_argument('--ex_opm', type=int, default=0)
    parser.add_argument('--std_state', choices=['est', 'one'],default = 'est')
    parser.add_argument('--cluster', choices=['normal', 'traj'],default = 'traj')
    parser.add_argument('--cluster_max_thre', type = float,default = 2.5)
    parser.add_argument('--cluster_mean_thre', type = float,default = 2.5)
    
    parser.add_argument('--guid_sampling', choices=['no_guid', 'guid'],default = 'no_guid')
    parser.add_argument('--guid_task', choices=['none', 'goal', 'target_vel', 'target_vego','rand_goal','rand_goal_rand_o'],default = 'none')
    parser.add_argument('--guid_method', choices=['none', 'ECM', 'ECMR'],default = 'none')
    parser.add_argument('--guid_plot',choices=['no_plot', 'plot'],default = 'no_plot')
    parser.add_argument('--std_reg',type = float, default=0.1)
    parser.add_argument('--path_pca_V_k', type = str,default = 'none')

    parser.add_argument('--network_mode', choices=['val', 'test'],default = 'test')
    parser.add_argument('--submission_file_name', type=str, default='submission')
    
    parser.add_argument('--cond_norm', type = int, default = 0)
    
    parser.add_argument('--cost_param_costl', type = float, default = 1.0)
    parser.add_argument('--cost_param_threl', type = float, default = 1.0)
    # === Optimization-specific arguments ===
    parser.add_argument('--property', type=str, default='reach',
                        choices=['reach', 'safe_lane', 'cyclist_yield', 'heading_stable', 'safe_lane', 'ped_some','ped_yield','reach_adapt', 'veh_space','ped_clear'])
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--lambda_reg', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max_steps', type=int, default=300)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    
    args = parser.parse_args()

    split='val'

    # ========================================================
    # Model + dataset setup
    # ========================================================

    model = GuidedDiffNet.from_pretrained(
        checkpoint_path=args.ckpt_path,
        data_path=os.path.join(args.root, args.split)
    )
    model.add_extra_param(args)
    model.sampling = args.sampling
    model.sampling_stride = args.sampling_stride
    model.check_param()
    model.num_eval_samples = args.num_eval_samples

    test_dataset = ArgoverseV2Dataset(
        root=args.root,
        split=args.split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
    )

    # Example list of scenarios
    # top_num_agents_scenarios = [
    #      (25, 7520), (24, 11135), (23, 4611),
    #      (20, 6323),
    #     (19, 1359), (19, 6937)
    # ]

    top_num_agents_scenarios = [(19, 6937)]

    num_dim = 10
    save_dir = f"outputs_{args.property}"
    os.makedirs(save_dir, exist_ok=True)

    # ========================================================
    # Store all results in a dict for reproducibility
    # ========================================================

    summary_results = {
        "seed": seed_value,
        "property": args.property,
        "lambda_reg": args.lambda_reg,
        "lr": args.lr,
        "num_samples": args.num_samples,
        "scenarios": {}
    }

    # ========================================================
    # Scenario loop
    # ========================================================

    for num_agents, scen_idx in top_num_agents_scenarios:
        print(f"\n=== Scenario {scen_idx} (agents={num_agents}) ===")

        # Load graph and bind conditioning
        graph = test_dataset[scen_idx]
        graph = Batch.from_data_list([graph])
        model.cond_data = graph
        x_T = torch.randn([num_agents, 1, num_dim])
    
        full_world, pred_eval_local, mask_eval, eval_mask, node_types = model.latent_generator(x_T, scen_idx, plot=False, enable_grads=True, return_pred_only=False, return_types=True)
        
        rec_pred, pred_types = model.latent_generator(x_T, scen_idx, plot=False, enable_grads=True, return_pred_only=True, return_types=True)


        full_reshaped = su.reshape_trajectories(full_world, node_types)

        print("Full world summary:")
        su.summarize_reshaped(full_reshaped)

        loc_reshaped = su.reshape_trajectories(rec_pred, pred_types)
        print("Reconstructed prediction summary:")
        su.summarize_reshaped(loc_reshaped)

        z0 = torch.randn([num_agents, args.num_samples, num_dim], device=args.device)

        gen_model = GenFromLatent(model, scen_idx, node_types, property_name=args.property).to(args.device)

        # --- Evaluate initial robustness per sample ---
        rob_init = []
        for s in range(args.num_samples):
            z_s = z0[:, s:s+1, :]
            r = gen_model(z_s).detach().item()
            rob_init.append(r)
        rob_init = torch.tensor(rob_init)

        avg_init = rob_init.mean().item()
        neg_init = (rob_init < 0).sum().item()
        perc_neg_init = 100.0 * neg_init / args.num_samples

        print(f"Initial avg robustness: {avg_init:.4f}")
        print(f"Initial negatives: {neg_init}/{args.num_samples} ({perc_neg_init:.1f}%)")

        # --- Vanilla generation ---
        model.latent_generator(
            z0, scen_idx, plot=True,
            enable_grads=False, return_pred_only=True,
            exp_id=f"{save_dir}_vanilla_{scen_idx}",
            img_folder=args.property,
            sub_folder=f'scen_{scen_idx}'
        )

        # --- Optimization ---
        z_opt = su.optimize_samples_individually(
            qmodel=gen_model,
            z0=z0,
            lr=args.lr,
            tol=args.tol,
            max_steps=args.max_steps,
            lambda_reg=args.lambda_reg,
            verbose=True
        )

        # --- Evaluate optimized robustness per sample ---
        rob_opt = []
        for s in range(args.num_samples):
            z_s = z_opt[:, s:s+1, :]
            r = gen_model(z_s).detach().item()
            rob_opt.append(r)
        rob_opt = torch.tensor(rob_opt)

        avg_opt = rob_opt.mean().item()
        neg_opt = (rob_opt < 0).sum().item()
        perc_neg_opt = 100.0 * neg_opt / args.num_samples

        print(f"Optimized avg robustness: {avg_opt:.4f}")
        print(f"Optimized negatives: {neg_opt}/{args.num_samples} ({perc_neg_opt:.1f}%)")

        # --- Optimized generation ---
        model.latent_generator(
            z_opt, scen_idx, plot=True,
            enable_grads=False, return_pred_only=True,
            exp_id=f"{save_dir}_opt_{scen_idx}",
            img_folder=args.property,
            sub_folder=f'scen_{scen_idx}'
        )

        # --- Store results ---
        summary_results["scenarios"][scen_idx] = {
            "num_agents": num_agents,
            "avg_init": avg_init,
            "avg_opt": avg_opt,
            "neg_init": neg_init,
            "neg_opt": neg_opt,
            "perc_neg_init": perc_neg_init,
            "perc_neg_opt": perc_neg_opt,
        }

        print(f"Finished scenario {scen_idx} ({args.property}) — results saved in {save_dir}/")

    # ========================================================
    # Save results to pickle + JSON
    # ========================================================

    stats_path_pkl = os.path.join(save_dir, f"robustness_summary_seed{seed_value}.pkl")
    stats_path_json = os.path.join(save_dir, f"robustness_summary_seed{seed_value}.json")

    with open(stats_path_pkl, "wb") as f:
        pickle.dump(summary_results, f)

    with open(stats_path_json, "w") as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n✅ All scenarios completed. Results saved in:")
    print(f"   → {stats_path_pkl}")
    print(f"   → {stats_path_json}")
