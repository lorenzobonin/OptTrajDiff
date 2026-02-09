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
from enum import Enum
import time
import utils.safety_metrics as saf

# ============================================================
# --- Generator wrapper for property evaluation
# ============================================================

# al momento qui, da spostare in utils, c'è una copia come metodo della classe in guided diffnet, perché???
def decode_types_from_scenario(num_types):
        # Mapping from Argoverse numeric types to names
        ID_TO_TYPE = {
            0: "VEHICLE",
            1: "PEDESTRIAN",
            2: "CYCLIST",
            3: "MOTORCYCLIST",
            4: "BUS",
            5: "STATIC",
            6: "BACKGROUND",
            7: "CONSTRUCTION",
            8: "RIDERLESS_BICYCLE",
            9: "UNKNOWN",
        }

        types = [ID_TO_TYPE.get(int(t), "UNKNOWN") for t in num_types]

        # eval_mask = data['agent']['category'] >= 2
        # types = [t for i, t in enumerate(types) if eval_mask[i]]

        return types

def clean_and_filter_agents(full_world):
    """
    full_world: [N, T, 2]

    returns:
      world_valid: [N_valid, T, 2]   (only valid agents, cleaned)
      agent_mask:  [N]               (True = kept agent)
    """
    N, T, _ = full_world.shape
    device = full_world.device

    # 1) timestep validity mask
    valid = (full_world.abs().sum(-1) != 0)  # [N, T]

    # 2) agent-level validity
    agent_mask = valid.any(dim=1)             # [N]

    # 3) remove fully invalid agents
    world = full_world[agent_mask]             # [N_valid, T, 2]
    valid = valid[agent_mask]                  # [N_valid, T]

    # early exit
    if world.numel() == 0:
        return world, agent_mask

    Nv = world.shape[0]
    time = torch.arange(T, device=device)

    # 4) forward fill indices
    last_valid = torch.where(
        valid,
        time.unsqueeze(0),
        torch.full((Nv, T), -1, device=device)
    )
    last_valid = torch.cummax(last_valid, dim=1).values

    # 5) backward fill indices
    next_valid = torch.where(
        valid,
        time.unsqueeze(0),
        torch.full((Nv, T), T, device=device)
    )
    next_valid = torch.cummin(next_valid.flip(1), dim=1).values.flip(1)

    # 6) choose valid index per timestep
    idx = torch.where(last_valid >= 0, last_valid, next_valid)
    idx = idx.clamp(0, T - 1)

    # 7) gather filled trajectories
    idx = idx.unsqueeze(-1).expand(-1, -1, 2)
    world_valid = torch.gather(world, dim=1, index=idx)

    return world_valid, agent_mask

class GenFromLatent(pl.LightningModule):
        def __init__(self, model, scen_id, types, property_name="reach_uns", tmax=0.2, tglob=3):
            super().__init__()
            self.model = model
            self.scen_id = scen_id
            self.node_types = types
            self.property_name = property_name
            self.tmax = tmax
            self.tglob = tglob
            self.valid_types = node_types

        def forward(self, z):

            out = self.model.latent_generator(
                z,
                self.scen_id,
                plot=False,
                enable_grads=True,
                return_pred_only=False,

            )
            full_world, pred_eval_local, mask_eval, eval_mask = out

            # clean the agents tensor from invalid data
            full_world, agent_mask = clean_and_filter_agents(full_world)
            self.valid_types = self.node_types[agent_mask.to(self.node_types.device)]

            #fix also indexing of predicted agents
            orig_to_new = torch.full(
                (agent_mask.shape[0],),
                -1,
                device=agent_mask.device,
                dtype=torch.long
            )

            orig_to_new[agent_mask] = torch.arange(
                agent_mask.sum(),
                device=agent_mask.device
            )

            eval_mask = orig_to_new[eval_mask]

            # Choose STREL property
            if self.property_name == "head_real":
                robustness = sp.evaluate_heading_stability_real(pred_eval_local, self.valid_types, self.tmax, self.tglob)
            elif self.property_name == "reach_uns":
                robustness = sp.evaluate_eg_reach_mask(
                    full_world, mask_eval, eval_mask, self.valid_types,
                    left_label=None, right_label=None, threshold_1=1.3, threshold_2=1.0, d_max=10
                )
            elif self.property_name == "pred_reach":
                robustness = sp.evaluate_eventually_reach(
                    pred_eval_local, mask_eval, eval_mask, self.valid_types,
                    left_label=[0,1,2,3,4], right_label=[0,1,2,3,4], threshold_1=1.3, threshold_2=1.0, d_max=10
                )
            elif self.property_name == "reach_simp":
                robustness = sp.evaluate_simple_reach(
                    full_world, mask_eval, eval_mask, self.valid_types,
                    left_label=[0,1,2,3,4], right_label=[0,1,2,3,4], threshold_1=1.3, threshold_2=1.0, d_max=20
                )
            elif self.property_name == "surround_accel":
                robustness = sp.evaluate_accel_surrounded_mask(full_world, mask_eval, eval_mask, self.valid_types)

            elif self.property_name == "mean_reach":
                robustness = sp.meaningful_reach(full_world, mask_eval, eval_mask, self.valid_types)

            elif self.property_name == "surround_fast":
                robustness = sp.evaluate_speeding_surrounded_unsafe_mask(full_world, mask_eval, eval_mask, valid_types)

            elif self.property_name =="ped_pred":
                robustness = sp.evaluate_ped_somewhere_unmask(pred_eval_local, self.valid_types,d_zone=3)

            elif self.property_name =="ped_eg":
                robustness = sp.evaluate_ped_reach_eg_mask(full_world, mask_eval, eval_mask, self.valid_types, d_zone=1.5)

            elif self.property_name == "ped_unsafe":
                robustness = sp.evaluate_ped_reach_mask(full_world, mask_eval, eval_mask, self.valid_types, d_zone= 20.0)

            elif self.property_name == "fast_slow":
                robustness = sp.evaluate_fast_reach_slow_mask(full_world, mask_eval, eval_mask, self.valid_types, d_zone= 5)

            elif self.property_name == "lane_change":
                robustness = sp.evaluate_unsafe_lanechange_mask(full_world, mask_eval, eval_mask, self.valid_types,theta_turn=self.tmax, v_lat=1.0, d_prox=20)

            elif self.property_name == "min_vel":
                robustness = sp.test_grad_minimize_movement_with_reshape(pred_eval_local, self.node_types) # da fixare
            else:
                raise ValueError(f"Unknown property type '{self.property_name}'")

            return robustness


# ============================================================
# --- Main
# ============================================================

if __name__ == '__main__':
    seed_value = 80085
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
    parser.add_argument('--property', type=str, default='reach_uns',
                        choices=['reach_uns', 'head_real', 'ped_unsafe', 'reach_simp', 'pred_reach', 'ped_pred', 
                                 'surround_fast', 'ped_eg',
                                 'surround_accel', 'lane_change', 'fast_slow', 'mean_reach', 'min_vel'])
    
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--lambda_reg', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--max_steps', type=int, default=500)
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

    #Example list of scenarios
    # top_num_agents_scenarios = [
    #      (25, 7520), (24, 11135), (23, 4611),
    #      (20, 6323),
    #     (19, 1359), (19, 6937)
    # ]
    #for ped unsafe
    top_num_agents_scenarios = [(19, 1359)]

    #for surround
    #top_num_agents_scenarios = [(20, 6323), (19, 6937)]

    #for heading
    #top_num_agents_scenarios = [(24, 11135)]

    #for reach
    #top_num_agents_scenarios = [(25, 7520)]

    #top_num_agents_scenarios = [(9, 10863)]
    
    num_dim = 10
    out_dir = f"outputs_{args.property}"
    save_dir = os.path.join('results_opt', out_dir)
    os.makedirs(save_dir, exist_ok=True)
    print('property:', args.property)
    print('distance front and thresholds applied where needed!')
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
        time_start = time.time()
        
        print(f"\n=== Scenario {scen_idx} (agents={num_agents}) ===")

        # Load graph and bind conditioning
        graph = test_dataset[scen_idx]
        graph = Batch.from_data_list([graph])

        model.cond_data = graph
        x_T = torch.randn([num_agents, 1, num_dim])
    
        full_world, pred_eval_local, mask_eval, eval_mask, full_types = model.latent_generator(x_T, scen_idx, plot=False, enable_grads=True, return_pred_only=False, return_types=True)
        
        rec_pred, pred_types = model.latent_generator(x_T, scen_idx, plot=False, enable_grads=True, return_pred_only=True, return_types=True)


        full_reshaped = su.reshape_trajectories(full_world, full_types)



        print("Full world summary:")
        su.summarize_reshaped(full_reshaped)

        loc_reshaped = su.reshape_trajectories(rec_pred, pred_types)
        print("Reconstructed prediction summary:")
        su.summarize_reshaped(loc_reshaped)

        tmax, tglob = su.estimate_heading_thresholds(full_world)
        if args.property == 'head_real' or args.property == 'pred_reach' or args.property== 'ped_pred' or args.property=='min_vel':
            node_types = pred_types
        else:
            node_types = full_types

        z0 = torch.randn([num_agents, args.num_samples, num_dim], device=args.device)

        gen_model = GenFromLatent(model, scen_idx, node_types, property_name=args.property, tmax=tmax, tglob=tglob).to(args.device)

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



        img_dir = os.path.join(save_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)
        # --- Vanilla generation ---
        vanilla_traj = model.latent_generator(
            z0, scen_idx, plot=True,
            enable_grads=False, return_pred_only=True,
            exp_id=f"{seed_value}_vanilla_{scen_idx}",
            img_folder=img_dir,
            sub_folder=f'scen_{scen_idx}'
        )

        #su.debug_property(gen_model, z0)

        # --- Optimization ---
        # z_opt = su.reg_samples_individually(
        #     qmodel=gen_model,
        #     z0=z0,
        #     lr=args.lr,
        #     tol=args.tol,
        #     max_steps=args.max_steps,
        #     lambda_reg=args.lambda_reg,
        #     verbose=True
        # )

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
        perc_neg_opt = 100.0 - 100.0 * neg_opt / args.num_samples

        print(f"Optimized avg robustness: {avg_opt:.4f}")
        print(f"Optimized negatives: {neg_opt}/{args.num_samples} ({perc_neg_opt:.1f}%)")


        # --- Optimized generation ---
        opt_traj = model.latent_generator(
            z_opt, scen_idx, plot=True,
            enable_grads=False, return_pred_only=True,
            exp_id=f"{seed_value}_opt_{scen_idx}",
            img_folder= img_dir,
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
            "init_tensor": z0,
            "opt_tensor": z_opt
        }
        time_end = time.time()
        print(f"Complete optimization of the scenario: {time_end - time_start:.4f}")
        print(f"Finished scenario {scen_idx} ({args.property}) — results saved in {save_dir}/")

        type_list = decode_types_from_scenario(gen_model.valid_types)

        try: 
            traj_path_pkl = os.path.join(save_dir, f"{scen_idx}_vanilla_traj_seed{seed_value}.pkl")
            opt_path_pkl = os.path.join(save_dir, f"{scen_idx}_opt_traj_seed{seed_value}.pkl")
            zopt_path_pkl = os.path.join(save_dir, f"{scen_idx}_z_opt_seed{seed_value}.pkl")

            with open(traj_path_pkl, "wb") as f:
                pickle.dump(vanilla_traj, f)
            
            with open(opt_path_pkl, "wb") as f:
                pickle.dump(opt_traj, f)

            with open(zopt_path_pkl, "wb") as f:
                pickle.dump(z_opt, f)
        except:
            print('cannot dump pickles!')
        try:
            #all_types = [0,1,2,3,4,5,6,7,8]
            print(vanilla_traj.shape)
            print(len(type_list))
            min_d_van = saf.min_vehicle_related_distance_per_sample(vanilla_traj, type_list)
            print('minimum distance for vanilla_traj', min_d_van)
            min_d_opt = saf.min_vehicle_related_distance_per_sample(opt_traj, type_list)
            print('minimum distance for vanilla_traj', min_d_opt)
        except Exception as e:
            print(e)
            #print('cannot compute distances!')
        
        try:
            #all_types = [0,1,2,3,4,5,6,7,8]
            coll_van = saf.collision_flag_per_sample(vanilla_traj, type_list)
            print('collisions for vanilla_traj', coll_van)
            coll_opt = saf.collision_flag_per_sample(opt_traj, type_list)
            print('collisions for vanilla_traj', coll_opt)
        except Exception as e:
            print(e)
            #print('cannot compute collisions!')
        try:
            safety_results ={
                "orig_distance" : min_d_van,
                "opt_distance" : min_d_opt,
                "orig_coll" : coll_van,
                "opt_coll" : coll_opt
            }
            safe_path= os.path.join(save_dir, f"{scen_idx}_safety_summary_seed{seed_value}.pkl")
            with open(safe_path, "wb") as f:
                pickle.dump(safety_results, f)
        except:
            print('cannot save safety results!')


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


    