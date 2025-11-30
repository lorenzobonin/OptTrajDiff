import numpy as np
from utils.safety_metrics import min_vehicle_related_distance_per_sample
import pickle
import json

from argparse import ArgumentParser

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

from datasets import ArgoverseV2Dataset
from predictors.guided_diffnet import GuidedDiffNet
from transforms import TargetBuilder
import os
import torch
import matplotlib.pyplot as plt

def decode_types_from_scenario(data):
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

        type_ids = data['agent']['type'].cpu().numpy()
        types = [ID_TO_TYPE.get(int(t), "UNKNOWN") for t in type_ids]

        # eval_mask = data['agent']['category'] >= 2
        # types = [t for i, t in enumerate(types) if eval_mask[i]]

        return types


if __name__ == '__main__':
    pl.seed_everything(1998, workers=True)

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
    
    args = parser.parse_args()

    split='val'

    model = {
        'GuidedDiffNet': GuidedDiffNet,
    }['GuidedDiffNet'].from_pretrained(checkpoint_path=args.ckpt_path, data_path = os.path.join(args.root, split))
    
    model.add_extra_param(args)
    
    
    model.sampling = args.sampling
    model.sampling_stride = args.sampling_stride
    model.check_param()
    model.num_eval_samples = args.num_eval_samples
    model.eval_mode_error_2 = args.eval_mode_error_2

    test_dataset = {
        'argoverse_v2': ArgoverseV2Dataset,
    }[model.dataset](root=args.root, split=split,
                     transform=TargetBuilder(model.num_historical_steps, model.num_future_steps))
    res_path = "/leonardo_scratch/fast/IscrC_ADGA/STREL_results"
    formula = "outputs_ped_unsafe"
    idx = 1359
    output_folder = "./safety_results/"
    output_filename = f"{formula}_{idx}_safety_results.json"
    output_path = output_folder + output_filename

    res = {
        "opt_min_distance_vehicle" : None,
        "opt_collisions_vehicle": None,
        "opt_min_distance_all" : None,
        "opt_collisions_all": None,
        "vanilla_min_distance_vehicle" : None,
        "vanilla_collisions_vehicle": None,
        "vanilla_min_distance_all" : None,
        "vanilla_collisions_all": None,
    }

    # opt_filename = f"{res_path}/{formula}/{idx}_opt_traj_seed9.pkl"
    # vanilla_filename = f"{res_path}/{formula}/{idx}_vanilla_traj_seed9.pkl"

    opt_z_filename = f"{res_path}/{formula}/{idx}_z_opt_seed1998.pkl"

    # with open(opt_filename, "rb") as f:
    #     opt_traj = pickle.load(f)

    # with open(vanilla_filename, "rb") as f:
    #     vanilla_traj = pickle.load(f)

    with open(opt_z_filename, "rb") as f:
        opt_z = pickle.load(f)

    graph = test_dataset[idx]
    graph = Batch.from_data_list([graph])

    type_list = decode_types_from_scenario(graph)

    model.cond_data = graph
    num_dim = 10

    vanilla_z = torch.randn_like(opt_z)

    vanilla_traj = torch.zeros((len(type_list), 20, 60, 2))
    opt_traj = torch.zeros((len(type_list), 20, 60, 2))

    for sample in range(vanilla_z.shape[1]):
        vanilla_traj[:, sample, :, :] = model.latent_generator(vanilla_z[:, sample, :].unsqueeze(1), idx, return_pred_only = False)[0]
        opt_traj[:, sample, :, :] = model.latent_generator(opt_z[:, sample, :].unsqueeze(1), idx, return_pred_only = False)[0]

    opt_traj = opt_traj.cpu().numpy()
    vanilla_traj = vanilla_traj.cpu().numpy()

    opt_veh_distances, where_minor = min_vehicle_related_distance_per_sample(opt_traj, type_list, only_vehicles = True)
    vanilla_veh_distances, _ = min_vehicle_related_distance_per_sample(vanilla_traj, type_list, only_vehicles = True)
    opt_all_distances, _ = min_vehicle_related_distance_per_sample(opt_traj, type_list, only_vehicles = False)
    vanilla_all_distances, _ = min_vehicle_related_distance_per_sample(vanilla_traj, type_list, only_vehicles = False)

    print(where_minor)

    opt_veh_collided = opt_veh_distances < 1
    vanilla_veh_collided = vanilla_veh_distances < 1
    opt_all_collided = (opt_all_distances < 0.4) | opt_veh_collided
    vanilla_all_collided = (vanilla_all_distances < 0.4) | vanilla_veh_collided
 
    res["opt_min_distance_vehicle"] = opt_veh_distances.tolist()
    res["opt_collisions_vehicle"] = int(np.sum(opt_veh_collided))
    res["opt_min_distance_all"] = opt_all_distances.tolist()
    res["opt_collisions_all"] = int(np.sum(opt_all_collided))
    res["vanilla_min_distance_vehicle"] = vanilla_veh_distances.tolist()
    res["vanilla_collisions_vehicle"] = int(np.sum(vanilla_veh_collided))
    res["vanilla_min_distance_all"] = vanilla_all_distances.tolist()
    res["vanilla_collisions_all"] = int(np.sum(vanilla_all_collided))

    with open(output_path, "w") as f:
        json.dump(res, f)

    import matplotlib.pyplot as plt

    # Create a box plot
    plt.boxplot([vanilla_veh_distances, opt_veh_distances], labels=['Vanilla', 'Opt'])
    plt.title("Distances vehicles")
    plt.ylabel("Values")

    # Save the figure
    plt.savefig(f"./safety_results/{idx}_boxplot_veh.png", dpi=300, bbox_inches='tight')  # saves to current directory

    # Create a box plot
    plt.boxplot([vanilla_all_distances, opt_all_distances], labels=['Vanilla', 'Opt'])
    plt.title("Distances all")
    plt.ylabel("Values")

    # Save the figure
    plt.savefig(f"./safety_results/{idx}_boxplot_all.png", dpi=300, bbox_inches='tight')  # saves to current directory
    