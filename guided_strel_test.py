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

import strel.strel_utils as su
import strel.strel_properties as sp
from enum import Enum
import copy
import time


class Agent(Enum):
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    MOTORCYCLIST = 3
    BUS = 4
    STATIC = 5
    BACKGROUND = 6
    CONSTRUCTION = 7
    RIDERLESS_BICYCLE = 8
    UNKNOWN = 9


# functional syntax
Agent = Enum('Agent', [('VEHICLE', 0),('PEDESTRIAN', 1),('CYCLIST', 2),
                       ('MOTORCYCLIST', 3),('BUS', 4),('STATIC', 5),('BACKGROUND', 6),
                       ('CONSTRUCTION', 7),('RIDERLESS_BICYCLE', 8),('UNKNOWN', 9)])




def softmax_max(x, dim, temp=10.0):
    # higher temp → closer to hard max
    weights = torch.softmax(x * temp, dim=dim)
    return (x * weights).sum(dim=dim)


def plot_trajectories(trajectories, filename="trajectories.png"):
    """
    trajectories: tensor [num_agents, samples, timesteps, 2]
    """
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    
    num_agents, num_samples, _, _ = trajectories.shape
    
    plt.figure(figsize=(6, 6))
    
    for agent in range(num_agents):
        for sample in range(num_samples):
            traj = trajectories[agent, sample]  # shape [timesteps, 2]
            x, y = traj[:, 0], traj[:, 1]
            plt.plot(x, y, linewidth=1, alpha=0.7)
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("2D Trajectories")
    plt.axis("equal")
    plt.grid(True)
    
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == '__main__':
    pl.seed_everything(45 , workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4) 
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

    parser.add_argument('--property', type=str, default='reach_uns',
                        choices=['reach_uns', 'head_real', 'ped_unsafe', 'reach_simp', 'pred_reach'])
    
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

    test_dataset = ArgoverseV2Dataset(
        root=args.root,
        split=split,
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
    )

    scen_idx = 1359

    num_agents = 19

    num_dim = 10

    
    

    # Turn it back into a HeteroDataBatch object (the only one working)
    graph = test_dataset[scen_idx]
    graph = Batch.from_data_list([graph])
    model.cond_data = graph


    num_dim = 10 # Latent dim
    x_T = torch.randn([num_agents, 1, num_dim])
    
    full_world, pred_eval_local, mask_eval, eval_mask, full_types = model.latent_generator(x_T, scen_idx, plot=False, enable_grads=True, return_pred_only=False, return_types=True)
    
    tmax, tglob = su.estimate_heading_thresholds(full_world)
    
    pred_local, pred_types = model.latent_generator(x_T, scen_idx, plot=False, enable_grads=True, return_pred_only=True, return_types=True)
    
    N, T, _ = full_world.shape
    print("Full types:", full_types)
    print("Pred types:", pred_types)
    print("full type shape:", full_types.size())
    print("pred type shape:", pred_types.size())

    # Node categories (adapt if you have heterogeneous agents)
    if args.property == 'head_real' or args.property == 'pred_reach':
        node_types = pred_types
    else:
        node_types = full_types

    #node_types = torch.zeros(N, dtype=torch.long)
    full_reshaped = su.reshape_trajectories(full_world, full_types)

    print('salerno')
    su.summarize_reshaped(full_reshaped)
    try:
        avg_ped_veh = su.average_intertype_distance(full_world, full_types, type_a=Agent.PEDESTRIAN, type_b=Agent.VEHICLE)
        print(f"Average pedestrian–vehicle distance: {avg_ped_veh:.2f} m")
    except:
        print('failed printing interype distance')
    class GenFromLatent(pl.LightningModule):
        def __init__(self, model, scen_id, types, property_name="reach_uns", tmax=0.2, tglob=3):
            super().__init__()
            self.model = model
            self.scen_id = scen_id
            self.node_types = types
            self.property_name = property_name
            self.tmax = tmax
            self.tglob = tglob

        def forward(self, z):



            out = self.model.latent_generator(
                z,
                self.scen_id,
                plot=False,
                enable_grads=True,
                return_pred_only=False,

            )
            full_world, pred_eval_local, mask_eval, eval_mask = out

            # Choose STREL property
            if self.property_name == "head_real":
                robustness = sp.evaluate_heading_stability_real(pred_eval_local, self.node_types, self.tmax, self.tglob)
            elif self.property_name == "reach_uns":
                robustness = sp.evaluate_eg_reach_mask(
                    full_world, mask_eval, eval_mask, self.node_types,
                    left_label=[], right_label=[], threshold_1=1.3, threshold_2=1.0, d_max=10
                )
            elif self.property_name == "pred_reach":
                robustness = sp.evaluate_eg_reach(
                    pred_eval_local, mask_eval, eval_mask, self.node_types,
                    left_label=[0,1,2,3,4], right_label=[0,1,2,3,4], threshold_1=1.3, threshold_2=1.0, d_max=20
                )
            elif self.property_name == "reach_simp":
                robustness = sp.evaluate_simple_reach(
                    full_world, mask_eval, eval_mask, self.node_types,
                    left_label=[0,1,2,3,4], right_label=[0,1,2,3,4], threshold_1=1.3, threshold_2=1.0, d_max=20
                )
            elif self.property_name == "ped_unsafe":
                robustness = sp.evaluate_ped_somewhere_unsafe(full_world, mask_eval, eval_mask, self.node_types, d_zone= 150)
            else:
                raise ValueError(f"Unknown property type '{self.property_name}'")

            return robustness
    
    
    gen_model = GenFromLatent(model, scen_idx, node_types, property_name=args.property, tmax = tmax, tglob = tglob)
    gen_model.eval()
    #pred = model.latent_generator(x_T, i, plot=True)
    z_param = torch.nn.Parameter(x_T.clone())
    robust = gen_model(z_param)
    print("robust.requires_grad:", robust.requires_grad)  # should be True

    g = torch.autograd.grad(robust, z_param, retain_graph=True, allow_unused=True)[0]
    print("‖grad‖:", 0.0 if g is None else g.detach().abs().max().item())

    z_opt = su.grad_ascent_reg(qmodel = gen_model, z0 = x_T, lr=0.01, tol=1e-12, lambda_reg=-0.001, max_steps=700)

    z_reg = su.grad_ascent_reg(qmodel = gen_model, z0 = x_T, lr=0.01, tol=1e-12, lambda_reg=0.001, max_steps=700)

    print("Initial latent point:", x_T)
    print("Optimal latent point:", z_opt)
    
    print("Initial robustness:", robust.item())
    robust_opt = gen_model(z_opt)
    print("Optimal robustness:", robust_opt.item())

    robust_reg = gen_model(z_reg)
    print("Optimal robustness:", robust_reg.item())
    
            
    r2_init, r2_opt, dlogp = su.latent_loglik_diff(z_param, z_opt)
    print("Initial vs optimal loglik diff:", r2_init, r2_opt, dlogp)

    r2_init, r2_opt, dlogp = su.latent_loglik_diff(z_param, z_reg)
    print("Initial vs reg optimal loglik diff:", r2_init, r2_opt, dlogp)

    #model.latent_generator(x_T, i, plot=True, enable_grads=False, return_pred_only=False)

    
    model.latent_generator(x_T, scen_idx, plot=True, enable_grads=False, return_pred_only=True, exp_id= f"{args.property}_init")

    model.latent_generator(z_opt, scen_idx, plot=True, enable_grads=False, return_pred_only=True, exp_id= f"{args.property}_opt")

    model.latent_generator(z_reg, scen_idx, plot=True, enable_grads=False, return_pred_only=True, exp_id= f"{args.property}_reg")
            
    #print(model.cond_data['agent']['predict_mask'].sum()) # should be equal to num_agents
    #print(model.cond_data['agent']['valid_mask'].sum())