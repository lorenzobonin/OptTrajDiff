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

import strel_utils as su

import copy








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

class GenFromLatent(pl.LightningModule):
    def __init__(self, model, scen_id):
        super().__init__()
        self.model = model
        self.scen_id = scen_id

        #insert STREL here if needed

    

    def forward(self, x_T):
        # Ask for fused world + differentiable pieces
        out = self.model.latent_generator(
            x_T,
            self.scen_id,
            plot=False,
            enable_grads=True,
            return_pred_only=False
        )

    
        full_world, pred_eval_local, mask_eval, eval_mask = out

        #full_world = out
        
        # Robustness over the whole fused world trajectory
        #robustness = su.toy_safety_function(full_world, min_dist=2.0)
        #robustness = su.evaluate_reach_property(full_world, left_label=1, right_label=1, threshold_1=3.0, threshold_2=3.0)
        #robustness = su.evaluate_reach_property_mask(full_world, mask_eval, eval_mask, left_label=1, right_label=1, threshold_1=3.0, threshold_2=3.0)
        robustness = su.evaluate_eg_reach_mask(full_world, mask_eval, eval_mask, left_label=[1], right_label=[1], threshold_1=4.0, threshold_2=2.0, d_max=50)
        return robustness



        #return self.model.latent_generator(x_T, self.scen_id, plot=False, enable_grads=True, return_pred_only=False)

if __name__ == '__main__':
    pl.seed_everything(20 , workers=True)

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
    
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    iterator = iter(dataloader)
    data_batch = next(iterator)

    i = 5 # select scenario in the batch
    
    first_graph = data_batch.to_data_list()[i]

    # Turn it back into a HeteroDataBatch object (the only one working)
    first_graph = Batch.from_data_list([first_graph])

    model.cond_data = first_graph

    
    

    # Getting an input with the right dimensionality
    num_agents = 5 # Number of predictable trajectories in the batch
    num_dim = 10 # Latent dim
    x_T = torch.randn([num_agents, 1, num_dim])
    
    full_world, pred_eval_local, mask_eval, eval_mask = model.latent_generator(x_T, i, plot=False, enable_grads=True, return_pred_only=False)
    N, T, _ = pred_eval_local.shape

    # Node categories (adapt if you have heterogeneous agents)
    node_types = torch.ones(N, device = pred_eval_local.device)
    full_wolrd = su.clean_near_zero(pred_eval_local, eps=1e-4)
    imputed_full = su.impute_positions_closest(full_wolrd)
    full_reshaped = su.reshape_trajectories(imputed_full, node_types)
    dirty_reshaped = su.reshape_trajectories(pred_eval_local, node_types)

    su.summarize_reshaped(full_reshaped)
    su.summarize_reshaped(dirty_reshaped)
    print("mask eval size is ", mask_eval.size())
    print("eval mask size is ", eval_mask.size())
    print("pred eval local size is ", pred_eval_local.size())
    print("full world", full_world[0])
    print("imputed full", imputed_full[0])
    #print("traj requires grad:", full_world.requires_grad)
    #print("traj grad_fn:", full_world.grad_fn)
    print('only pred size is ', model.latent_generator(x_T, i, plot=False, enable_grads=True, return_pred_only=True).size())
    print('full pred size is ', full_world.size())
    print('full traj size is', model.cond_data['agent']['predict_mask'].size())
    
    
    gen_model = GenFromLatent(model, i)
    gen_model.eval()
    #pred = model.latent_generator(x_T, i, plot=True)
    z_param = torch.nn.Parameter(x_T.clone())
    robust = gen_model(z_param)
    print("robust.requires_grad:", robust.requires_grad)  # should be True

    g = torch.autograd.grad(robust, z_param, retain_graph=True, allow_unused=True)[0]
    print("‖grad‖:", 0.0 if g is None else g.detach().abs().max().item())

    z_opt = su.grad_ascent_opt(qmodel = gen_model, z0 = x_T, lr=0.005, tol=1e-12)

    print("Initial latent point:", x_T)
    print("Optimal latent point:", z_opt)
    
    print("Initial robustness:", robust.item())
    robust_opt = gen_model(z_opt)
    print("Optimal robustness:", robust_opt.item())
    
            
    r2_init, r2_opt, dlogp = su.latent_loglik_diff(z_param, z_opt)
    print("Initial vs optimal loglik diff:", r2_init, r2_opt, dlogp)

    #model.latent_generator(x_T, i, plot=True, enable_grads=False, return_pred_only=False)

    
    model.latent_generator(x_T, i, plot=True, enable_grads=False, return_pred_only=True, exp_id= "_init")

    model.latent_generator(z_opt, i, plot=True, enable_grads=False, return_pred_only=True, exp_id= "_opt")
            
    #print(model.cond_data['agent']['predict_mask'].sum()) # should be equal to num_agents
    #print(model.cond_data['agent']['valid_mask'].sum())