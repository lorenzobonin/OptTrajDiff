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


if __name__ == '__main__':
    pl.seed_everything(2025, workers=True)

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

    top_num_agents_scenarios = [(28, 18070), (25, 7520), (24, 11135), (23, 4611), (22, 23297), (20, 6323), (20, 7129), (19, 1359), (19, 6569), (19, 6937)]
    top_diversity_scenarios = [(10, 8709), (10, 9817), (9, 4433), (9, 7391), (9, 7928), (9, 9290), (9, 9738), (9, 10302), (9, 10863), (9, 12518)]

    for num_agents, idx in top_num_agents_scenarios:
        
        first_graph = test_dataset[idx]
        first_graph = Batch.from_data_list([first_graph])

        model.cond_data = first_graph
        num_dim = 10
        x_T = torch.randn([num_agents, 1, num_dim])
        pred = model.latent_generator(x_T, idx, plot=True)

    # for _, idx in top_diversity_scenarios:
        
    #     first_graph = test_dataset[idx]

    #     # print(first_graph)
    #     # Turn it back into a HeteroDataBatch object (the only one working)
    #     first_graph = Batch.from_data_list([first_graph])

    #     model.cond_data = first_graph

    #     # Getting an input with the right dimensionality
    #     #num_agents = 5 # Number of predictable trajectories in the batch
    #     num_dim = 10 # Latent dim
    #     x_T = torch.randn([5, 1, num_dim])
    #     pred = model.latent_generator(x_T, idx, plot=True)


        # print(pred) # [num_agents x samples x timesteps x output_dim (2D position)]
        # print(pred) # [num_agents x samples x timesteps x output_dim (2D position)]

    
    # dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    #                         pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)

    # # iterator = iter(dataloader)
    # # data_batch = next(iterator)

    # top_k = 10
    # top_agents = []      # [(num_agents, idx)]
    # top_diverse = []     # [(num_diverse_agents, idx)]


    # print(f"DS size: {len(test_dataset)}")
    # for idx, data_batch in enumerate(dataloader):
    #     if idx%1000 == 0: 
    #         print(idx)
    #     # i = 5 # select scenario in the batch

    #     first_graph = data_batch.to_data_list()[0]

    #     # Turn it back into a HeteroDataBatch object (the only one working)
    #     first_graph = Batch.from_data_list([first_graph])

    #     # print(first_graph)

    #     model.cond_data = first_graph

    #     # Getting an input with the right dimensionality
    #     #num_agents = 5 # Number of predictable trajectories in the batch
    #     num_dim = 10 # Latent dim
    #     x_T = torch.randn([5, 1, num_dim])
    #     num_agents = model.latent_generator(x_T, idx, plot=False)

    #     diverse_agents = first_graph['agent'].type.unique().numel()

    #     # Track top 10 by predicted agents
    #     top_agents.append((num_agents, idx))
    #     top_agents = sorted(top_agents, key=lambda x: x[0], reverse=True)[:top_k]

    #     # Track top 10 by diversity
    #     top_diverse.append((diverse_agents, idx))
    #     top_diverse = sorted(top_diverse, key=lambda x: x[0], reverse=True)[:top_k]


    #     # print(pred) # [num_agents x samples x timesteps x output_dim (2D position)]

    # print(top_agents)
    # print(top_diverse)
            
            
        