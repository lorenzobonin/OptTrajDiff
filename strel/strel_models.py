import os
import torch
import pytorch_lightning as pl

import strel.strel_properties as sp


class GenFromLatent(pl.LightningModule):
        def __init__(self, model, scen_id, full_types, pred_types, property_name="reach_uns", tmax=0.2, tglob=3):
            super().__init__()
            self.model = model
            self.scen_id = scen_id
            self.full_types = full_types
            self.pred_types = pred_types
            self.property_name = property_name
            self.tmax = tmax
            self.tglob = tglob
            self.valid_types = full_types


            

        def forward(self, z):

            out = self.model.latent_generator(
                z,
                self.scen_id,
                plot=False,
                enable_grads=True,
                return_pred_only=False,

            )
            full_world, pred_eval_local, mask_eval, eval_mask = out

            # # clean the agents tensor from invalid data
            # full_world, agent_mask = clean_and_filter_agents(full_world)
            # self.valid_types = self.node_types[agent_mask.to(self.node_types.device)]

            # #fix also indexing of predicted agents
            # orig_to_new = torch.full(
            #     (agent_mask.shape[0],),
            #     -1,
            #     device=agent_mask.device,
            #     dtype=torch.long
            # )

            # orig_to_new[agent_mask] = torch.arange(
            #     agent_mask.sum(),
            #     device=agent_mask.device
            # )

            # eval_mask = orig_to_new[eval_mask]

            # self.full_traj = full_world
            # self.pred_traj = pred_eval_local
            # self.mask_eval = mask_eval
            # self.eval_mask = eval_mask


            # Choose STREL property
            if self.property_name == "head_real":
                robustness = sp.evaluate_heading_stability_real(pred_eval_local, self.pred_types, self.tmax, self.tglob)
            elif self.property_name == "reach_uns":
                robustness = sp.evaluate_reach_fast_slow(full_world, self.valid_types, d_zone=2)
            elif self.property_name == "pred_reach":
                robustness = sp.evaluate_reach_fast_slow(pred_eval_local, self.pred_types,d_zone=4)
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
                robustness = sp.evaluate_speeding_surrounded_unmask(full_world, self.valid_types)
            elif self.property_name == "surround_pred":
                robustness = sp.evaluate_speeding_surrounded_unmask(pred_eval_local, self.pred_types)
            elif self.property_name == "surround_slow":
                robustness = sp.evaluate_slowing_surrounded_unmask(pred_eval_local, self.pred_types)

            elif self.property_name =="ped_pred":
                robustness = sp.evaluate_ped_somewhere_unmask(pred_eval_local, self.pred_types,d_zone=2.5)

            elif self.property_name =="ped_eg":
                robustness = sp.evaluate_ped_reach_eg_mask(full_world, mask_eval, eval_mask, self.valid_types, d_zone=1.5)

            elif self.property_name == "ped_unsafe":
                robustness = sp.evaluate_ped_somewhere_unmask(full_world, self.valid_types, d_zone= 2.5)

            elif self.property_name == "fast_slow":
                robustness = sp.evaluate_fast_reach_slow_mask(full_world, mask_eval, eval_mask, self.valid_types, d_zone= 5)

            elif self.property_name == "lane_change":
                robustness = sp.evaluate_unsafe_lanechange_mask(full_world, mask_eval, eval_mask, self.valid_types,theta_turn=self.tmax, v_lat=1.0, d_prox=20)
            else:
                raise ValueError(f"Unknown property type '{self.property_name}'")

            return robustness