from predictors import DiffNet
from modules import JointDiffusion
import torch
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from typing import Dict, Mapping
import copy
import numpy as np
import time

from visualization import *
from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

import os
from pathlib import Path

# --- safe deepcopy for tensors ---
def deepcopy_preserve_tensors(obj):
    """Deepcopy that replaces Tensors with .clone(), preserving autograd."""
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    try:
        return copy.deepcopy(obj)
    except Exception:
        return obj
# --------------------------------


class GuidedJointDiffusion(JointDiffusion):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_latent(self,
               x_T,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               mean=None,
               std=None,
               mm=None,
               mmscore=None,
               if_output_diffusion_process=False,
               reverse_steps=None,
               eval_mask=None,
               stride=20,
               cond_gen=None,
               enable_grads: bool = False   # NEW flag
               ) -> Dict[str, torch.Tensor]:
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps

        num_samples = 1
        device = mean.device
        num_agents = mean.size(0)

        mean = mean.unsqueeze(1)
        std = std.unsqueeze(1)

        x_T = x_T * std
        s_T = torch.sqrt(self.var_sched.alpha_bars[reverse_steps].to(device)) * mean
        
        x_T = x_T + s_T
        x_t_list = [x_T]
        torch.cuda.empty_cache()

        for t in range(reverse_steps, 0, -stride):
            z = torch.randn_like(x_T) * std if t > 1 else torch.zeros_like(x_T)
            beta = self.var_sched.betas[t]
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]

            x_t = x_t_list[-1]
            if cond_gen is not None:
                [idx, target_mode] = cond_gen
                x_t[idx, :, :] = target_mode.unsqueeze(0).repeat(num_samples, 1)

            # choose context
            ctx = torch.enable_grad() if enable_grads else torch.no_grad()
            with ctx:
                beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                g_theta = self.net(deepcopy_preserve_tensors(x_t), beta_emb, data, scene_enc,
                                   num_samples=num_samples, mm=mm, mmscore=mmscore, eval_mask=eval_mask)
        
            # ddim update
            x0_t = (x_t - g_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * g_theta

            if True in torch.isnan(x_next):
                print('nan:', t)

            x_t_list.append(x_next if enable_grads else x_next.detach())
            
            if not if_output_diffusion_process:
                x_t_list.pop(0)
            
        if if_output_diffusion_process:
            return x_t_list
        else:
            return x_t_list[-1]

    @classmethod
    def from_existing(cls, base_model):
        guided = cls.__new__(cls)
        guided.__dict__ = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
                           for k, v in base_model.__dict__.items()}
        return guided

 
        
class GuidedDiffNet(DiffNet):
    def __init__(self, *args, cond_data=None, data_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cond_data = cond_data
        self._data_path = data_path

    @property
    def cond_data(self):
        return self._cond_data

    @property
    def data_path(self):
        return self._data_path

    @classmethod
    def from_pretrained(cls, checkpoint_path, cond_data=None, data_path=None):
        guided_model = cls.load_from_checkpoint(checkpoint_path, cond_data=cond_data, data_path=data_path)
        guided_model.joint_diffusion = GuidedJointDiffusion.from_existing(guided_model.joint_diffusion)
        return guided_model

    @cond_data.setter
    def cond_data(self, value):
        self._cond_data = value

    def plot_predictions(
        self,
        b_idx,
        data,
        eval_mask,
        traj_refine,
        rec_traj,
        gt_eval,
        goal_point,
        num_scenes,
        num_agents_per_scene,
        exp_id = ""
    ):
        """Handles only the plotting/visualization logic for latent_generator."""

        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos

        rec_traj_world = torch.matmul(rec_traj[:, :, :, :2],
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)

        marginal_trajs = traj_refine[eval_mask, :, :, :2]
        marg_traj_world = torch.matmul(marginal_trajs,
                                    rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        marg_traj_world = marg_traj_world.detach().cpu().numpy()

        gt_eval_world = torch.matmul(gt_eval[:, :, :2], rot_mat) + origin_eval[:, :2].reshape(-1, 1, 2)
        gt_eval_world = gt_eval_world.detach().cpu().numpy()

        goal_point_world = torch.matmul(goal_point[:, None, None, :],
                                        rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        goal_point_world = goal_point_world.squeeze(1).squeeze(1).detach().cpu().numpy()

        img_folder = 'visual'
        sub_folder = 'opd_method'
        rec_traj_world_np = rec_traj_world.detach().cpu().numpy()

        for i in range(num_scenes):
            start_id = torch.sum(num_agents_per_scene[:i])
            end_id = torch.sum(num_agents_per_scene[:i+1])

            if end_id - start_id == 1:
                print(f"Not plotting scenario {b_idx} because it has just one agent")
                continue

            temp = gt_eval[start_id:end_id]
            temp_start = temp[:, 0, :]
            temp_end = temp[:, -1, :]
            norm = torch.norm(temp_end - temp_start, dim=-1)

            if torch.max(norm) < 10:
                print(f"Not plotting scenario {b_idx} because agents didn't move")
                continue

            scenario_id = data['scenario_id'][i]
            print(f"Plotting scenario {b_idx}")

            base_path_to_data = Path(os.path.join(self.data_path, "raw"))
            scenario_folder = base_path_to_data / scenario_id

            static_map_path = scenario_folder / f"log_map_archive_{scenario_id}.json"
            scenario_path = scenario_folder / f"scenario_{scenario_id}.parquet"

            scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
            static_map = ArgoverseStaticMap.from_json(static_map_path)

            viz_output_dir = Path(img_folder) / sub_folder
            os.makedirs(viz_output_dir, exist_ok=True)

            viz_save_path = viz_output_dir / (f'viz_{b_idx}'+exp_id+'.jpg')

            additional_traj = {
                'gt': gt_eval_world[start_id:end_id],
                'goal_point': goal_point_world,
                'marg_traj': marg_traj_world[start_id:end_id],
                'rec_traj': rec_traj_world_np[start_id:end_id],
            }

            traj_visible = {
                'gt': False,
                'gt_goal': False,
                'goal_point': False,
                'marg_traj': False,
                'rec_traj': True,
            }

            visualize_scenario_prediction(scenario, static_map, additional_traj, traj_visible, viz_save_path, show_legend = True)

    

    def latent_generator(self, latent_point, b_idx, plot=False, enable_grads=False, return_pred_only=True,exp_id = ""):
        """Runs diffusion from a latent and reconstructs trajectories."""
        if self.cond_data is None:
            raise RuntimeError("cond_data must be set before calling latent_generator().")

        data = self.cond_data.to(self.device)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        # Masks & encoder
        reg_mask = data['agent']['predict_mask'][:, self.num_historical_steps:]   # [N_total, 60] (True = predict)
        pred, scene_enc = self.qcnet(data)

        # Mode heads (unchanged)
        if self.output_head:
            traj_refine = torch.cat([
                pred['loc_refine_pos'][..., :self.output_dim],
                pred['loc_refine_head'],
                pred['scale_refine_pos'][..., :self.output_dim],
                pred['conc_refine_head'],
            ], dim=-1)
        else:
            traj_refine = torch.cat([
                pred['loc_refine_pos'][..., :self.output_dim],
                pred['scale_refine_pos'][..., :self.output_dim],
            ], dim=-1)

        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim],
                        data['agent']['target'][..., -1:]], dim=-1)              # [N_total, 60, K]

        # for best mode selection etc. (unchanged)
        l2_norm = (torch.norm(traj_refine[..., :self.output_dim] -
                            gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)

        if self.s_mean is None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(gt.device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(gt.device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(gt.device)
            else:
                self.V_k = self.VT_k.transpose(0, 1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(gt.device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(gt.device)

        # Eval agents mask
        eval_mask = data['agent']['category'] >= 2

        # --- your preprocessing for gt_n / target_mode (unchanged) ---
        mask = (data['agent']['category'] >= 2) & (reg_mask[:, -1] == True) & (reg_mask[:, 0] == True)
        gt_n = gt[mask][..., :self.output_dim]
        gt_n[0, :, :] = (gt_n[0, :, :] - gt_n[0, 0:1, :]) / 4 * 3 + gt_n[0, 0:1, :]
        reg_mask_n = reg_mask[mask]
        num_agent = gt_n.size(0)
        reg_start_list, reg_end_list = [], []
        for i in range(num_agent):
            start, end = [], []
            for j in range(59):
                if reg_mask_n[i, j] == True and reg_mask_n[i, j + 1] == False:
                    start.append(j)
                elif reg_mask_n[i, j] == False and reg_mask_n[i, j + 1] == True:
                    end.append(j + 1)
            reg_start_list.append(start)
            reg_end_list.append(end)
        for i in range(num_agent):
            count = 0
            for j in range(59):
                if reg_mask_n[i, j] == False:
                    start_id = reg_start_list[i][count]
                    end_id = reg_end_list[i][count]
                    start_pt = gt_n[i, start_id]
                    end_pt = gt_n[i, end_id]
                    gt_n[i, j] = start_pt + (end_pt - start_pt) / (end_id - start_id) * (j - start_id)
                    if j == end_id - 1:
                        count += 1

        flat_gt = gt_n.reshape(gt_n.size(0), -1)
        k_vector = torch.matmul(flat_gt - self.s_mean, self.VT_k)
        rec_flat_gt = torch.matmul(k_vector, self.V_k) + self.s_mean
        rec_gt = rec_flat_gt.view(-1, 60, 2)

        target_mode = torch.matmul(flat_gt - self.s_mean, self.VT_k)
        target_mode = self.normalize(target_mode, self.latent_mean, self.latent_std)

        # choose best mode (unchanged)
        fde_marginal_gt = torch.norm(traj_refine[:, :, -1, :2] - gt[:, -1, :2].unsqueeze(1), dim=-1)
        if self.choose_best_mode == 'FDE':
            best_fde_mode = fde_marginal_gt.argmin(dim=-1)
            best_l2_mode = l2_norm.argmin(dim=-1)
            best_mode = best_l2_mode
            fde_valid = reg_mask[:, -1] == True
            best_mode[fde_valid] = best_fde_mode[fde_valid]
        elif self.choose_best_mode == 'ADE':
            best_l2_mode = l2_norm.argmin(dim=-1)
            best_mode = best_l2_mode

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError(f'{self.dataset} is not a valid dataset')

        valid_mask_eval = reg_mask[eval_mask]
        gt_eval = gt[eval_mask]

        # Build marginal modes (unchanged)
        marginal_trajs = traj_refine[eval_mask, :, :, :2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0), self.num_modes, -1)
        marginal_mode = torch.matmul(
            (marginal_trajs - self.s_mean.unsqueeze(1)).permute(1, 0, 2),
            self.VT_k.unsqueeze(0).repeat(self.num_modes, 1, 1)
        )
        marginal_mode = marginal_mode.permute(1, 0, 2)
        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)
        marg_mean = marginal_mode.mean(dim=1)
        marg_std = marginal_mode.std(dim=1) + self.std_reg

        if self.cond_norm:
            marginal_mode = self.normalize(marginal_mode, marg_mean, marg_std)
            target_mode = self.normalize(target_mode, marg_mean, marg_std)
            mean = torch.zeros_like(marg_mean)
            std = torch.ones_like(marg_std)
        else:
            mean = marg_mean
            std = marg_std

        # Diffusion from latent (eval agents only)
        self.joint_diffusion.eval()
        num_samples = 1
        reverse_steps = None
        device = traj_refine.device


        pred_modes = self.joint_diffusion.from_latent(
            latent_point.to(device), data=data, scene_enc=scene_enc,
            mean=mean, std=std, mm=marginal_mode,
            mmscore=pi.exp()[eval_mask],
            stride=self.sampling_stride,
            reverse_steps=reverse_steps,
            eval_mask=eval_mask,
            enable_grads=enable_grads
        )


        if self.cond_norm:
            pred_modes = self.unnormalize(pred_modes, marg_mean, marg_std)

        # Back to trajectory space (local)
        unnorm_pred_modes = self.unnormalize(pred_modes, self.latent_mean, self.latent_std)
        rec_traj = torch.matmul(
            unnorm_pred_modes.permute(1, 0, 2),
            (self.V_k).unsqueeze(0).repeat(1, num_samples, 1)
        ) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1, 0, 2)                         # [N_eval, 1, 60, 2]
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1), self.num_future_steps, 2)

        # ---------- RETURN BRANCHES ----------
        if return_pred_only:
            # Transform eval agents only to world coords (unchanged behavior)
            origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
            theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
            cos, sin = theta_eval.cos(), theta_eval.sin()
            rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos

            rec_traj_world = torch.matmul(rec_traj[:, :, :, :2], rot_mat.unsqueeze(1)) \
                            + origin_eval[:, :2].reshape(-1, 1, 1, 2)

            if plot:
                goal_point = gt_eval[:, -1, :2].detach().clone()
                batch_idx = data['agent']['batch'][eval_mask]
                num_scenes = batch_idx[-1].item() + 1
                num_agents_per_scene = torch.bincount(batch_idx, minlength=batch_idx.max().item() + 1)
                self.plot_predictions(
                    b_idx=b_idx, data=data, eval_mask=eval_mask,
                    traj_refine=traj_refine, rec_traj=rec_traj,
                    gt_eval=gt_eval, goal_point=goal_point,
                    num_scenes=num_scenes, num_agents_per_scene=num_agents_per_scene, exp_id = exp_id
                )

            return rec_traj_world.squeeze(1)   # [N_eval, 60, 2]

        else:
            # ---------- NEW: return full + components (scene-consistent) ----------
            N_total = data['agent']['category'].size(0)
            T = self.num_future_steps

            # Predicted local trajectories for all eval agents (global order)
            pred_eval_local_all = rec_traj.squeeze(1)        # [N_eval_all, 60, 2]
            mask_eval_all = reg_mask[eval_mask].unsqueeze(-1)  # [N_eval_all, 60, 1]
            gt_eval_local_all = gt[eval_mask][..., :2]       # [N_eval_all, 60, 2]

            # Start with GT for all agents
            full_local = gt[:, :T, :2].clone()               # [N_total, 60, 2]

            # Fused eval (for display/logging only)
            fused_eval_local_all = pred_eval_local_all * mask_eval_all.float() \
                                   + gt_eval_local_all * (1.0 - mask_eval_all.float())
            full_local[eval_mask] = fused_eval_local_all

            # Rotate ALL agents into world coords
            origin_all = data['agent']['position'][:, self.num_historical_steps - 1, :2]   # [N_total, 2]
            theta_all = data['agent']['heading'][:, self.num_historical_steps - 1]
            cos_all, sin_all = theta_all.cos(), theta_all.sin()
            rot_all = torch.zeros(N_total, 2, 2, device=self.device)
            rot_all[:, 0, 0] = cos_all
            rot_all[:, 0, 1] = sin_all
            rot_all[:, 1, 0] = -sin_all
            rot_all[:, 1, 1] = cos_all

            full_world = torch.einsum('ntc,ncd->ntd', full_local, rot_all) \
                        + origin_all[:, :2].unsqueeze(1)   # [N_total, 60, 2]

            # --------- SCENE-CONSISTENT INDICES/MASKS ----------
            batch_ids = data['agent']['batch']  # [N_total]
            # scene_mask is True for agents belonging to this b_idx.
            if (batch_ids == b_idx).any():
                scene_mask = (batch_ids == b_idx)
            else:
                # If cond_data is single-scene, b_idx likely doesn't exist here; take all agents.
                scene_mask = torch.ones_like(batch_ids, dtype=torch.bool)

            # Global eval indices (positions in [0..N_total-1])
            eval_idx_all = torch.nonzero(eval_mask, as_tuple=False).squeeze(-1)  # [N_eval_all]

            # Among those eval agents, pick only those belonging to this scene
            # This yields positions inside the eval list
            scene_positions_in_eval = torch.nonzero(scene_mask[eval_mask],
                                                    as_tuple=False).squeeze(-1)  # [N_eval_scene]

            # Row indices into full_world for the scene’s eval agents
            eval_idx_scene = eval_idx_all[scene_positions_in_eval]               # [N_eval_scene]

            # Slice predicted things to scene only
            pred_eval_local_scene = pred_eval_local_all[scene_positions_in_eval]  # [N_eval_scene, 60, 2]
            mask_eval_scene = mask_eval_all[scene_positions_in_eval]              # [N_eval_scene, 60, 1]

            # Return full world + scene-consistent pieces
            return full_world, pred_eval_local_scene, mask_eval_scene, eval_idx_scene