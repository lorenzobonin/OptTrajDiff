from predictors import DiffNet
from modules import JointDiffusion
import torch
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from typing import Dict, List, Mapping, Optional
import copy
import numpy as np
import time
# from pynvml import *
# nvmlInit()

class GuidedJointDiffusion(JointDiffusion):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def from_latent(self,
               x_T,
               data: HeteroData,
               scene_enc: Mapping[str, torch.Tensor],
               mean = None,
               mm = None,
               mmscore = None,
               if_output_diffusion_process = False,
               reverse_steps = None,
               eval_mask = None,
               stride=20,
               cond_gen = None
               ) -> Dict[str, torch.Tensor]:
        
        if reverse_steps is None:
            reverse_steps = self.var_sched.num_steps

        num_samples = 1
        
        device = mean.device

        num_agents = mean.size(0)
        
        x_t_list = [x_T]
        torch.cuda.empty_cache()
        # pt = time.time()
        # h = nvmlDeviceGetHandleByIndex(2)
        # info = nvmlDeviceGetMemoryInfo(h)
        # pu = info.used/ (1024 ** 3)
        for t in range(reverse_steps, 0, -stride):

            beta = self.var_sched.betas[t]
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            alpha_bar_next = self.var_sched.alpha_bars[t-stride]
            c0 = 1 / torch.sqrt(alpha)
            c1 = (1-alpha) / torch.sqrt(1 - alpha_bar)
            sigma = self.var_sched.get_sigmas(t, 0)
            
            x_t = x_t_list[-1]
            if cond_gen != None:
                [idx, target_mode] = cond_gen
                x_t[idx,:,:] = target_mode.unsqueeze(0).repeat(num_samples,1)
            
            with torch.no_grad():
                beta_emb = beta.to(device).repeat(num_agents * num_samples).unsqueeze(-1)
                g_theta = self.net(copy.deepcopy(x_t), beta_emb, data, scene_enc, num_samples = num_samples, mm = mm, mmscore = mmscore, eval_mask=eval_mask)
        
            ### ddim ###
            x0_t = (x_t - g_theta * (1 - alpha_bar).sqrt()) / alpha_bar.sqrt()
            x_next = alpha_bar_next.sqrt() * x0_t + (1 - alpha_bar_next).sqrt() * g_theta
            ############

            if True in torch.isnan(x_next):
                print('nan:',t)
            x_t_list.append(x_next.detach())
            if not if_output_diffusion_process:
                x_t_list.pop(0)
            
        # cost_time = (time.time() - pt)/10
        # h = nvmlDeviceGetHandleByIndex(2)
        # info = nvmlDeviceGetMemoryInfo(h)
        # cu = info.used/ (1024 ** 3)
        # cost_u = cu - pu
        # self.GPU_incre_memory.append(cost_u)
        # self.infer_time_per_step.append(cost_time)
            
        if if_output_diffusion_process:
            return x_t_list
        else:
            return x_t_list[-1]

    @classmethod
    def from_existing(cls, base_model):

        guided = cls.__new__(cls)
        guided.__dict__ = copy.deepcopy(base_model.__dict__)
        return guided

 
        
class GuidedDiffNet(DiffNet):
    def __init__(self, *args, cond_data=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._cond_data = cond_data

    @property
    def cond_data(self):
        return self._cond_data

    @classmethod
    def from_pretrained(cls, checkpoint_path, cond_data=None):
        # Load base model
        guided_model = cls.load_from_checkpoint(checkpoint_path, qcnet_ckpt_path=qcnet_ckpt_path, cond_data=cond_data)

        #guided_model = cls(cond_data)
        #guided_model.load_state_dict(base_model.state_dict(), strict=False)
        guided_model.joint_diffusion = GuidedJointDiffusion.from_existing(guided_model.joint_diffusion)

        return guided_model

    @cond_data.setter
    def cond_data(self, value):
        self._cond_data = value

    def latent_generator(self, latent_point):
        if self.cond_data is None:
            raise RuntimeError("cond_data must be set before calling latent_generator().")

        data = self.cond_data.to(self.device)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        
        pred, scene_enc = self.qcnet(data)
        
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        device = traj_refine.device
        pi = pred['pi']

        if self.dataset == 'argoverse_v2':
            eval_mask = data['agent']['category'] >= 2
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))
        
        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        
        if self.s_mean == None:
            s_mean = np.load(self.path_pca_s_mean)
            self.s_mean = torch.tensor(s_mean).to(device)
            VT_k = np.load(self.path_pca_VT_k)
            self.VT_k = torch.tensor(VT_k).to(device)
            if self.path_pca_V_k != 'none':
                V_k = np.load(self.path_pca_V_k)
                self.V_k = torch.tensor(V_k).to(device)
            else:
                self.V_k = self.VT_k.transpose(0,1)
            latent_mean = np.load(self.path_pca_latent_mean)
            self.latent_mean = torch.tensor(latent_mean).to(device)
            latent_std = np.load(self.path_pca_latent_std) * 2
            self.latent_std = torch.tensor(latent_std).to(device)
            
        marginal_trajs = traj_refine[eval_mask,:,:,:2]
        marginal_trajs = marginal_trajs.view(marginal_trajs.size(0),self.num_modes,-1)
        marginal_mode = torch.matmul((marginal_trajs-self.s_mean.unsqueeze(1)).permute(1,0,2), self.VT_k.unsqueeze(0).repeat(self.num_modes,1,1))
        marginal_mode = marginal_mode.permute(1,0,2)
        marginal_mode = self.normalize(marginal_mode, self.latent_mean, self.latent_std)
 
        mean = marginal_mode.mean(dim=1)

        self.joint_diffusion.eval()
        num_samples = 1

        reverse_steps = 70
        
        pred_modes = self.joint_diffusion.from_latent(latent_point.to(device), data = data, scene_enc = scene_enc, 
                                                mean = mean, mm = marginal_mode, 
                                                mmscore = pi.exp()[eval_mask],
                                                stride=self.sampling_stride,
                                                reverse_steps=reverse_steps,
                                                eval_mask=eval_mask)
        
        pred_modes = self.unnormalize(pred_modes,self.latent_mean, self.latent_std)
        rec_traj = torch.matmul(pred_modes.permute(1,0,2), (self.V_k).unsqueeze(0).repeat(num_samples,1,1)) + self.s_mean.unsqueeze(0)
        rec_traj = rec_traj.permute(1,0,2)
        rec_traj = rec_traj.view(rec_traj.size(0), rec_traj.size(1),self.num_future_steps,2)

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

        return rec_traj_world