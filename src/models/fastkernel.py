
"""
This file is from: https://github.com/Lilac-Lee/FastKernelSF/blob/main/kernel_flow.py
with some modifications to have unified format with all benchmark.
"""

import dztimer, torch, copy, math
import torch.nn as nn


import torch.nn.functional as F
import FastGeodis
from copy import deepcopy


from .unsfp.model import EarlyStopping, my_chamfer_fn
from .basic import cal_pose0to1
from assets.cuda.chamfer3D import nnChamferDis
MyCUDAChamferDis = nnChamferDis()

def init_params(param_shape, init_method='', init_scaling=1.0, device='cuda:0', requires_grad=True):
    if init_method == 'same_as_linear':
        stdv = 1. / math.sqrt(param_shape[1]*param_shape[2])
        param = torch.distributions.Uniform(-stdv, stdv).sample(param_shape)
    
    param = param.to(device)
    if requires_grad:
        param.requires_grad = True
    
    return param

class DT:
    # Calculate the distance transform efficiently using tensors
    def __init__(self, pc1, pc2, grid_factor, device='cuda:0', use_dt_loss=True):
        self.device = device
        self.grid_factor = grid_factor
        
        pc1_min = torch.min(pc1, 1)[0].squeeze(0)
        pc1_max = torch.max(pc1, 1)[0].squeeze(0)
        pc2_min = torch.min(pc2, 1)[0].squeeze(0)
        pc2_max = torch.max(pc2, 1)[0].squeeze(0)
        
        xmin_int, ymin_int, zmin_int = torch.floor(torch.where(pc1_min<pc2_min, pc1_min, pc2_min\
                                            ) * grid_factor-1) / grid_factor
        xmax_int, ymax_int, zmax_int = torch.ceil(torch.where(pc1_max>pc2_max, pc1_max, pc2_max\
                                            )* grid_factor+1) / grid_factor
            
        sample_x = ((xmax_int - xmin_int) * grid_factor).ceil().int() + 2
        sample_y = ((ymax_int - ymin_int) * grid_factor).ceil().int() + 2
        sample_z = ((zmax_int - zmin_int) * grid_factor).ceil().int() + 2
        
        self.Vx = torch.linspace(0, sample_x, sample_x+1, device=self.device)[:-1] / grid_factor + xmin_int
        self.Vy = torch.linspace(0, sample_y, sample_y+1, device=self.device)[:-1] / grid_factor + ymin_int
        self.Vz = torch.linspace(0, sample_z, sample_z+1, device=self.device)[:-1] / grid_factor + zmin_int
        
        # NOTE: build a binary image first, with 0-value occuppied points, then use opencv function
        grid_x, grid_y, grid_z = torch.meshgrid(self.Vx, self.Vy, self.Vz, indexing="ij")
        self.grid = torch.stack([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1).float().squeeze()
        
        if use_dt_loss:
            H, W, D, _ = self.grid.size()
            pts_mask = torch.ones(H, W, D, device=device)
            self.pts_sample_idx_x = ((pc2[...,0:1] - self.Vx[0]) * self.grid_factor).round()
            self.pts_sample_idx_y = ((pc2[...,1:2] - self.Vy[0]) * self.grid_factor).round()
            self.pts_sample_idx_z = ((pc2[...,2:3] - self.Vz[0]) * self.grid_factor).round()
            pts_mask[self.pts_sample_idx_x.long(), self.pts_sample_idx_y.long(), self.pts_sample_idx_z.long()] = 0.
            
            iterations = 1
            image_pts = torch.zeros(H, W, D, device=device).unsqueeze(0).unsqueeze(0)
            pts_mask = pts_mask.unsqueeze(0).unsqueeze(0)
            self.D = FastGeodis.generalised_geodesic3d(
                image_pts, pts_mask, [1./self.grid_factor, 1./self.grid_factor, 1./self.grid_factor], 1e10, 0.0, iterations
            ).squeeze()
        else:
            self.D = deepcopy(self.grid)
            
    def torch_bilinear_distance(self, Y):
        H, W, D = self.D.size()
        target = self.D[None, None, ...]
        
        sample_x = ((Y[:,0:1] - self.Vx[0]) * self.grid_factor).clip(0, H-1)
        sample_y = ((Y[:,1:2] - self.Vy[0]) * self.grid_factor).clip(0, W-1)
        sample_z = ((Y[:,2:3] - self.Vz[0]) * self.grid_factor).clip(0, D-1)
        
        sample = torch.cat([sample_x, sample_y, sample_z], -1)
        
        # NOTE: normalize samples to [-1, 1]
        sample = 2 * sample
        sample[...,0] = sample[...,0] / (H-1)
        sample[...,1] = sample[...,1] / (W-1)
        sample[...,2] = sample[...,2] / (D-1)
        sample = sample -1
        sample_ = torch.cat([sample[...,2:3], sample[...,1:2], sample[...,0:1]], -1)
        
        dist = F.grid_sample(target, sample_.view(1,-1,1,1,3), mode="bilinear", align_corners=True).view(-1)
        
        return dist
    

# NOTE: simple 3D encoding. All method use 3 directions.
class encoding_func_3D:
    def __init__(self, name, param=None, device='cpu', dim_x=3):
        self.name = name

        if name == 'none': self.dim=2
        elif name == 'basic': self.dim=4
        else:
            self.dim = param[1]
            if name == 'RFF':
                self.b = param[0]*torch.randn(1,dim_x,int(param[1]/2), device=device)   # make it to have batch_size=1
            else:
                print('Undifined encoding!')
                
    def __call__(self, x):
        if self.name == 'none':
            return x
        elif self.name == 'basic':
            emb = torch.cat((torch.sin((2.*math.pi*x)),torch.cos((2.*math.pi*x))),-1)
            emb = emb/(emb.norm(dim=1).max())
        elif (self.name == 'RFF')|(self.name == 'rffb'):
            emb = torch.cat((torch.sin((2.*math.pi*x).bmm(self.b)),torch.cos((2.*math.pi*x).bmm(self.b))),-1)   # batch_size=1
        return emb

# --iters 1000 
# --earlystopping --early_patience 10 --early_min_delta 0.001 
# --kernel_grid --grid_factor 0.2 --model pe -
# -weight_decay 0. --use_dt_loss --dt_grid_factor 10. 
# --use_all_points --alpha_init_method same_as_linear 
# --alpha_init_scaling 1. 
# --reg_name l1 --reg_scaling 5. --epsilon 1e-7 
# --pe_type RFF --pe_dim 256 --pe_sigma 0.01 --log_sigma 10. 
# --alpha_lr 0.008    
class FastKernel(nn.Module):
    def __init__(self, itr_num=1000, min_delta=0.001, early_patience=100,
                 grid_factor=0.2, dt_grid_factor=10, alpha_lr=0.008,
                 use_dt_loss=True, kernel_grid=True, model='pe',
                pe_type='RFF', pe_dim=256, pe_sigma=0.01, log_sigma=10,
                alpha_init_scaling=1., reg_name='l1', reg_scaling=5.,
                 verbose=False, point_cloud_range = [-51.2, -51.2, -3, 51.2, 51.2, 3],
                 voxel_size=None, grid_feature_size=None):
        super().__init__()
        
        self.iteration_num = itr_num
        self.min_delta = min_delta
        self.alpha_lr = alpha_lr
        self.early_patience = early_patience
        self.dt_grid_factor = dt_grid_factor
        self.grid_factor = grid_factor
        self.use_dt_loss = use_dt_loss
        self.kernel_grid = kernel_grid
        self.kernel_type = 'gaussian'
        self.alpha_init_method = 'same_as_linear'
        self.alpha_init_scaling = alpha_init_scaling
        self.weight_decay = 0.
        self.reg_name = reg_name
        self.reg_scaling = reg_scaling

        self.model = model
        self.pe_type = pe_type
        self.pe_dim = pe_dim
        self.pe_sigma = pe_sigma
        self.log_sigma = log_sigma

        self.verbose = verbose
        self.point_cloud_range = point_cloud_range
        self.timer = dztimer.Timing()
        self.timer.start("FastKernel Model Inference")

    def optimize(self, pc1, pc2):
        device = pc1.device
        early_stopping = EarlyStopping(patience=self.early_patience, min_delta=self.min_delta)

        # ANCHOR: loss preprocessing -- do not need since we have GT flow
        if self.use_dt_loss:
            dt = DT(pc1, pc2, grid_factor=self.dt_grid_factor, device=device, use_dt_loss=True)
        
        # ANCHOR: kernel function
        if self.kernel_grid:   # K(p1,p*)
            # ANCHOR: for complex encoding grid computation, similar to building a DT map
            complex_grid = DT(pc1, pc2, grid_factor=self.grid_factor, device=device, use_dt_loss=False)
            grid_pts = complex_grid.grid
            pc2_ = grid_pts.reshape(-1, pc1.shape[-1]).unsqueeze(0)
        else:
            pc2_ = pc2.clone()

        if self.model == 'none':
            # NOTE: for point-based kernel
            feats1_loc = pc1.clone()
            feats2_loc = pc2_.clone()
        elif self.model == 'pe':	
            pe3d = encoding_func_3D(self.pe_type, param=(self.pe_sigma, self.pe_dim), device=device, dim_x=3)
            feats1_loc = pe3d(pc1)
            feats2_loc = pe3d(pc2_)

        # NOTE: pc1 -- observation; kernel grid -- known points; therefore, alpha should have the same size as kernel grid.
        feats1_gram = torch.linalg.norm(feats1_loc, dim=-1, keepdim=True) ** 2
        feats2_gram = torch.linalg.norm(feats2_loc, dim=-1, keepdim=True) ** 2
        feats1_dot_feats2 = torch.einsum('ijk,ilk->ijl', feats1_loc, feats2_loc)
        rbf = feats1_gram + feats2_gram.permute(0,2,1) - 2 * feats1_dot_feats2
        
        if self.kernel_type == 'gaussian':
            rbf = torch.exp(-1./(2*self.log_sigma) * rbf)   # BxNxK

        # ANCHOR: set optimization parameters, and begin optimization
        alpha = init_params((1, rbf.shape[2], 3), init_method=self.alpha_init_method, init_scaling=self.alpha_init_scaling, device=device, requires_grad=True)
        param = [{'params': alpha, 'lr': self.alpha_lr, 'weight_decay': self.weight_decay}]
        optimizer = torch.optim.Adam(param)

        # ANCHOR: initialize best metrics
        best_forward = {'loss': 1e10, 'flow': None}

        for itr_ in range(self.iteration_num):
            flow_pred = alpha.transpose(1,2).bmm(rbf.transpose(1,2)).transpose(1,2)
            pc1_deformed = pc1 + flow_pred
            
            if self.use_dt_loss:
                loss_corr = dt.torch_bilinear_distance(pc1_deformed.squeeze(0)).mean()
            elif self.use_chamfer:
                loss_corr = MyCUDAChamferDis.truncated_dis(pc2, pc1_deformed)
               
            # NOTE: add TV regularizer?
            if self.reg_name == 'l1':
                reg_scaled = self.reg_scaling * alpha.abs().mean()
            elif self.reg_name == 'l2':
                reg_scaled = self.reg_scaling * (alpha ** 2).mean()
            
            if self.reg_name != 'none':
                loss = loss_corr + reg_scaled
            else:
                loss = loss_corr

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            flow_pred_final = pc1_deformed - pc1

            if loss <= best_forward['loss']:
                best_forward['loss'] = loss.item()
                best_forward['flow'] = flow_pred_final.squeeze(0)

            if early_stopping.step(loss):
                break

        return best_forward
    
    def range_limit_(self, pc):
        """
        Limit the point cloud to the given range.
        """
        mask = (pc[:, 0] >= self.point_cloud_range[0]) & (pc[:, 0] <= self.point_cloud_range[3]) & \
               (pc[:, 1] >= self.point_cloud_range[1]) & (pc[:, 1] <= self.point_cloud_range[4]) & \
               (pc[:, 2] >= self.point_cloud_range[2]) & (pc[:, 2] <= self.point_cloud_range[5])
        return pc[mask], mask
    
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])

        
        pose_flows = []
        pose_0to1s = []
        for batch_id in range(batch_sizes):
            self.timer[0].start("Data Processing")
            pc0 = batch["pc0"][batch_id]
            pc1 = batch["pc1"][batch_id]
            selected_pc0, rm0 = self.range_limit_(pc0)
            selected_pc1, _ = self.range_limit_(pc1)
            self.timer[0][0].start("pose")
            pose_0to1 = cal_pose0to1(batch["pose0"][batch_id], batch["pose1"][batch_id])
            pose_0to1s.append(pose_0to1)
            self.timer[0][0].stop()
            
            self.timer[0][1].start("transform")
            # transform selected_pc0 to pc1
            transform_pc0 = selected_pc0 @ pose_0to1[:3, :3].T + pose_0to1[:3, 3]
            self.timer[0][1].stop()
            pose_flows.append(transform_pc0 - selected_pc0)

            self.timer[0].stop()

            with torch.inference_mode(False):
                with torch.enable_grad():
                    # pc0_ = transform_pc0.clone().detach().requires_grad_(True)
                    # pc1_ = selected_pc1.clone().detach().requires_grad_(True)
                    model_res = self.optimize(transform_pc0.unsqueeze(0), selected_pc1.unsqueeze(0))
            
            final_flow = torch.zeros_like(pc0)
            final_flow[rm0] = model_res["flow"]
            
        res_dict = {"flow": final_flow,
                    "pose_flow": pose_flows,
                    "pose_0to1": pose_0to1s
                    }
        
        return res_dict