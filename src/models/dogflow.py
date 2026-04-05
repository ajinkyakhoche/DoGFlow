
"""
# Created: 2025-04-10 16:58
# Copyright (C) 2025-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche (https://ajinkyakhoche.github.io/)
#
# This file is part of DoGFlow (https://ajinkyakhoche.github.io/DoGFlow/).
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
"""

import dztimer, torch, copy
import torch.nn as nn

from .unsfp.model import Neural_Prior, EarlyStopping, my_chamfer_fn
from assets.cuda.chamfer3D import ChamferDis, nnChamferDis
MyCUDAChamferDis = nnChamferDis()
from .basic import cal_pose0to1
from ..lossfuncs import seflowLoss

from scipy.optimize import lsq_linear
from scipy.sparse.csgraph import connected_components
import rerun as rr
import rerun.blueprint as rrb
from src.utils.o3d_view import color_map
from src.utils.mics import flow_to_rgb
from src.models.basic import cal_pose0to1

import numpy as np
import open3d as o3d
import math
import cv2

class DoGFlow(nn.Module):
    def __init__(self, dynamic_classification_threshold=0.25, max_association_error=3.0, min_association_error=0.1, 
                flow_bounds=[-4, -4, -0.2, 4, 4, 0.2], thresh_eucledian=3, thresh_speed=1.5, visualize=True,
                radar_to_lidar_tf_noise=[0,0,0,0,0,0], 
                voxel_size=None, grid_feature_size=None, point_cloud_range=None):
        super().__init__()
        
        self.timer = dztimer.Timing()
        self.timer.start("DoGFlow Model Inference")
        
        self.dynamic_classification_threshold = dynamic_classification_threshold
        self.max_association_error = max_association_error
        self.min_association_error = min_association_error
        self.flow_bounds = flow_bounds
        self.thresh_eucledian = thresh_eucledian
        self.thresh_speed = thresh_speed
        self.radar_to_lidar_tf_noise = radar_to_lidar_tf_noise

        self.visualize = visualize
        if visualize:
            rr.init("DoGFlow Visualization", spawn=True)
        # print all the inputs for logging
        print(
            "\n---LOG[model]:\n"
            f"  dynamic_classification_threshold: {dynamic_classification_threshold}\n"
            f"  max_association_error: {max_association_error}\n"
            f"  min_association_error: {min_association_error}\n"
            f"  flow_bounds: {flow_bounds}\n"
            f"  thresh_eucledian: {thresh_eucledian}\n"
            f"  thresh_speed: {thresh_speed}\n"
            f"  radar_to_lidar_tf_noise: {radar_to_lidar_tf_noise}.\n"
        )

    def log_vehicle_frames(self, radar0_to_refego_tf, cam0_to_refego_tf=np.array([]), cam0_size=np.array([]), cam0_intrinsic=np.array([])):
        # log ego vehicle frame
        rr.log(
            f"world/ego_vehicle/frames/ref_ego",
            rr.Transform3D(
                translation=np.zeros((3,)),
                rotation=rr.Quaternion(xyzw=np.array([0,0,0,1])), #rr.Quaternion(xyzw=rotation_xyzw),
                from_parent=False,
                scale=0.25,
            ),
            static=True,
        )
        
        # log radar frames
        for r_idx in range(radar0_to_refego_tf.shape[0]):
            rr.log(
                f"world/ego_vehicle/frames/radar_frame_{r_idx}",
                rr.Transform3D(
                    translation=radar0_to_refego_tf[r_idx,:3,3],
                    mat3x3=rr.datatypes.Mat3x3(radar0_to_refego_tf[r_idx,:3,:3]),
                    from_parent=False,
                    scale=0.1,
                ),
                static=True,
            )

    def radar_dynamic_classification(self, radar0, radar0_flow_raw, radar0_id, radar0_to_refego_tf, ego_motion):
        radar0_flow_mc = torch.zeros((radar0_flow_raw.shape[0], 1), dtype=torch.float32, device=radar0.device)
        A = torch.zeros_like(radar0_flow_raw, device=radar0.device)

        radar0_homo = torch.hstack((radar0[:, :3], torch.ones((radar0.shape[0], 1), device=radar0.device)))
        radar0_radarframe = torch.zeros_like(radar0[:, :3], device=radar0.device)
        
        for r_idx in torch.unique(radar0_id):
            r_idx = r_idx.item()
            radar0_radarframe[radar0_id == r_idx, :] = (torch.linalg.inv(radar0_to_refego_tf[r_idx]) @ radar0_homo[radar0_id == r_idx, :].T).T[:, :3]
            radar_bearing_vector = radar0_radarframe / (torch.linalg.norm(radar0_radarframe, dim=1, keepdim=True) + 1e-7)
            tf_ego2radar = torch.linalg.inv(radar0_to_refego_tf[r_idx])[:3, :3]
            A[radar0_id == r_idx, :] = radar_bearing_vector[radar0_id == r_idx, :] @ tf_ego2radar

        ego_motion_beam_component = (A @ -ego_motion[:3, 3].reshape(-1, 1)) * radar_bearing_vector
        radar0_flow_mc = radar0_flow_raw + ego_motion_beam_component
        radar0_dynamic_mask = torch.any(torch.abs(radar0_flow_mc) > self.dynamic_classification_threshold, dim=1)
        return radar0_dynamic_mask
    

    def make_extrinsic_noise(self, radar_to_lidar_tf_noise, device="cpu", dtype=torch.float32):
        """
        Build a 4x4 SE(3) transform from translation (tx, ty, tz) and
        rotations (rx, ry, rz) given in DEGREES around x, y, z axes.
        """
        tx, ty, tz, rx_deg, ry_deg, rz_deg = radar_to_lidar_tf_noise
        # convert to radians
        rx = math.radians(rx_deg)
        ry = math.radians(ry_deg)
        rz = math.radians(rz_deg)

        # rotation around x
        Rx = torch.eye(3, device=device, dtype=dtype)
        Rx[1, 1] = math.cos(rx)
        Rx[1, 2] = -math.sin(rx)
        Rx[2, 1] = math.sin(rx)
        Rx[2, 2] = math.cos(rx)

        # rotation around y
        Ry = torch.eye(3, device=device, dtype=dtype)
        Ry[0, 0] = math.cos(ry)
        Ry[0, 2] = math.sin(ry)
        Ry[2, 0] = -math.sin(ry)
        Ry[2, 2] = math.cos(ry)

        # rotation around z
        Rz = torch.eye(3, device=device, dtype=dtype)
        Rz[0, 0] = math.cos(rz)
        Rz[0, 1] = -math.sin(rz)
        Rz[1, 0] = math.sin(rz)
        Rz[1, 1] = math.cos(rz)

        # compose rotations: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx

        # build 4x4 transform
        T = torch.eye(4, device=device, dtype=dtype)
        T[:3, :3] = R
        T[:3, 3] = torch.tensor([tx, ty, tz], device=device, dtype=dtype)

        return T


    @torch.no_grad()
    def forward(self, batch):
        batch_sizes = len(batch["pose0"])

        pose_flows = []
        batch_final_flow = []
        pc_dynamic_mask_list = []
        radar0_flow_est_list = []
        for batch_id in range(batch_sizes):
            self.timer[0].start("Data Processing")
            pc0 = batch['pc0'][batch_id]
            pc1 = batch['pc1'][batch_id]
            gm0 = batch['gm0'][batch_id]
            gm1 = batch['gm1'][batch_id]
            pc0_range = torch.linalg.norm(pc0[:,:3], dim=1)
        
            pose0 = batch['pose0'][batch_id]
            pose1 = batch['pose1'][batch_id]
            ego_pose = cal_pose0to1(pose0, pose1)
            pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
            pose_flows.append(pose_flow)
            # ego_motion = batch['ego_motion'][batch_id]
            ego_motion = cal_pose0to1(pose0, pose1)
            
            pc0_cluster_label = batch['pc0_cluster_label'][batch_id][~gm0]
            # pc0_dynamic_mask = batch['pc0_dynamic_mask'][batch_id][~gm0]
            # pc0_associated_radar = batch['pc0_associated_radar'][batch_id][~gm0]
            
            radar0 = batch['radar0'][batch_id]
            radar0_flow_raw = batch['radar0_flow_raw'][batch_id]
            radar0_id = batch['radar0_id'][batch_id]
            radar0_to_refego_tf = batch['radar0_to_refego_tf'][batch_id]
            self.timer[0].stop()
            
            self.timer[1].start("Radar dynamic classification")
            # step 1: radar0 dynamic classification
            radar0_dynamic_mask = self.radar_dynamic_classification(radar0, radar0_flow_raw, radar0_id, radar0_to_refego_tf, ego_motion)
            self.timer[1].stop()

            self.timer[2].start("Nearest radar point association")
            # 03-12-2025: add noise to radar to lidar tf for robustness ablation
            T_noise = self.make_extrinsic_noise(self.radar_to_lidar_tf_noise, device=radar0.device, dtype=radar0.dtype)
            # Extract rotation (3x3) and translation (3,)
            R_noise = T_noise[:3, :3]          # (3,3)
            t_noise = T_noise[:3, 3]           # (3,)
            # radar0 shape: (N, 6) or (N, >=3)
            radar_xyz = radar0[:, :3]          # (N,3)
            # Apply p' = R * p + t
            radar_xyz_perturbed = (R_noise @ radar_xyz.T).T + t_noise
            # Create perturbed radar array
            radar0_perturbed = radar0.clone()
            radar0_perturbed[:, :3] = radar_xyz_perturbed

            ## step 2: use chamferdist to find nearest radar point for each lidar point
            # dist_lidar, dist_radar, idx_lidar, idx_radar = ChamferDis.apply(pc0[:,:3].contiguous(), radar0[:,:3].contiguous())
            dist_lidar, dist_radar, idx_lidar, idx_radar = ChamferDis.apply(pc0[:,:3].contiguous(), radar0_perturbed[:,:3].contiguous())
            
            # if the distance is greater than a threshold, then no associated radar point
            association_threshold = self.min_association_error + (self.max_association_error - self.min_association_error) * pc0_range / pc0_range.max()
            idx_lidar[dist_lidar>association_threshold] = -1
            pc0_associated_radar = idx_lidar
            self.timer[2].stop()

            self.timer[3].start("Lidar dynamic classification")
            # step 3: use pc0_cluster_label and pc0_associated_radar to compute lidar dynamic mask
            pc0_dynamic_mask = torch.zeros_like(pc0_cluster_label, dtype=torch.bool, device=pc0_cluster_label.device)
            for label_i in torch.unique(pc0_cluster_label):
                if label_i <= 0:
                    continue
                cluster_mask = pc0_cluster_label == label_i
                # for associated radar points, if most radar points are dynamic, then the cluster is dynamic
                if torch.any(idx_lidar[cluster_mask] != -1):
                    radar_cluster_mask = idx_lidar[cluster_mask] != -1
                    # figure out majority between unique true and false
                    cluster_associated_radar_dyn_mask = radar0_dynamic_mask[torch.unique(idx_lidar[cluster_mask][radar_cluster_mask])]
                    if torch.sum(cluster_associated_radar_dyn_mask) > len(cluster_associated_radar_dyn_mask) - torch.sum(cluster_associated_radar_dyn_mask):
                        pc0_dynamic_mask[cluster_mask] = True
            self.timer[3].stop()

            self.timer[4].start("Dynamic radar clustering")
            # step 4: cluster dynamic radar points uing connected component labeling
            ego_flow = - ego_motion[:3,3]
            radar0_cluster_label = torch.zeros(radar0.size(0), dtype= torch.int16).to(radar0.device) - 1
            if torch.count_nonzero(radar0_dynamic_mask) > 0:
                # cluster radar using connected component labeling:
                radar0_dynamic = radar0[radar0_dynamic_mask,:6]
                dist_mat_eucledian = radar0_dynamic[:, None, :3] - radar0_dynamic[None, :, :3]
                dist_mat_eucledian = (dist_mat_eucledian ** 2).sum(2) ** 0.5
                dist_mat_speed = radar0_dynamic[:, None, 3:6] - radar0_dynamic[None, :, 3:6]
                dist_mat_speed = (dist_mat_speed ** 2).sum(2) ** 0.5
                # adj_mat = dist_mat_eucledian + dist_mat_speed < 2.5
                adj_mat = torch.logical_and(dist_mat_eucledian < self.thresh_eucledian, dist_mat_speed < self.thresh_speed)
                radar0_cluster_label[radar0_dynamic_mask] = torch.tensor(connected_components(adj_mat.cpu().numpy(), directed=False)[1], dtype=torch.int16).to(radar0.device)
            self.timer[4].stop()

            self.timer[5].start("Radar cluster velocity estimation")
            # step 5: analytical estimation of flow per radar cluster
            radar0_flow_est = torch.zeros_like(radar0_flow_raw)
            # radar0_vel_cost = torch.zeros((radar0_flow_raw.shape[0])) -1
            # radar0_vel_residual = torch.zeros((radar0_flow_raw.shape[0])) -1
            # radar0_relative_residual = torch.zeros((radar0_flow_raw.shape[0])) -1
            # radar0_vel_rank = torch.zeros((radar0_flow_raw.shape[0])) -1
            # radar0_vel_gt = torch.zeros_like(radar0_flow_raw) -1

            for label_i in torch.unique(radar0_cluster_label):
                if label_i <= 0:
                    continue
                cluster_mask = radar0_cluster_label == label_i
                
                radar_roi = radar0[cluster_mask,:3]
                radar_roi_flow = radar0_flow_raw[cluster_mask,:3]
                radar_roi_id = radar0_id[cluster_mask]
                # radar_rel_flow = radar_roi_flow* 10
                radar_rel_flow = radar_roi_flow

                radar_roi_homo = torch.hstack((radar_roi, torch.ones((radar_roi.shape[0],1)).to(radar_roi.device)))
                radar_roi_radarframe = torch.zeros_like(radar_roi).to(radar_roi.device)
                
                # method 1
                A = torch.zeros_like(radar_roi).to(radar_roi.device)
                for r_idx in torch.unique(radar_roi_id):
                    r_idx = r_idx.item()
                    radar_roi_radarframe[radar_roi_id==r_idx,:] = (torch.linalg.inv(radar0_to_refego_tf[r_idx]) @ radar_roi_homo[radar_roi_id==r_idx,:].T).T[:,:3]
                    radar_bearing_vector = radar_roi_radarframe / (torch.linalg.norm(radar_roi_radarframe, axis=1, keepdims=True)+1e-7)
                    tf_ego2radar = torch.linalg.inv(radar0_to_refego_tf[r_idx])[:3,:3]
                    A[radar_roi_id==r_idx,:] = radar_bearing_vector[radar_roi_id==r_idx,:] @ tf_ego2radar

                b_full = radar_rel_flow + (A @ ego_flow.reshape(-1,1))*radar_bearing_vector
                b = torch.median(b_full / (radar_bearing_vector+1e-7), axis=1).values

                optimization_result = lsq_linear(A.cpu().numpy().astype('float64'), b.cpu().numpy().astype('float64'), bounds=(self.flow_bounds[:3], self.flow_bounds[3:6]))
                obj_flow_est = torch.tensor(optimization_result.x.astype('float32')).to(cluster_mask.device)
                if optimization_result.success and torch.linalg.norm(obj_flow_est[:]) < max(self.flow_bounds):
                    radar0_flow_est[cluster_mask,:] = obj_flow_est[:].squeeze()
                    # radar0_vel_cost[cluster_mask] = optimization_result.cost
                    # radar0_vel_residual[cluster_mask] = optimization_result.fun.astype(np.float32)
            self.timer[5].stop()

            self.timer[6].start("Label propagation")    
            # step 6: propagate the radar estimated velocity to all associated lidar points
            pc0_dynamic_cluster = pc0_cluster_label[pc0_dynamic_mask]
            pc0_dynamic_associated_radar = pc0_associated_radar[pc0_dynamic_mask]
            
            pc0_flow_est = torch.zeros((pc0.shape[0],3)).to(pc0.device)
            pc0_flow_est_dynamic = torch.zeros((pc0[pc0_dynamic_mask].shape[0],3)).to(pc0.device)
            pc0to1 = pc0[:,:3] @ ego_pose[:3, :3].T + ego_pose[:3, 3]
            for label_i in torch.unique(pc0_dynamic_cluster):
                cluster_mask = pc0_dynamic_cluster == label_i
                associated_radar = torch.unique(pc0_dynamic_associated_radar[cluster_mask])
                associated_radar = associated_radar[associated_radar != -1]
                if len(associated_radar) > 0:
                    if len(associated_radar) > 1:
                        # find all possible velocities
                        possible_velocities = torch.unique(radar0_flow_est[associated_radar.int(),:], dim=0)
                        dist_cost = []
                        for vel in possible_velocities:
                            # apply this velocity to the cluster
                            est_pc1 = pc0to1[pc0_dynamic_mask][cluster_mask] + vel #* 0.1
                            # calculate the residual
                            # dist0, _, _, _ = MyCUDAChamferDis.disid_res(torch.tensor(est_pc1).cuda(), torch.tensor(pc1[:,:3]).cuda())
                            dist0, _, _, _ = MyCUDAChamferDis.disid_res(est_pc1, pc1[:,:3])
                            # dist0 = dist0.cpu().numpy()
                            dist_cost.append(torch.sum(dist0))
                        # find the best velocity based on mininmum distance
                        best_vel_idx = torch.argmin(torch.hstack(dist_cost))
                        pc0_flow_est_dynamic[cluster_mask] = possible_velocities[best_vel_idx].reshape(1,3) 
                    else:
                        pc0_flow_est_dynamic[cluster_mask] = radar0_flow_est[associated_radar[0],:].reshape(1,3)

            pc0_flow_est[pc0_dynamic_mask] = pc0_flow_est_dynamic
            # pc0_vel_est_full = np.zeros((pc0_full.shape[0],3))
            # pc0_vel_est_full[~gm0] = pc0_vel_est

            batch_final_flow.append(pc0_flow_est)
            pc_dynamic_mask_list.append(pc0_dynamic_mask)
            radar0_flow_est_list.append(radar0_flow_est)
            self.timer[6].stop()

            if self.visualize:
                rr.log("scene_id", rr.TextDocument(str(batch["scene_id"][batch_id]+'_'+str(batch["timestamp"][batch_id]))))
                pcd_color = [0.5, 0.5, 0.5]
                point_size = 0.4
                # convert to numpy for ease of visualization
                radar0 = radar0.cpu().numpy()
                pc0 = pc0.cpu().numpy()
                pc0_flow_est = pc0_flow_est.cpu().numpy()
                radar0_flow_est = radar0_flow_est.cpu().numpy()
                radar0_dynamic_mask = radar0_dynamic_mask.cpu().numpy()
                pc0_dynamic_mask = pc0_dynamic_mask.cpu().numpy()
                pc0_associated_radar = pc0_associated_radar.cpu().numpy()
                pc0_cluster_label = pc0_cluster_label.cpu().numpy()
                pc0_dynamic_cluster = pc0_dynamic_cluster.cpu().numpy()
                pc0_dynamic_associated_radar = pc0_dynamic_associated_radar.cpu().numpy()
                radar0_cluster_label = radar0_cluster_label.cpu().numpy()
                radar0_id = radar0_id.cpu().numpy()
                ego_pose = ego_pose.cpu().numpy()
                ego_motion = ego_motion.cpu().numpy()
                radar0_to_refego_tf = radar0_to_refego_tf.cpu().numpy()
                radar0_perturbed = radar0_perturbed.cpu().numpy()

                # rr.set_time_seconds("timestamp", int(batch["timestamp"][batch_id]) * 1e-6)
                rr.set_time_nanos("timestamp", int(batch["timestamp"][batch_id]))
                
                # log ego vehicle and radar frames
                if 'cam0_to_refego_tf' in batch:
                    cam0_to_refego_tf = batch['cam0_to_refego_tf'][batch_id].cpu().numpy()
                    cam0_size = batch['cam0_size'][batch_id].cpu().numpy()
                    cam0_intrinsic = batch['cam0_intrinsic'][batch_id].cpu().numpy()
                    self.log_vehicle_frames(radar0_to_refego_tf, cam0_to_refego_tf, cam0_size, cam0_intrinsic)
                else:
                    self.log_vehicle_frames(radar0_to_refego_tf)
                # # log radar wise dynamic mask
                # flow_color_dynamic = np.tile(pcd_color, (radar0.shape[0], 1))
                # flow_color_dynamic[radar0_dynamic_mask.squeeze()] = color_map[1]
                # rr.log(f"world/ego_vehicle/radar/dynamic", rr.Points3D(radar0[:,:3], colors=flow_color_dynamic[:,:], radii=np.ones((radar0[:,:].shape[0],))*point_size))                         
                # # log radar acc to cluster
                # radar_cluster_color = np.tile(pcd_color, (radar0_cluster_label.shape[0], 1))
                # # log labels
                # for label_i in np.unique(radar0_cluster_label):
                #     if label_i > 0:
                #         radar_cluster_color[radar0_cluster_label == label_i] = color_map[label_i % len(color_map)]
                
                ## show only dynamic clusters
                # labels = ["cluster={:.2f}, v_rel={:.2f},{:.2f},{:.2f}".format(c_id, vx,vy,vz) for c_id, vx,vy,vz in np.hstack((radar0_cluster_label[radar0_dynamic_mask].reshape(-1,1), radar0[radar0_dynamic_mask,3:6])).round(2)]
                # rr.log(f"world/ego_vehicle/radar/labels", rr.Points3D(radar0[radar0_dynamic_mask,:3], colors=radar_cluster_color[radar0_dynamic_mask], radii=np.ones((radar0[radar0_dynamic_mask].shape[0],))*point_size, labels=labels))
                
                # labels = ["cluster={:.2f},v_rel={:.2f},{:.2f},{:.2f},v_est={:.2f},{:.2f},{:.2f},cost={:.4f},res={:.4f},id={:.2f}".format(c_id, vx,vy,vz, vest_x,vest_y,vest_z, cost, res, id) for c_id, vx,vy,vz, vest_x,vest_y,vest_z, cost, res, id in np.hstack((radar0_cluster_label.reshape(-1,1), radar0[:,3:6], radar0_flow_est, radar0_vel_cost.reshape(-1,1), radar0_vel_residual.reshape(-1,1), radar0_id.reshape(-1,1))).round(2)]
                radar_flow_est_color = flow_to_rgb(radar0_flow_est) / 255.0
                labels = ["flow_est={:.2f},{:.2f},{:.2f}".format(vx,vy,vz) for vx,vy,vz in radar0_flow_est.round(2)]
                rr.log(f"world/ego_vehicle/radar/flow_est", rr.Points3D(radar0_perturbed[:,:3], colors=radar_flow_est_color, radii=np.ones((radar0.shape[0],))*point_size, labels=labels))

                # # log pc0 with dynamic mask
                # pc_dynamic_color = np.tile(pcd_color, (pc0.shape[0], 1))
                # pc_dynamic_color[pc0_dynamic_mask] = color_map[1]
                # rr.log(f"world/ego_vehicle/lidar/dynamic", rr.Points3D(pc0[:,:3], colors=pc_dynamic_color[:,:], radii=np.ones((pc0[:,:].shape[0],))*point_size/4))
                # # log pc0 with labels
                # pc_cluster_color = np.tile(pcd_color, (pc0.shape[0], 1))    
                # for label_i in np.unique(pc0_cluster_label):
                #     if label_i > 0:
                #         pc_cluster_color[pc0_cluster_label == label_i] = color_map[label_i % len(color_map)]
                # # labels = ["cluster={:}, intensity={:.4f}".format(c, intensity) for c, intensity in np.hstack((pc0_cluster_label.reshape(-1,1), pc0[:,3].reshape(-1,1)))]
                # rr.log(f"world/ego_vehicle/lidar/labels", rr.Points3D(pc0[:,:3], colors=pc_cluster_color[:,:], radii=np.ones((pc0[:,:].shape[0],))*point_size/4)) #, labels=labels))
                
                # log estimated lidar velocity 
                labels = ["flow_est={:.2f},{:.2f},{:.2f}".format(vx,vy,vz) for vx,vy,vz in pc0_flow_est.round(2)]
                # compute point radii according to pc0_range: point_size/4 if range near 0, point_size for range far
                pc0_radii = (point_size / 8 + (point_size * (pc0_range / pc0_range.max()))).cpu().numpy()
                rr.log(f"world/ego_vehicle/lidar/flow_est", rr.Points3D(pc0[:,:3], colors=flow_to_rgb(pc0_flow_est) / 255.0, radii=pc0_radii, labels=labels)) # radii=np.ones((pc0.shape[0],))*point_size/2

                # # log ground truth flow
                # flow = batch['flow'][batch_id][~gm0] - pose_flow # ego motion compensation here.
                # flow = flow.cpu().numpy()
                # labels = ["flow_gt={:.2f},{:.2f},{:.2f}".format(vx,vy,vz) for vx,vy,vz in flow.round(2)]
                # rr.log(f"world/ego_vehicle/lidar/flow", rr.Points3D(pc0[:,:3], colors=flow_to_rgb(flow) / 255.0, radii=pc0_radii, labels=labels))

                # # log noisy points
                # pc0_noisy = pc0_cluster_label == 0
                # pc0_noisy_color = np.tile(pcd_color, (pc0_noisy.shape[0], 1))
                # rr.log(f"world/ego_vehicle/lidar/noisy", rr.Points3D(pc0[pc0_noisy,:3], colors=pc0_noisy_color, radii=np.ones((pc0[pc0_noisy,:3].shape[0],))*point_size/4))

                # log raw lidar points
                rr.log(f"world/ego_vehicle/lidar/raw", rr.Points3D(pc0[:,:3], colors=np.tile(pcd_color, (pc0.shape[0], 1)), radii=np.ones((pc0[:,:3].shape[0],))*point_size/4))

                # if 'cam0' in batch:
                #     for r_idx in range(cam0_to_refego_tf.shape[0]):
                #         img = cv2.cvtColor(batch['cam0'][batch_id][r_idx].cpu().numpy(), cv2.COLOR_BGR2RGB)
                #         rr.log(f"world/ego_vehicle/cam_frame_{r_idx}", rr.Image(img))
                        
        res_dict = {"flow": batch_final_flow,
                    "pose_flow": pose_flows,
                    "pc_dynamic_mask": pc_dynamic_mask_list,
                    "radar0_flow": radar0_flow_est_list,
                    }
        
        return res_dict