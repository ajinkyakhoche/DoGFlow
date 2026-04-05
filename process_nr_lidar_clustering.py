"""
# Created: 2024-11-24 21:30
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche  (https://ajinkyakhoche.github.io/)
#
# This file is part of DoGFlow (https://ajinkyakhoche.github.io/DogFlow/)
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.
# 
# Description: run noise resistant clustering on LiDAR point clouds using HDBSCAN, 
#              and save the cluster labels for later use in DoGFlow 
#              evaluation/pseudo labeling.
"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import fire, time, h5py, os
from hdbscan import HDBSCAN
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from typing import Optional

from src.utils.mics import HDF5Data
from assets.cuda.chamfer3D import ChamferDis, nnChamferDis
MyCUDAChamferDis = nnChamferDis()

import argparse
import rerun as rr
import rerun.blueprint as rrb
import torch
from src.utils.o3d_view import color_map
import matplotlib
import dztimer
from multiprocessing import Pool, current_process
from scipy.optimize import lsq_linear
from scipy.sparse.csgraph import connected_components


AZIMUTH_ERROR = 0.5 # radar sensing error in azimuth in degrees
MIN_ERROR = 0.1 # minimum sensing error in meters
MAX_ERROR = 3.0
DYNAMIC_CLASSIFICATION_THRESHOLD = 0.25  # meters
INTENSITY_THRESHOLD = 8e-3


def prepare_input(data_dir, scene_range, overwrite, vis_interval, tone, timer, nproc):
    if nproc > 1:
        vis_interval = 0
    data_path = Path(data_dir)
    dataset = HDF5Data(data_path)
    all_scene_ids = list(dataset.scene_id_bounds.keys())

    # Filter all_scene_ids based on scene_range
    if scene_range[0] != -1 and scene_range[-1] != -1:
        all_scene_ids = [scene_id for scene_in_data_index, scene_id in enumerate(all_scene_ids) 
                        if scene_range[0] <= scene_in_data_index < scene_range[1]]

    args = sorted([(data_path, scene_id, overwrite, vis_interval, tone, timer) for scene_id in all_scene_ids])
    return args

def log_vehicle_frames(radar0_to_refego_tf):
    # log ego vehicle frame
    rr.log(
        f"world/ego_vehicle",
        rr.Transform3D(
            translation=np.zeros((3,)),
            rotation=rr.Quaternion(xyzw=np.array([0,0,0,1])), #rr.Quaternion(xyzw=rotation_xyzw),
            from_parent=False,
        ),
        static=True,
    )
    
    # log radar frames
    for r_idx in range(radar0_to_refego_tf.shape[0]):
        rr.log(
            f"world/ego_vehicle/radar_frame_{r_idx}",
            rr.Transform3D(
                translation=radar0_to_refego_tf[r_idx,:3,3],
                mat3x3=rr.datatypes.Mat3x3(radar0_to_refego_tf[r_idx,:3,:3]),
                from_parent=False,
                scale=0.1,
            ),
            static=True,
        )


def run_lidar_cluster(data_dir, scene_range, overwrite, vis_interval, tone, timer=None, nproc=1):
    args = prepare_input(data_dir, scene_range, overwrite, vis_interval, tone, timer, nproc)
    print(f'HDBSCAN LiDAR Clustering Using {nproc} processes')
    if nproc <= 1:
        for x in tqdm(args):
            proc_lidar_clustering(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc_lidar_clustering, args), total=len(args), ncols=100))

def proc_lidar_clustering(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    run_lidar_clustering_thread(*x, n=pos)

def run_lidar_clustering_thread(data_path, scene_id, overwrite, vis_interval, tone, timer, n: Optional[int] = None):
    if vis_interval:
        point_size = 0.4
        if tone == 'light':
            pcd_color = [0.5, 0.5, 0.5]
            ground_color = [0.75, 0.75, 0.75]
            noise_color = [0, 0, 0]
        elif tone == 'dark':
            pcd_color = [1., 1., 1.]
            ground_color = [0.5, 0.5, 0.5]
            noise_color = [0, 0, 0]

    dataset = HDF5Data(data_path, flow_view=True)
    bounds = dataset.scene_id_bounds[scene_id]
    flag_exist_label = True
    with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
        for ii in range(bounds["min_index"], bounds["max_index"]+1):
            key = str(dataset[ii]['timestamp'])
            if 'label' not in f[key]:
                flag_exist_label = False
                break
    if flag_exist_label and not overwrite:
        print(f"==> Scene {scene_id} has plus label, skip.")
        return
    
    hdb = HDBSCAN(min_cluster_size=5, cluster_selection_epsilon=0.9, core_dist_n_jobs=1)
    for i in range(bounds["min_index"], bounds["max_index"]+1):
        if i == bounds["max_index"]:
            data = dataset.__getitem__(i, force=True)
        else:
            data = dataset[i]
        pc0 = data['pc0']
        gm0 = data['gm0']
        pc0_range = np.linalg.norm(pc0[:,:3], axis=1)
        pc0_full = pc0.copy()

        radar0_to_refego_tf = data['radar0_to_refego_tf']
        radar0 = data['radar0']

        timer[1].start("Lidar NN association")
        # use chamferdist to find nearest radar point for each lidar point
        dist_lidar, dist_radar, idx_lidar, idx_radar = ChamferDis.apply(torch.tensor(pc0[~gm0,:3]).cuda().float().contiguous(), torch.tensor(radar0[:,:3]).cuda().float().contiguous())
        dist_lidar = dist_lidar.cpu().numpy()
        idx_lidar = idx_lidar.cpu().numpy()
        # if the distance is greater than a threshold, then no associated radar point
        association_threshold = MIN_ERROR + (MAX_ERROR - MIN_ERROR) * pc0_range[~gm0] / pc0_range[~gm0].max()
        idx_lidar[dist_lidar>association_threshold] = -1
        timer[1].stop()

        timer[2].start("noise thresholding")
        # point which have  intensity lower than threshold and have no associated radar point will be considered as noise
        idx_lidar_full = np.full(pc0.shape[0], -1)
        idx_lidar_full[~gm0] = idx_lidar
        noise0 = (pc0[:,3] < INTENSITY_THRESHOLD) & (idx_lidar_full == -1) & ~gm0
        
        # # point which have  intensity lower than threshold will be considered as noise
        # noise0 = (pc0[:,3] < INTENSITY_THRESHOLD) & ~gm0
        
        # use chamferdist to find nearest clean pc for each noisy pc
        dist_clean, dist_noise, idx_clean, idx_noise = ChamferDis.apply(torch.tensor(pc0[~np.logical_or(gm0, noise0),:3]).cuda().float().contiguous(), torch.tensor(pc0[noise0,:3]).cuda().float().contiguous())
        dist_noise = dist_noise.cpu().numpy()
        idx_noise = idx_noise.cpu().numpy()
        idx_noise[dist_noise>0.5] = -1
        timer[2].stop()

        # IMPORTANT: remove ground and noise points from pc0 before clustering, otherwise they will be assigned to clusters
        pc0 = pc0[~np.logical_or(gm0, noise0),:]

        timer[3].start("HDBSCAN Clustering")
        # cluster points
        pc0_cluster_label = np.zeros(pc0.shape[0], dtype= np.int16)
        hdb.fit(pc0[:,:4])
        # NOTE(Qingwen): since -1 will be assigned if no cluster. We set it to 0.
        pc0_cluster_label = hdb.labels_ + 1

        
        # assign cluster labels to noise points according to their nearest clean point
        noise0_cluster_label = np.zeros_like(idx_noise)
        noise0_cluster_label[idx_noise!=-1] = pc0_cluster_label[idx_noise[idx_noise!=-1]]
        
        pc0_cluster_label_full = np.zeros(pc0_full.shape[0], dtype= np.int16)
        pc0_cluster_label_full[~np.logical_or(gm0, noise0)] = pc0_cluster_label
        pc0_cluster_label_full[noise0] = noise0_cluster_label
        timer[3].stop()

        if vis_interval>0 and (i+1)%vis_interval == 0:
            rr.set_time_sequence('frame_idx', i)
            # rr.set_time_seconds("timestamp", int(data["timestamp"]) * 1e-6)

            # log ego vehicle and radar frame
            log_vehicle_frames(radar0_to_refego_tf)
            
            pc_ground_color = np.tile(ground_color, (pc0_full[gm0].shape[0], 1))
            pc_noise_color = np.tile(noise_color, (pc0_full[noise0].shape[0], 1))
            # log ground points
            labels = ["intensity={:.4f}".format(intensity) for intensity in pc0_full[gm0,3]]
            rr.log(f"world/ego_vehicle/lidar/ground", rr.Points3D(pc0_full[gm0,:3], colors=pc_ground_color, radii=np.ones((pc0_full[gm0,:].shape[0],))*point_size/4, labels=labels))
            
            # log noise points
            labels = ["intensity={:.4f}".format(intensity) for intensity in pc0_full[noise0,3]]
            rr.log(f"world/ego_vehicle/lidar/noise", rr.Points3D(pc0_full[noise0,:3], colors=pc_noise_color, radii=np.ones((pc0_full[noise0,:].shape[0],))*point_size/4, labels=labels))
            
            # log pc0 with labels
            pc_cluster_color = np.tile(pcd_color, (pc0_full.shape[0], 1))
            for label_i in np.unique(pc0_cluster_label_full):
                if label_i > 0:
                    pc_cluster_color[pc0_cluster_label_full == label_i] = color_map[label_i % len(color_map)]
            labels = ["cluster={:}, intensity={:.4f}".format(c, intensity) for c, intensity in np.hstack((pc0_cluster_label_full.reshape(-1,1), pc0_full[:,3].reshape(-1,1)))]
            rr.log(f"world/ego_vehicle/lidar/labels", rr.Points3D(pc0_full[~gm0,:3], colors=pc_cluster_color[~gm0,:], radii=np.ones((pc0_full[~gm0,:].shape[0],))*point_size/4, labels=labels))
            
        # save labels
        timestamp = data['timestamp']
        key = str(timestamp)
        with h5py.File(os.path.join(data_path, f'{scene_id}.h5'), 'r+') as f:
            # if 'pc_associated_radar' in f[key]:
            #     del f[key]['pc_associated_radar']
            if 'pc_cluster_label' in f[key]:
                del f[key]['pc_cluster_label']
            # if 'pc_dynamic_mask' in f[key]:
            #     del f[key]['pc_dynamic_mask']
            # if 'noise' in f[key]:
            #     del f[key]['noise']
            # f[key].create_dataset('pc_associated_radar', data=idx_lidar_full.astype(np.int16))
            f[key].create_dataset('pc_cluster_label', data=pc0_cluster_label_full.astype(np.int16))
            # f[key].create_dataset('pc_dynamic_mask', data=pc0_dynamic_mask_full.astype(np.uint8))
            # f[key].create_dataset('noise', data=noise0.astype(np.uint8))
    print(f"==> Scene {scene_id} finished, used: {(time.time() - start_time)/60:.2f} mins")



if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Visualizes flow dataset using the Rerun SDK.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ajinkya/datasets/truckscenes/man-truckscenes/preprocess/mini",
        help="data directory to preprocess",
    )
    parser.add_argument(
        "--scene_range",
        type=list,
        default=[-1],
        help="scenes to preprocess, set to [-1] to process all scenes, otherwise should be a list of [start_scene_index, end_scene_index]",
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=True,
        help="whether to overwrite the existing labels",
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=0,
        help="Optional: visualize every x steps, set 0 to disable visualization",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="light",
        help="tone of the visualization",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    if args.tone == 'light':
        background_color = (255, 255, 255)
    else:
        background_color = (80, 90, 110)

    if args.vis_interval:
        # setup the rerun environment
        blueprint = rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    name="3D",
                    origin="world",
                    # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
                    defaults=[rr.components.ImagePlaneDistance(4.0)],
                    background=background_color,
                    # Transform arrows for the vehicle shouldn't be too long.
                    overrides={"world/ego_vehicle": [rr.components.AxisLength(5.0)]},
                ),
                rrb.TextDocumentView(origin="description", name="Description"),
                column_shares=[3, 1],
            ),
        )

        rr.script_setup(args, "process_nr_lidar_clustering", default_blueprint=blueprint)


    timer = dztimer.Timing()
    timer.start("Preprocessing for DoGFlow: Noise-resistant LiDAR Clustering")

    run_lidar_cluster(
        data_dir=args.data_dir, 
        scene_range=args.scene_range, 
        overwrite=args.overwrite, 
        vis_interval=args.vis_interval, 
        tone=args.tone,
        timer=timer,
        )

    timer.print(random_colors=True, bold=True)
    
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")