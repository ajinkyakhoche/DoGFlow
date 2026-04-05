"""
# Created: 2024-11-20 22:30
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche  (https://ajinkyakhoche.github.io/)
#
# 
# Description: view scene flow dataset after preprocess or evaluation. 
# --flow_mode accepts a list eg. ["flow", "flow_est", ...] where "flow" is ground truth and "flow_est" is estimated from a neural network (result of save.py script).

# Usage with demo data: (flow is ground truth flow, `other_name` is the estimated flow from the model)
* python tools/rerun_visualization.py --data_dir /home/kin/data/av2/preprocess_v2/demo/sensor/val --flow_mode ['flow', 'deflow' , 'ssf'] 
"""

import numpy as np
import fire, time
from tqdm import tqdm

import open3d as o3d
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import HDF5Data, flow_to_rgb
from src.utils.o3d_view import color_map
from src.models.basic import cal_pose0to1_np
import rerun as rr
import rerun.blueprint as rrb
import argparse
import cv2

VIEW_FILE = f"{BASE_DIR}/assets/view/av2.json"
DESCRIPTION = """
Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
Author: [Ajinkya Khoche](https://ajinkyakhoche.github.io/)
Visualize scene flow dataset including lidar.

The code is modified from rerun example
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/nuscenes_dataset).
"""

import matplotlib
cmap_attr = matplotlib.colormaps["turbo"]
norm_speed = matplotlib.colors.Normalize(
    vmin=0.0,
    vmax=70.0,
)

norm_intensity = matplotlib.colors.Normalize(
    vmin=-30,
    vmax=30,
)

cmap_lidar_id = {
    0: (0.5, 0.0, 0.0, 1.0),  # Dark Red
    1: (0.0, 0.5, 0.0, 1.0),  # Dark Green
    2: (0.0, 0.0, 0.5, 1.0),  # Dark Blue
    3: (0.5, 0.5, 0.0, 1.0),  # Olive
    4: (0.5, 0.0, 0.5, 1.0),  # Purple
    5: (0.0, 0.5, 0.5, 1.0)   # Teal
}

def log_sensor_frames(radar0_to_refego_tf):
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


def vis_rerun(data_dir, flow_mode, start_id, vis_interval, point_size, tone, weather):
    dataset = HDF5Data(data_dir, vis_name=flow_mode, flow_view=True, weather=weather)

    if tone == 'light':
        pcd_color = [0.25, 0.25, 0.25]
        ground_color = [0.75, 0.75, 0.75]
    elif tone == 'dark':
        pcd_color = [1., 1., 1.]
        ground_color = [0.25, 0.25, 0.25]
    
    prev_ego_motion = np.eye(4)
    for data_id in (pbar := tqdm(range(start_id, len(dataset)))):
        if data_id % vis_interval != 0:
            continue
            
        data = dataset[data_id]
        rr.set_time_sequence('frame_idx', data_id)
        # rr.set_time_seconds("timestamp", int(data["timestamp"]) * 1e-6)
        # rr.set_time_nanos("timestamp", int(data["timestamp"])*1000)  
    
        now_scene_id = data['scene_id']
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")

        rr.log(f"world/ego_vehicle/scene_id", rr.TextDocument(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}"))
                
        pc0 = data['pc0']
        gm0 = data['gm0']
        pose0 = data['pose0']
        pose1 = data['pose1']

        pc0_range = np.linalg.norm(pc0[:, :3], axis=1)
        pc0_radii = point_size / 8 + (pc0_range / np.max(pc0_range)) * point_size*2

        if 'ego_motion' in data:
            ego_motion = data['ego_motion']
            prev_ego_motion = ego_motion
        else:
            # use the previous ego motion
            ego_motion = prev_ego_motion

        if 'velocity_gt' in data:
            velocity_gt = data['velocity_gt']

        # ego_pose = np.linalg.inv(pose1) @ pose0
        # ego_pose = cal_pose0to1(pose0, pose1)
        ego_pose = cal_pose0to1_np(pose0, pose1)
        
        pose_flow = pc0[:, :3] @ ego_pose[:3, :3].T + ego_pose[:3, 3] - pc0[:, :3]
        # pose_flow = pc0[:, :3] @ ego_motion[:3, :3].T + ego_motion[:3, 3] - pc0[:, :3]  

        # log ego pose
        rr.log(
            f"world/ego_vehicle",
            rr.Transform3D(
                translation=np.zeros((3,)),
                rotation=rr.Quaternion(xyzw=np.array([0,0,0,1])),
                from_parent=False,
            ),
            static=True,
        )
        
        log_sensor_frames(data['radar0_to_refego_tf'])


        for mode in flow_mode:
            flow_color = np.tile(pcd_color, (pc0.shape[0], 1))
            # flow_color[gm0] = ground_color

            if mode in ['dufo_label', 'label']:
                if mode in data:
                    labels = data[mode]
                    for label_i in np.unique(labels):
                        if label_i > 0:
                            flow_color[labels == label_i] = color_map[label_i % len(color_map)]

                # log flow mode
                rr.log(f"world/ego_vehicle/lidar/{mode}", rr.Points3D(pc0[~gm0,:3], colors=flow_color[~gm0,:], radii=pc0_radii[~gm0]))# radii=np.ones((pc0[~gm0].shape[0],))*point_size/2))

            elif mode in data:
                # print('ego_pose: ', ego_pose[:3, 3], ' ego_motion: ', ego_motion[:3, 3])
                # print(pose1[:3, 3] - pose0[:3, 3])
        
                flow = data[mode] - pose_flow # ego motion compensation here.
                flow_nanmask = np.isnan(data[mode]).any(axis=1)
                
                flow_color = np.tile(pcd_color, (pc0.shape[0], 1))

                flow_color[~flow_nanmask,:] = flow_to_rgb(flow[~flow_nanmask,:]) / 255.0
                
                # flow_color = np.array([cmap_lidar_id[l] for l in pc0[:,-2].tolist()])[:,:3]
                # flow_color[gm0] = ground_color
                
                # # log flow mode
                # labels = ["flow_gt={:.2f},{:.2f},{:.2f} vel_gt={:.2f},{:.2f},{:.2f}".format(fx,fy,fz, vx,vy,vz) for fx,fy,fz, vx,vy,vz in np.hstack((flow, velocity_gt)).round(2)]
                # labels = ["flow={:.2f},{:.2f},{:.2f}".format(fx,fy,fz) for fx,fy,fz in flow.round(2)]
                labels = ["flow={:.2f},{:.2f},{:.2f}, pose_flow={:.2f},{:.2f},{:.2f}".format(fx,fy,fz, pfx,pfy,pfz) for fx,fy,fz, pfx,pfy,pfz in np.hstack((flow[~gm0,:], pose_flow[~gm0,:])).round(2)]
                rr.log(f"world/ego_vehicle/lidar/{mode}", rr.Points3D(pc0[~gm0,:3], colors=flow_color[~gm0,:], radii=pc0_radii[~gm0], labels=labels))

        # log ground points
        rr.log(f"world/ego_vehicle/lidar/ground", rr.Points3D(pc0[gm0,:3], colors=ground_color, radii=np.ones((pc0[gm0,:3].shape[0],))*point_size/2))

        if 'pc0_dynamic_mask' in data:
            pc0_dynamic_mask = data['pc0_dynamic_mask']
            flow_color_dynamic = np.tile(pcd_color, (pc0.shape[0], 1)) #np.ones((pc0.shape[0],3))
            # flow_color_dynamic[gm0] = ground_color
            flow_color_dynamic[pc0_dynamic_mask] = color_map[1]
            rr.log(f"world/ego_vehicle/lidar/dynamic", rr.Points3D(pc0[~gm0,:3], colors=flow_color_dynamic[~gm0,:], radii=np.ones((pc0[~gm0,:].shape[0],))*point_size/4))
        
        if 'radar0' in data:
            radar0 = data['radar0']
            radar0_id = data['radar0_id']

            radii0_range = np.linalg.norm(radar0[:, :3], axis=1)
            radar0_radii = point_size + (radii0_range / np.max(radii0_range)) * point_size*2

            # for r_idx in np.unique(radar0_id):
            #     radar_mask = radar0_id == r_idx
            #     # radar_color_speed = cmap_attr(norm_speed(np.linalg.norm(radar0[radar_mask,3:6], axis=1)))
            #     radar_color_speed = flow_to_rgb(radar0[radar_mask,3:6]) / 255.0
            #     labels = ["v_raw={:.2f},{:.2f},{:.2f}".format(vx,vy,vz) for vx,vy,vz in radar0[radar_mask,3:6].round(2)]
            #     rr.log(f"world/ego_vehicle/radar_{r_idx}", rr.Points3D(radar0[radar_mask,:3], colors=radar_color_speed, radii=np.ones((radar0[radar_mask].shape[0],))*point_size, labels=labels))
            for r_idx in np.unique(radar0_id):
                radar_mask = radar0_id == r_idx
                # Filter for 60x60m area
                radar_points = radar0[radar_mask]
                xy_mask = (np.abs(radar_points[:, 0]) <= 30) & (np.abs(radar_points[:, 1]) <= 20)
                if np.any(xy_mask):
                    radar_color_speed = flow_to_rgb(radar_points[xy_mask, 3:6]) / 255.0
                    labels = ["v_raw={:.2f},{:.2f},{:.2f}".format(vx, vy, vz) for vx, vy, vz in radar_points[xy_mask, 3:6].round(2)]
                    rr.log(
                        f"world/ego_vehicle/radar_{r_idx}",
                        rr.Points3D(
                            radar_points[xy_mask, :3],
                            colors=radar_color_speed,
                            radii=np.ones((radar_points[xy_mask].shape[0],)) * point_size,
                            labels=labels
                        )
                    )
            
        if 'radar0_flow' in data:
            radar0_flow = data['radar0_flow']
            radar0_id = data['radar0_id']
            radar_color_speed = flow_to_rgb(radar0_flow[:,:3]) / 255.0
            labels = ["v_raw={:.2f},{:.2f},{:.2f}".format(vx,vy,vz) for vx,vy,vz in radar0_flow.round(2)]
            rr.log(f"world/ego_vehicle/radar0_flow", rr.Points3D(radar0[:,:3], colors=radar_color_speed, radii=radar0_radii, labels=labels))

        rr.log(f"world/ego_vehicle/lidar/raw", rr.Points3D(pc0[~gm0,:3], colors=pcd_color, radii=pc0_radii[~gm0]))

if __name__ == '__main__':
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Visualizes flow dataset using the Rerun SDK.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ajinkya/datasets/truckscenes/man-truckscenes/preprocess/mini_full",
        help="data directory to preprocess",
    )
    parser.add_argument(
        "--flow_mode",
        type=list,
        default=["flow", "dogflow"], #"label", "dufo_label", 
        help="flow modes to visualize",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="start id to visualize",
    )
    parser.add_argument(
        "--vis_interval",
        type=int,
        default=1,
        help="Optional: visualize every x steps",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=0.25,
        help="point size",
    )
    parser.add_argument(
        "--tone",
        type=str,
        default="light",
        help="tone of the visualization",
    )
    parser.add_argument(
        "--weather",
        type=str,
        default="snow",
        help="weather condition to filter the data",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    if args.tone == 'dark':
        background_color = (255, 255, 255)
    else:
        background_color = (80, 90, 110)

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

    rr.script_setup(args, "rerun_vis", default_blueprint=blueprint)

    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        timeless=True,
    )

    # call the main function
    vis_rerun(args.data_dir, args.flow_mode, args.start_id, args.vis_interval, args.point_size, args.tone, args.weather)
    print(f"Time used: {time.time() - start_time:.2f} s")
