"""
# 
# Created: 2024-02-24 10:48
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Ajinkya Khoche  (https://ajinkyakhoche.github.io/)
#
# Description: Preprocess Data, save as h5df format for faster loading
# This one is for MAN TruckScenes dataset
# 
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import multiprocessing
from pathlib import Path
from multiprocessing import Pool, current_process
from typing import Optional, Tuple, Dict, Union, Final
from tqdm import tqdm
import numpy as np
import fire, time, h5py

from truckscenes import TruckScenes
from truckscenes.utils import splits
from truckscenes.utils.geometry_utils import transform_matrix
from truckscenes.utils.data_classes import Box
from truckscenes.utils.geometry_utils import points_in_box
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion

import os, sys
PARENT_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..'))
sys.path.append(PARENT_DIR)
from dataprocess.misc_data import create_reading_index, create_meta_index, find_closest_integer_in_ref_arr
from src.models.basic import cal_pose0to1_np
from src.utils.av2_eval import CATEGORY_TO_INDEX, ManNamMap
from linefit import ground_seg
import cv2

BOUNDING_BOX_EXPANSION: Final = 0.2
GROUNDSEG_config = f"{PARENT_DIR}/conf/others/truckscenes.toml"
ref_chan = 'LIDAR_LEFT'
LIDAR_FREQUENCY = 10
RADAR_FREQUENCY = 20
EPSILON = 1e-6

def remove_ego_points(pc: np.ndarray,
                    length_threshold: float = 2.0, 
                    width_threshold: float = 7.0) -> np.ndarray:
    """
    Remove ego points from a point cloud.
    :param pc: point cloud to remove ego points from, shape: (N, 4)
    :param length_threshold: Length threshold.
    :param width_threshold: Width threshold.
    :return: point cloud without ego points, shape: (N, 4).
    """
    # NuScenes LiDAR position
    # x: left --> width_threshold; y: front --> length_threshold; z: up --> height_threshold
    x_filt = np.logical_and(pc[:, 0] > -width_threshold/2, pc[:, 0] < width_threshold)
    y_filt = np.logical_and(pc[:, 1] > -length_threshold, pc[:, 1] < length_threshold)
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    not_close = not_close.astype(bool)
    return pc[not_close]
        
def get_pose(nusc, sweep_data):
    # pose from lidar to world
    ref_to_ego = nusc.get('calibrated_sensor', sweep_data['calibrated_sensor_token'])
    ego_to_world = nusc.get('ego_pose', sweep_data['ego_pose_token'])
    ref_to_ego_np = transform_matrix(ref_to_ego['translation'], Quaternion(ref_to_ego['rotation']))
    ego_to_world_np = transform_matrix(ego_to_world['translation'], Quaternion(ego_to_world['rotation']))
    return np.dot(ref_to_ego_np, ego_to_world_np)
    

def process_log(nusc_mode, data_dir: Path, scene_num_id: int, scene_id: str, output_dir: Path, resample2frequency=10, n: Optional[int] = None) :

    def create_group_data(group, pc, radar, radar_flow, radar_flow_raw, radar_id, radar_rcs, radar_to_refego_tf, 
                          pose, gm = None, flow_0to1=None, flow_valid=None, flow_category=None, ego_motion=None,
                          velocity_gt=None,
                          ):
        group.create_dataset('lidar', data=pc.astype(np.float32))
        group.create_dataset('radar', data=radar.astype(np.float32))
        # group.create_dataset('radar_velocity', data=radar_velocity.astype(np.float32))
        group.create_dataset('radar_flow', data=radar_flow.astype(np.float32))
        group.create_dataset('radar_flow_raw', data=radar_flow_raw.astype(np.float32))
        
        # group.create_dataset('radar_azimuth', data=radar_azimuth.astype(np.float32))
        group.create_dataset('radar_id', data=radar_id.astype(np.uint8))
        group.create_dataset('radar_rcs', data=radar_rcs.astype(np.float32))
        group.create_dataset('radar_to_refego_tf', data=radar_to_refego_tf.astype(np.float32))
        
        group.create_dataset('pose', data=pose.astype(np.float64))
        if ego_motion is not None:
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
        if gm is not None:
            group.create_dataset('ground_mask', data=gm.astype(bool))
        if flow_0to1 is not None:
            # ground truth flow information
            group.create_dataset('flow', data=flow_0to1.astype(np.float32))
            group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
            group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
            group.create_dataset('velocity_gt', data=velocity_gt.astype(np.float32))

    def compute_flow_simple(pc0, pose0, pose1, ts0, ts1, sample_ann_list):
        # compute delta transform between pose0 and pose1
        ego1_SE3_ego0 = cal_pose0to1_np(pose0, pose1)
        # flow due to ego motion
        flow = np.zeros_like(pc0[:,:3])
        pose_flow = pc0[:,:3] @ ego1_SE3_ego0[:3,:3].T + ego1_SE3_ego0[:3,3] - pc0[:,:3] # pose flow
        flow = pose_flow
        # valid mask
        valid = np.ones(len(pc0), dtype=np.bool_)
        # category mask
        classes = np.zeros(len(pc0), dtype=np.uint8)
        # velocity 
        velocity = np.zeros((len(pc0), 3))
        points_in_box_mask_all = np.zeros((pc0.shape[0],), dtype=bool)
        delta_t = (ts1 - ts0) * 1e-6
        for ann in sample_ann_list:
            ann_vel = ann.velocity
            if not np.isnan(ann.velocity).any():
                pt_cls = ann.name
                # compute points_in_box mask, expansion factor acc to velocity.
                points_in_box_mask = points_in_box(ann, pc0[:,:3].T, wlh_factor=1.2)
                # points_in_box_mask = points_in_box(ann, pc0[:,:3].T, wlh_custom=(ann_vel * 0.1).tolist())
                points_in_box_mask_all = np.logical_or(points_in_box_mask_all, points_in_box_mask)
                classes[points_in_box_mask] = CATEGORY_TO_INDEX[ManNamMap[pt_cls]]
                if np.sum(points_in_box_mask) > 5:
                    velocity[points_in_box_mask] = ann_vel
                else:
                    valid[points_in_box_mask] = False
        obj_flow = velocity * delta_t
        flow += obj_flow
        return {#'pcl_0': sweeps[0].xyz, 'pcl_1' :sweeps[1].xyz, 
                'flow_0_1': flow,
                'valid_0': valid, 'classes_0': classes, 
                #'pose_0': poses[0], 'pose_1': poses[1],
                'ego_motion': ego1_SE3_ego0,
                'velocity_gt': velocity
                }
    
    def radar_flow_motion_compensation(radar0_flow, radar0_id, radar0_to_refego_tf_arr, ego_motion):
        # in truckscenes, the radar flow needs to be compensated by ego motion
        radar0_flow_mc = np.zeros_like(radar0_flow)
        for r_idx in np.unique(radar0_id).astype(int):
            radar_mask = radar0_id == r_idx
            # step: transform ego flow to radar frame
            ego_flow_radar_frame = (np.linalg.inv(radar0_to_refego_tf_arr[r_idx][:3,:3]) @ ego_motion[:3,3][None,:].T).T[:,:3]
            # compute component of estimated flow along radar beam  
            ego_flow_radar_frame_beam_component = np.matmul(radar0_flow_unit[radar_mask], ego_flow_radar_frame.T) * radar0_flow_unit[radar_mask]
            # ego_flow_radar_frame_beam_component = np.matmul(radar0_unit[radar_mask], ego_flow_radar_frame.T) * radar0_unit[radar_mask]
            radar0_flow_mc[radar_mask] = radar0_flow[radar_mask] - ego_flow_radar_frame_beam_component
        return radar0_flow_mc
    
    log_id = scene_id
    
    if os.path.exists(output_dir/f'{log_id}.h5'):
        print(f'{log_id}.h5 already exists')
        return

    nusc = TruckScenes(dataroot=data_dir, version=nusc_mode, verbose=False)
    scene = nusc.scene[scene_num_id]
    mygroundseg = ground_seg(GROUNDSEG_config)

    # step note down timestamps for sweeps (interpolation points) and samples (at which data for interpolation exists)
    now_sample_token_str = scene['first_sample_token']
    sample = nusc.get('sample', now_sample_token_str)
    
    sweep_data_dict = {}
    skipFrame = int(LIDAR_FREQUENCY / resample2frequency) # since truckscenes sweep at 10Hz, we want to resample to 10Hz,
    for channel, token in sample['data'].items():
        ## Ajinkya: some camera tokens give error
        #if 'RADAR' in channel or 'LIDAR' in channel:
        cnt = 0
        sweep_data_lst = []
        sample_data = nusc.get('sample_data', token)
        while sample_data['next'] != '':
            if cnt % skipFrame == 0:
                sweep_data_lst.append(sample_data)
            sample_data = nusc.get('sample_data', sample_data['next'])
            cnt += 1
        sweep_data_dict[channel] = sweep_data_lst
    
    # to handle the last frame
    prev_ego_motion = np.eye(4)

    with h5py.File(output_dir/f'{log_id}.h5', 'a') as f:
        for cnt, sweep_data in enumerate(sweep_data_dict[ref_chan]):
            gt_flow_flag = False
            # timestamp for reference channel
            ts0 = sweep_data['timestamp']
            # get pose0
            ref_pose_record = nusc.get('ego_pose', sweep_data['ego_pose_token'])
            ref_ego2global = transform_matrix(ref_pose_record['translation'], Quaternion(ref_pose_record['rotation']))
            
            # get sensor2ego
            ref_cs_record = nusc.get('calibrated_sensor', sweep_data['calibrated_sensor_token'])
            ref_sensor2ego = transform_matrix(ref_cs_record['translation'], Quaternion(ref_cs_record['rotation']))
    
            lidar_list = []
            lidar_timestamp_list = []
            lidar_id_list = []
            radar_list = []
            radar_delta_t_list = []
            radar_id_list = []
            radar_azimuth_list = []
            radar_id = 0; lidar_id = 0
            radar_to_refego_tf_list = []

            for channel in sweep_data_dict.keys():
                # load closest sensor point cloud in reference ego frame
                sensor_sweep_list = sweep_data_dict[channel]
                closest_ch_ind, closest_ch_timestamp, timestamp_diff = find_closest_integer_in_ref_arr(
                    ts0, np.array([t['timestamp'] for t in sensor_sweep_list])
                )
                sensor_sweep = sensor_sweep_list[closest_ch_ind]
                curr_cs_record = nusc.get('calibrated_sensor',
                        sensor_sweep['calibrated_sensor_token'])
                curr_pose_record = nusc.get('ego_pose', sensor_sweep['ego_pose_token'])

                curr_sensor2ego = transform_matrix(curr_cs_record['translation'], Quaternion(curr_cs_record['rotation']))
                curr_ego2global = transform_matrix(curr_pose_record['translation'], Quaternion(curr_pose_record['rotation']))
                
                if 'RADAR' in channel:
                    # FIELDS x y z vrel_x vrel_y vrel_z rcs 
                    pc = RadarPointCloud.from_file(os.path.join(str(data_dir), sensor_sweep['filename'])).points.T
                    # compute azimuth for radar points
                    azimuth = np.arctan2(pc[:,1], pc[:,0])
                    radar_azimuth_list.append(azimuth)

                    # transform from sensor frame to reference ego frame
                    temp = np.hstack((pc[:,:3], np.ones((pc.shape[0],1))))
                    radar_to_refego_tf = np.linalg.inv(ref_ego2global) @ curr_ego2global @ curr_sensor2ego
                    radar_to_refego_tf_list.append(radar_to_refego_tf)
                    pc[:,:3] = (radar_to_refego_tf @ temp.T).T[:,:3]
                    # add sensor_id and delta_t as additional attributes
                    delta_t = np.ones((pc.shape[0])) * timestamp_diff * 1e-6
                    # pc = np.hstack((pc, np.ones((pc.shape[0],1))*radar_id, delta_t))
                    radar_list.append(pc)
                    radar_delta_t_list.append(delta_t)
                    radar_id_list.append(np.ones((pc.shape[0]))*radar_id)
                    radar_id += 1
                elif 'LIDAR' in channel:
                    point_cloud = LidarPointCloud.from_file(os.path.join(str(data_dir), sensor_sweep['filename'])) #.points.T
                    pc, timestamp = point_cloud.points.T, point_cloud.timestamps.T
                    # transform from sensor frame to reference ego frame
                    temp = np.hstack((pc[:,:3], np.ones((pc.shape[0],1))))
                    pc[:,:3] = (np.linalg.inv(ref_ego2global) @ curr_ego2global @ curr_sensor2ego @ temp.T).T[:,:3]
                    # add lidar_id and delta_t as additional attributes
                    delta_t = np.ones((pc.shape[0],1)) * timestamp_diff * 1e-6
                    pc = np.hstack((pc, np.ones((pc.shape[0],1))*lidar_id, delta_t))
                    lidar_list.append(pc)
                    lidar_timestamp_list.append(timestamp.squeeze())
                    lidar_id_list.append(np.ones((pc.shape[0]))*lidar_id)
                    lidar_id += 1

            pc0 = np.vstack(lidar_list)
            pc0_timestamp = np.hstack(lidar_timestamp_list).astype(np.int64)
            radar0 = np.vstack(radar_list)
            radar0_id = np.hstack(radar_id_list)
            radar0_delta_t = np.hstack(radar_delta_t_list)
            radar0_to_refego_tf_arr = np.stack(radar_to_refego_tf_list)
            radar0_azimuth = np.hstack(radar_azimuth_list)
            radar0_velocity = radar0[:,3:6]
            radar0_rcs = radar0[:,6]
            radar0_flow = radar0_velocity * 1/resample2frequency
            # radar0_flow_norm = np.linalg.norm(radar0_flow, axis=1); radar0_flow_norm[radar0_flow_norm == 0] = 1
            # radar0_flow_unit = radar0_flow / radar0_flow_norm[:,None]
            radar0_flow_unit = radar0_flow / (np.linalg.norm(radar0_flow, axis=1)[:,None] + EPSILON)
            radar0_unit = radar0[:,:3] / (np.linalg.norm(radar0[:,:3], axis=1)[:,None] + EPSILON)

            # ground segmentation
            pc0 = remove_ego_points(pc0)
            is_ground_0 = np.array(mygroundseg.run(pc0[:, :3]))

            if cnt == len(sweep_data_dict[ref_chan]) - 1:
                ego_motion = prev_ego_motion
                radar0_flow_mc = radar_flow_motion_compensation(radar0_flow, radar0_id, radar0_to_refego_tf_arr, ego_motion)
                # radar0_flow_mc = radar0_flow

                group = f.create_group(str(ts0))
                create_group_data(group=group, pc=pc0, radar=radar0, radar_flow=radar0_flow_mc, radar_flow_raw=radar0_flow, radar_id=radar0_id, radar_rcs=radar0_rcs,
                                  radar_to_refego_tf=radar0_to_refego_tf_arr, gm=is_ground_0.astype(np.bool_), pose=ref_ego2global)
            else:
                sweep_data_next = sweep_data_dict[ref_chan][cnt+1] 
                ts1 = sweep_data_next['timestamp']
                # get pose0
                ref_pose_record_next = nusc.get('ego_pose', sweep_data_next['ego_pose_token'])
                ref_ego2global_next = transform_matrix(ref_pose_record_next['translation'], Quaternion(ref_pose_record_next['rotation']))
                
                ego_motion = cal_pose0to1_np(ref_ego2global, ref_ego2global_next)
                prev_ego_motion = ego_motion
                radar0_flow_mc = radar_flow_motion_compensation(radar0_flow, radar0_id, radar0_to_refego_tf_arr, ego_motion)
                # radar0_flow_mc = radar0_flow
            
                # load annotations if exists
                sd_record = nusc.get('sample_data', sweep_data['token'])
                curr_sweep_record = nusc.get('sample', sd_record['sample_token'])

                if sweep_data['prev'] == "" or sd_record['is_key_frame']:
                    sweep_ann_list = nusc.get_boxes(sweep_data['token'])
                    gt_flow_flag = True
                    # transform boxes to reference ego frame
                    cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
                    s_record = nusc.get('sample', sd_record['sample_token'])
                    s_pose_record = nusc.getclosest('ego_pose', s_record['timestamp'])
                    for box in sweep_ann_list:
                        # Move box to ego vehicle coord system: in-place operation
                        box.translate(-np.array(s_pose_record['translation']))
                        box.rotate(Quaternion(s_pose_record['rotation']).inverse)

                group = f.create_group(str(ts0))
                if not gt_flow_flag: # no annotations, only save data
                    create_group_data(group=group, pc=pc0, radar=radar0, radar_flow=radar0_flow_mc, radar_flow_raw=radar0_flow, radar_id=radar0_id, radar_rcs=radar0_rcs,
                                      radar_to_refego_tf=radar0_to_refego_tf_arr, gm=is_ground_0.astype(np.bool_), pose=ref_ego2global, \
                                        ego_motion=ego_motion)
                else: # compute sceneflow
                    scene_flow = compute_flow_simple(pc0, ref_ego2global, ref_ego2global_next, ts0, ts1, sweep_ann_list)
                    create_group_data(group=group, pc=pc0, radar=radar0, radar_flow=radar0_flow_mc, radar_flow_raw=radar0_flow, radar_id=radar0_id, radar_rcs=radar0_rcs,
                                      radar_to_refego_tf=radar0_to_refego_tf_arr, gm=is_ground_0.astype(np.bool_), pose=ref_ego2global,
                                      flow_0to1=scene_flow['flow_0_1'], flow_valid=scene_flow['valid_0'], flow_category=scene_flow['classes_0'],
                                      ego_motion=scene_flow['ego_motion'].astype(np.float32), velocity_gt=scene_flow['velocity_gt'].astype(np.float32))



def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)
    
def process_logs(nusc, mode, data_dir: Path, scene_list: list, output_dir: Path, nproc: int, resample2frequency: int):
    """Compute sceneflow for all logs in the dataset. Logs are processed in parallel.
       Args:
         data_dir: Argoverse 2.0 directory
         output_dir: Output directory.
    """
    
    if not (data_dir).exists():
        print(f'{data_dir} not found')
        return

    args = sorted([(mode, data_dir, scene_num_id, scene_id, output_dir, resample2frequency) for scene_num_id, scene_id in enumerate(scene_list)])
    print(f'Using {nproc} processes')
    
    # for debug
    proc(args[0], ignore_current_process=True)
    # for x in tqdm(args):
    #     # scene = nusc.scene[x[2]]['name']
    #     # if not os.path.exists(output_dir/f'{scene}.h5'):
    #     proc(x, ignore_current_process=True)
    #         # break

    ## for parallel processing
    # if nproc <= 1:
    #     for x in tqdm(args):
    #         proc(x, ignore_current_process=True)
    # else:
    #     with Pool(processes=nproc) as p:
    #         res = list(tqdm(p.imap_unordered(proc, args), total=len(scene_list), ncols=100))

def main(
    data_dir: str = "/home/ajinkya/datasets/truckscenes/man-truckscenes/",
    mode: str = "v1.0-mini",
    output_dir: str ="/home/ajinkya/datasets/truckscenes/man-truckscenes/preprocess",
    nproc: int = 1,  #(multiprocessing.cpu_count() - 1),
    resample2frequency: int = 10,
    create_index_only: bool = False,
    split_name = None
):
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini'] # defined by nus.
    assert mode in available_vers
    nusc = TruckScenes(dataroot=data_dir, version=mode, verbose=True)

    # get records with description tags
    recs = [
            (nusc.get('sample', record['first_sample_token'])['timestamp'], record)
            for record in nusc.scene
        ]
    # create dictionary of scene names and their descriptions
    description_dict = {l[1]['name']: l[1]['description'] for l in recs if 'description' in l[1]}

    if mode == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
        if split_name is not None and split_name == 'train':
            input_dict = {'train': train_scenes}
        elif split_name is not None and split_name == 'val':
            input_dict = {'val': val_scenes}
        else:
            input_dict = {'train': train_scenes, 'val': val_scenes}
    elif mode == 'v1.0-test':
        test_scenes = splits.test
        input_dict = {'test': test_scenes}
    elif mode == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
        # input_dict = {
        #     'mini_train': train_scenes,
        #     'mini_val': val_scenes
        # }
        # NOTE(Qingwen): or if you don't want to split mini, use below
        input_dict = {'mini': train_scenes + val_scenes}
    else:
        raise ValueError('unknown')

    for input_key, input_val in input_dict.items():
        output_dir_ = Path(output_dir) / input_key
        if not create_index_only:
            output_dir_.mkdir(exist_ok=True, parents=True)
            process_logs(nusc, mode, Path(data_dir), input_val, output_dir_, nproc, resample2frequency)
        
        create_reading_index(output_dir_)
        create_meta_index(description_dict, input_val, Path(output_dir_))

if __name__ == '__main__':
    start_time = time.time()
    fire.Fire(main)
    print(f"\nTime used: {(time.time() - start_time)/60:.2f} mins")