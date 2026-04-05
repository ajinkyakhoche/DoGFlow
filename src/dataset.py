"""
# Created: 2023-11-04 15:52
# Updated: 2024-07-12 23:16
# 
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/), Jaeyeul Kim (jykim94@dgist.ac.kr)
#
# Change Logs:
# 2024-07-12: Merged num_frame based on Flow4D model from Jaeyeul Kim.
# 
# Description: Torch dataloader for the dataset we preprocessed.
# 
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py, os, pickle, argparse, sys
from tqdm import tqdm
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
import random

def collate_fn_pad(batch):

    num_frames = 2
    while f'pch{num_frames - 1}' in batch[0]:
        num_frames += 1

    # padding the data
    pc0_after_mask_ground, pc1_after_mask_ground= [], []
    pch_after_mask_ground = [[] for _ in range(num_frames - 2)]
    for i in range(len(batch)):
        pc0_after_mask_ground.append(batch[i]['pc0'][~batch[i]['gm0']])
        pc1_after_mask_ground.append(batch[i]['pc1'][~batch[i]['gm1']])
        for j in range(1, num_frames - 1):
            pch_after_mask_ground[j-1].append(batch[i][f'pch{j}'][~batch[i][f'gmh{j}']])

    pc0_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pc1_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_after_mask_ground, batch_first=True, padding_value=torch.nan)
    pch_after_mask_ground = [torch.nn.utils.rnn.pad_sequence(pch_, batch_first=True, padding_value=torch.nan) for pch_ in pch_after_mask_ground]


    res_dict =  {
        'pc0': pc0_after_mask_ground,
        'pc1': pc1_after_mask_ground,
        'pose0': [batch[i]['pose0'] for i in range(len(batch))],
        'pose1': [batch[i]['pose1'] for i in range(len(batch))]
    }

    for j in range(1, num_frames - 1):
        res_dict[f'pch{j}'] = pch_after_mask_ground[j-1]
        res_dict[f'poseh{j}'] = [batch[i][f'poseh{j}'] for i in range(len(batch))]

    if any(['flow' in batch[i] for i in range(len(batch))]):
        for i in range(len(batch)):
            if 'flow' not in batch[i]:
                batch[i]['flow'] = torch.full_like(batch[i]['pc0'], torch.nan)
                batch[i]['flow_is_valid'] = torch.zeros((batch[i]['pc0'].size(0)), dtype=torch.bool)
                batch[i]['flow_category_indices'] = torch.zeros((batch[i]['pc0'].size(0)), dtype=torch.uint8)
        flow = torch.nn.utils.rnn.pad_sequence([batch[i]['flow'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_is_valid = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_is_valid'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        flow_category_indices = torch.nn.utils.rnn.pad_sequence([batch[i]['flow_category_indices'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        res_dict['flow'] = flow
        res_dict['flow_is_valid'] = flow_is_valid
        res_dict['flow_category_indices'] = flow_category_indices

    if 'ego_motion' in batch[0]:
        res_dict['ego_motion'] = [batch[i]['ego_motion'] for i in range(len(batch))]
        
    if 'pc0_dynamic' in batch[0]:
        pc0_dynamic_after_mask_ground, pc1_dynamic_after_mask_ground= [], []
        for i in range(len(batch)):
            pc0_dynamic_after_mask_ground.append(batch[i]['pc0_dynamic'][~batch[i]['gm0']])
            pc1_dynamic_after_mask_ground.append(batch[i]['pc1_dynamic'][~batch[i]['gm1']])
        pc0_dynamic_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_dynamic_after_mask_ground, batch_first=True, padding_value=0)
        pc1_dynamic_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_dynamic_after_mask_ground, batch_first=True, padding_value=0)
        res_dict['pc0_dynamic'] = pc0_dynamic_after_mask_ground
        res_dict['pc1_dynamic'] = pc1_dynamic_after_mask_ground

    if 'pc0_labels' in batch[0]:
        pc0_labels_after_mask_ground, pc1_labels_after_mask_ground= [], []
        for i in range(len(batch)):
            pc0_labels_after_mask_ground.append(batch[i]['pc0_labels'][~batch[i]['gm0']])
            pc1_labels_after_mask_ground.append(batch[i]['pc1_labels'][~batch[i]['gm1']])
        pc0_labels_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc0_labels_after_mask_ground, batch_first=True, padding_value=0)
        pc1_labels_after_mask_ground = torch.nn.utils.rnn.pad_sequence(pc1_labels_after_mask_ground, batch_first=True, padding_value=0)
        res_dict['pc0_labels'] = pc0_labels_after_mask_ground
        res_dict['pc1_labels'] = pc1_labels_after_mask_ground

    if 'pseudo_flow' in batch[0]:
        pseudo_flow = torch.nn.utils.rnn.pad_sequence([batch[i]['pseudo_flow'][~batch[i]['gm0']] for i in range(len(batch))], batch_first=True)
        res_dict['pseudo_flow'] = pseudo_flow
    if 'use_gt' in batch[0]:
        res_dict['use_gt'] = [batch[i]['use_gt'] for i in range(len(batch))]
    return res_dict


class HDF5Dataset(Dataset):
    def __init__(self, directory, n_frames=2, dynamic_classifier=None, eval=False, save=False, leaderboard_version=1, pseudo_labels=None, gt_fraction=1., weather='all_weather'):
        '''
        directory: the directory of the dataset
        n_frames: the number of frames we use, default is 2: current, next if more then it's the history from current.
        dufo: if True, we will read the dynamic cluster label
        eval: if True, use the eval index
        '''
        super(HDF5Dataset, self).__init__()
        self.directory = directory
        
        with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
        
        self.pseudo_labels = pseudo_labels
        self.gt_fraction = gt_fraction

        if os.path.exists(os.path.join(self.directory, 'is_keyframe_index_total.pkl')):
            with open(os.path.join(self.directory, 'is_keyframe_index_total.pkl'), 'rb') as f:
                is_keyframe_index = pickle.load(f)
        else:
            is_keyframe_index = [True] * len(self.data_index)

        if self.gt_fraction >=0 :
            num_keyframes = sum(is_keyframe_index) * self.gt_fraction
            keyframe_indices = [i for i in range(len(is_keyframe_index)) if is_keyframe_index[i]]
            sampled_indices = random.sample(keyframe_indices, int(num_keyframes))

        # create a mapping to data index based
        if self.pseudo_labels == None and self.gt_fraction > 0 and not save:
            # supervised learning, use a fraction of random sampled keyframes
            self.mapping_to_data_index = sampled_indices
            self.use_gt = [True] * len(self.mapping_to_data_index)
        elif self.pseudo_labels != None and self.gt_fraction >= 0:
            # semi-supervised learning, use all frames
            self.mapping_to_data_index = [i for i in range(len(is_keyframe_index))]
            # append to self.mapping_to_data_index a fraction of random sampled keyframes
            sampled_keyframes = [self.data_index[i] for i in sampled_indices]
            # add to self.mapping_to_data_index the sampled keyframes
            self.mapping_to_data_index += sampled_indices
            # self.use_gt is true only for sampled keyframes
            self.use_gt = [False] * len(is_keyframe_index) + [True] * int(num_keyframes)
        else:
            # unsupervised/self-supervised learning, use all frames
            self.mapping_to_data_index = [i for i in range(len(is_keyframe_index))]
            self.use_gt = [False] * len(is_keyframe_index)

        self.eval_index = False
        self.dynamic_classifier = dynamic_classifier
        self.n_frames = n_frames

        if eval:
            eval_index_file = os.path.join(self.directory, 'index_eval.pkl')
            if leaderboard_version == 2:
                print("Using index to leaderboard version 2!!")
                eval_index_file = os.path.join(BASE_DIR, 'assets/docs/index_eval_v2.pkl')
            if os.path.exists(eval_index_file):
                with open(eval_index_file, 'rb') as f:
                    self.eval_data_index = pickle.load(f)
            else:
                # just load total index
                with open(os.path.join(self.directory, 'index_total.pkl'), 'rb') as f:
                    self.eval_data_index = pickle.load(f)
                if os.path.exists(os.path.join(self.directory, 'is_keyframe_index_total.pkl')):
                    self.eval_data_index = [self.eval_data_index[i] for i in range(len(self.eval_data_index)) if is_keyframe_index[i]]
                # filter according to the weather condition
                if weather != 'all_weather':
                    # read meta index
                    with open(os.path.join(self.directory, 'meta_index.pkl'), 'rb') as f:
                        meta_index = pickle.load(f)
                    if weather == 'bad_weather':
                        bad_weather_tags = ['rain', 'snow', 'fog', 'hail', 'other_weather']
                        # filter out the scenes that have either of the bad weather tags
                        self.eval_data_index = [item for item in self.eval_data_index if any(tag in meta_index[item[0]] for tag in bad_weather_tags)]
                    else:
                        self.eval_data_index = [item for item in self.eval_data_index if weather in meta_index[item[0]]]
                # raise Exception(f"No eval index file found! Please check {self.directory}")
            self.eval_index = eval
            self.use_gt = [True] * len(self.eval_data_index)

        self.scene_id_bounds = {}  # 存储每个scene_id的最大最小timestamp和位置
        for idx, (scene_id, timestamp) in enumerate(self.data_index):
            if scene_id not in self.scene_id_bounds:
                self.scene_id_bounds[scene_id] = {
                    "min_timestamp": timestamp,
                    "max_timestamp": timestamp,
                    "min_index": idx,
                    "max_index": idx
                }
            else:
                bounds = self.scene_id_bounds[scene_id]
                # 更新最小timestamp和位置
                if timestamp < bounds["min_timestamp"]:
                    bounds["min_timestamp"] = timestamp
                    bounds["min_index"] = idx
                # 更新最大timestamp和位置
                if timestamp > bounds["max_timestamp"]:
                    bounds["max_timestamp"] = timestamp
                    bounds["max_index"] = idx

    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        else:
            return len(self.mapping_to_data_index)
    
    def __getitem__(self, index_):
        # whether the sample has ground truth
        use_gt = self.use_gt[index_]

        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
            # find this one index in the total index
            index_ = self.data_index.index([scene_id, timestamp])
        else:
            index_ = self.mapping_to_data_index[index_]

            scene_id, timestamp = self.data_index[index_]
            # to make sure we have continuous frames
            if self.scene_id_bounds[scene_id]["max_index"] == index_:
                index_ = index_ - 1
            # get the data again
            scene_id, timestamp = self.data_index[index_]

        key = str(timestamp)
        with h5py.File(os.path.join(self.directory, f'{scene_id}.h5'), 'r') as f:
            pc0 = torch.tensor(f[key]['lidar'][:][:,:3])
            gm0 = torch.tensor(f[key]['ground_mask'][:])
            pose0 = torch.tensor(f[key]['pose'][:])

            next_timestamp = str(self.data_index[index_+1][1])
            pc1 = torch.tensor(f[next_timestamp]['lidar'][:][:,:3])
            gm1 = torch.tensor(f[next_timestamp]['ground_mask'][:])
            pose1 = torch.tensor(f[next_timestamp]['pose'][:])
            res_dict = {
                'scene_id': scene_id,
                'timestamp': key,
                'pc0': pc0,
                'gm0': gm0,
                'pose0': pose0,
                'pc1': pc1,
                'gm1': gm1,
                'pose1': pose1,
            }

            if self.n_frames > 2: 
                past_frames = []
                num_past_frames = self.n_frames - 2  

                for i in range(1, num_past_frames + 1):
                    frame_index = index_ - i
                    if frame_index < self.scene_id_bounds[scene_id]["min_index"]: 
                        frame_index = self.scene_id_bounds[scene_id]["min_index"] 

                    past_timestamp = str(self.data_index[frame_index][1])
                    past_pc = torch.tensor(f[past_timestamp]['lidar'][:][:,:3])
                    past_gm = torch.tensor(f[past_timestamp]['ground_mask'][:])
                    past_pose = torch.tensor(f[past_timestamp]['pose'][:])

                    past_frames.append((past_pc, past_gm, past_pose))

                for i, (past_pc, past_gm, past_pose) in enumerate(past_frames):
                    res_dict[f'pch{i+1}'] = past_pc
                    res_dict[f'gmh{i+1}'] = past_gm
                    res_dict[f'poseh{i+1}'] = past_pose

            if 'flow' in f[key]:
                flow = torch.tensor(f[key]['flow'][:])
                flow_is_valid = torch.tensor(f[key]['flow_is_valid'][:])
                flow_category_indices = torch.tensor(f[key]['flow_category_indices'][:])
                res_dict['flow'] = flow
                res_dict['flow_is_valid'] = flow_is_valid
                res_dict['flow_category_indices'] = flow_category_indices
            if self.pseudo_labels!= None and self.pseudo_labels in f[key]:
                pc_pseudo_flow = torch.tensor(f[key][self.pseudo_labels][:])
                res_dict['pseudo_flow'] = pc_pseudo_flow

            res_dict['use_gt'] = use_gt

            if 'ego_motion' in f[key]:
                ego_motion = torch.tensor(f[key]['ego_motion'][:])
                res_dict['ego_motion'] = ego_motion

            # add more dynamic classification methods here
            if self.dynamic_classifier == 'dufo':
                res_dict['pc0_labels'] = torch.tensor(f[key]['label'][:].astype('int16'))
                res_dict['pc1_labels'] = torch.tensor(f[next_timestamp]['label'][:].astype('int16'))
                res_dict['pc0_dynamic'] = res_dict['pc0_labels'] > 0
                res_dict['pc1_dynamic'] = res_dict['pc1_labels'] > 0
            elif self.dynamic_classifier == 'radar':
                res_dict['pc0_labels'] = torch.tensor(f[key]['pc_cluster_label'][:].astype('int16'))
                res_dict['pc1_labels'] = torch.tensor(f[next_timestamp]['pc_cluster_label'][:].astype('int16'))
                res_dict['pc0_dynamic'] = torch.tensor(f[key]['pc_dynamic_mask'][:].astype('bool'))
                res_dict['pc1_dynamic'] = torch.tensor(f[next_timestamp]['pc_dynamic_mask'][:].astype('bool'))

            if self.eval_index:
                # looks like v2 not follow the same rule as v1 with eval_mask provided
                eval_mask = torch.tensor(f[key]['eval_mask'][:]) if 'eval_mask' in f[key] else torch.ones_like(pc0[:, 0], dtype=torch.bool)
                res_dict['eval_mask'] = eval_mask

            if 'radar' in f[key].keys():
                res_dict['radar0'] = torch.tensor(f[key]['radar'][:])
                res_dict['radar0_id'] = torch.tensor(f[key]['radar_id'][:])
                # res_dict['radar0_rcs'] = torch.tensor(f[key]['radar_rcs'][:])
                res_dict['radar0_to_refego_tf'] = torch.tensor(f[key]['radar_to_refego_tf'][:])
                # res_dict['radar0_flow'] = torch.tensor(f[key]['radar_flow'][:])
                res_dict['radar0_flow_raw'] = torch.tensor(f[key]['radar_flow_raw'][:])

                res_dict['radar1'] = torch.tensor(f[next_timestamp]['radar'][:])

            # if 'radar_dynamic_mask' in f[key]:
            #     res_dict['radar0_dynamic_mask'] = torch.tensor(f[key]['radar_dynamic_mask'][:].astype('bool'))
            # if 'pc_dynamic_mask' in f[key]:
            #     res_dict['pc0_dynamic_mask'] = torch.tensor(f[key]['pc_dynamic_mask'][:].astype('bool'))
            # if 'pc_associated_radar' in f[key]:
            #     res_dict['pc0_associated_radar'] = torch.tensor(f[key]['pc_associated_radar'][:].astype('int16'))
            if 'pc_cluster_label' in f[key]:
                res_dict['pc0_cluster_label'] = torch.tensor(f[key]['pc_cluster_label'][:].astype('int16'))
                res_dict['pc1_cluster_label'] = torch.tensor(f[next_timestamp]['pc_cluster_label'][:].astype('int16'))
            if 'doppler_flow' in f[key]:
                res_dict['doppler0_flow'] = torch.tensor(f[key]['doppler_flow'][:])
                res_dict['doppler1_flow'] = torch.tensor(f[next_timestamp]['doppler_flow'][:])

            if 'cam' in f[key].keys():
                res_dict['cam0'] = torch.tensor(f[key]['cam'][:])
                res_dict['cam0_size'] = torch.tensor(f[key]['cam_size'][:])
                res_dict['cam0_intrinsic'] = torch.tensor(f[key]['cam_intrinsic'][:])
                res_dict['cam0_to_refego_tf'] = torch.tensor(f[key]['cam_to_refego_tf'][:])
        return res_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DataLoader test")
    parser.add_argument('--data_mode', '-m', type=str, default='val', metavar='N', help='Dataset mode.')
    parser.add_argument('--data_dir', '-d', type=str, default='/home/kin/data/av2/preprocess/sensor', metavar='N', help='preprocess data path.')
    options = parser.parse_args()

    # testing eval mode
    dataset = HDF5Dataset(options.data_dir+"/"+options.data_mode, eval=True)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16, collate_fn=collate_fn_pad)
    for data in tqdm(dataloader, ncols=80, desc="eval mode"):
        res_dict = data