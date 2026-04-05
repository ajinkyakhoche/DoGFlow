# DoGFlow: Self-Supervised LiDAR Scene Flow via Cross-Modal Doppler Guidance

[![Paper](https://img.shields.io/badge/arXiv-2508.18506-b31b1b.svg)](https://arxiv.org/abs/2508.18506)
[![Website](https://img.shields.io/badge/Project_Page-online-blue)](https://ajinkyakhoche.github.io/DogFlow/)
[![Video](https://img.shields.io/badge/video-YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/o6PElKWVWVk)

<img width="1061" height="849" alt="DoGFlow teaser" src="https://github.com/user-attachments/assets/386de8fa-5fdf-4ce3-9517-4a2f07480407" />

Official code release for the RA-L 2026 paper **DoGFlow: Self-Supervised LiDAR Scene Flow via Cross-Modal Doppler Guidance**.

This repository supports:
- Training-free DoGFlow inference and evaluation on the MAN TruckScenes dataset,
- Pseudo-label generation via radar-guided scene flow,
- Training and evaluation of LiDAR scene flow models using DoGFlow pseudo-labels.

DoGFlow is released within the [KTH-RPL/OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow) codebase and reuses its data processing, training, and visualization utilities.

## Updates
- 2026/01/18: DoGFlow accepted to IEEE Robotics and Automation Letters (RA-L).
- 2026/04/05: Initial code release.

## 0. Installation

We use conda to manage the environment, you can install it follow [here](assets/README.md#system). Then create the base environment with the following command [5~15 minutes]:

```bash
git clone --recursive https://github.com/ajinkyakhoche/DoGFlow.git
cd DoGFlow && conda env create -f environment.yaml

# You may need export your LD_LIBRARY_PATH with env lib
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kin/mambaforge/lib
```

CUDA package (need install nvcc compiler), the compile time is around 1-5 minutes:
```bash
mamba activate dogflow
# CUDA already install in python environment. I also tested others version like 11.3, 11.4, 11.7, 11.8 all works
cd assets/cuda/mmcv && python ./setup.py install && cd ../../..
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
cd assets/cuda/histlib/ && python ./setup.py install && cd ../../..
```

## 1. Data Preparation

Please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow?tab=readme-ov-file#1-data-preparation) for raw data download and h5py files preparation.

- If you have setup the MAN TruckScenes dataset correctly, simply run [dataprocess/extract_man.py](dataprocess/extract_man.py) after modifying the fields `data_dir` and `output_dir`

- Pre-compute noise-resistant LiDAR cluster labels 
```bash
python process_nr_lidar_clustering.py --data-dir ${data_path}/[mini,train,val,test]
```

## 2. Training-Free Evaluation

To evaluate DoGFlow, run:

```bash
python eval.py model=dogflow dataset_path=${data_path} weather=<WEATHER>
```

- We also provide implementations for NSFP, FastNSF and ICPFlow. Please replace the model tag to [nsfp, fastnsf, icpflow]
- The `weather` tag accepts [all_weather, bad_weather, clear, overcast, rain, snow, hail, fog, other_weather]. 
- We focus on the range-wise EPE as well as IoU metric. For instance, running DoGFlow on TruckScenes validation set containing snow weather gives:

```SSF Metric on Distance-based:
| Distance   |     Static |   Dynamic |   NumPointsStatic |   NumPointsDynamic |   DynamicIOU |
|------------+------------+-----------+-------------------+--------------------+--------------|
| 0-35       | 0.0445699  |  0.34014  |       4.09337e+06 |        1.08394e+06 |     0.825005 |
| 35-50      | 0.00308036 |  0.924955 |  485464           |     6019           |     0.805542 |
| 50-75      | 0.00186762 |  1.09663  |  281349           |     3794           |     0.648605 |
| 75-100     | 0.0032783  |  1.12325  |  130683           |     1761           |     0.600462 |
| 100-inf    | 0.00357058 |  1.01682  |  229243           |     1359           |     0.518419 |
| Mean       | 0.0112734  |  0.900359 |     nan           |      nan           |     0.679607 |
```

The script also reports the Three-Way EPE, Dynamic Normalized EPE. 

## 3. Training on pseudo labels 

- Compute pseudo labels on training set of truckscenes. (tip: this can take time, set `overwrite=False` to avoid starting from scratch):
```bash
python save.py model=dogflow dataset_path="${data_path}/train" overwrite=True
```

- Run the training with the following command (modify the data path accordingly):
```bash
python train.py train_data="${data_path}/train" val_data="${data_path}/val" model=ssf lr=8e-3 epochs=100 loss_fn=deflowLoss point_cloud_range="[-204.8, -204.8, -3, 204.8, 204.8, 3]" gt_fraction=0 pseudo_labels=dogflow
```

- To combine ground truth with pseudo labels during training, i.e. for semi-supervised training, increase gt_fraction.

- Please check assets/slurm/ for exact settings used in the paper.

### Evaluation

TODO: Upload pretrained weights link for MAN TruckScenes.

You can also run the evaluation by yourself with the following command with trained weights:
```bash
python eval.py checkpoint=${path_to_pretrained_weights} dataset_path=${data_path}
```

### Visualization

Please refer to the [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow/tree/main?tab=readme-ov-file#4-visualization) for visualization instructions. Tip: when you run/evaluate 
DoGFlow you can set `visualize=True` to enable rerun


## Cite & Acknowledgements
```
@article{khoche2026dogflow,
  author={Khoche, Ajinkya and Zhang, Qingwen and Cai, Yixi and Mansouri, Sina Sharif and Jensfelt, Patric},
  journal={IEEE Robotics and Automation Letters}, 
  title={DoGFlow: Self-Supervised LiDAR Scene Flow via Cross-Modal Doppler Guidance}, 
  year={2026},
  volume={11},
  number={3},
  pages={3836-3843},
  doi={10.1109/LRA.2026.3662592}
}
```
This work was supported by Prosense (2020-02963) funded by Vinnova. The computations were enabled by the supercomputing resource Berzelius provided by National Supercomputer Centre at Linköping University and the Knut and Alice Wallenberg Foundation, Sweden.

We thank the authors of [NSFP](https://github.com/Lilac-Lee/Neural_Scene_Flow_Prior), [FastNSF](https://github.com/Lilac-Lee/FastNSF) and [ICPFlow](https://github.com/yanconglin/ICP-Flow) for making their code publicly available. Thanks to [Qingwen Zhang](https://kin-zhang.github.io/) for integrating these methods into OpenSceneFlow
