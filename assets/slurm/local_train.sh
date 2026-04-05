#!/bin/bash

export HYDRA_FULL_ERROR=1
SOURCE="/home/ajinkya/datasets/truckscenes/man-truckscenes/preprocess"

echo "Use this script to check your sbatch by running training locally..."

# # ====> supervised model = deflow
# /proj/berzelius-2023-364/users/x_ajikh/.conda/envs/opensf/bin/python train.py \
#     slurm_id=${SLURM_JOB_ID:-local} \
#     wandb_mode=disabled \
#     wandb_project_name=trucksc \
#     train_data="${SOURCE}/train" \
#     val_data="${SOURCE}/val" \
#     num_workers=16 \
#     model=ssf \
#     lr=2.5e-4 \
#     epochs=25 \
#     batch_size=8 \
#     loss_fn=deflowLoss \
#     val_every=1 \
#     keyframe_only=True

# # ====> supervised model = ssf
# /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python train.py \
#     slurm_id=${SLURM_JOB_ID:-local} \
#     wandb_mode=disabled \
#     wandb_project_name=trucksc \
#     train_data="${SOURCE}/train" \
#     val_data="${SOURCE}/val" \
#     num_workers=16 \
#     model=ssf \
#     lr=8e-3 \
#     epochs=100 \
#     batch_size=8 \
#     loss_fn=deflowLoss \
#     val_every=1 \
#     gt_fraction=1 \
#     point_cloud_range="[-204.8, -204.8, -3, 204.8, 204.8, 3]"

# # ====> leaderboard model = seflow trucksc
# /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python train.py \
#     slurm_id=$SLURM_JOB_ID \
#     wandb_mode=disabled \
#     train_data="${SOURCE}/train" \
#     val_data="${SOURCE}/val" \
#     num_workers=16 \
#     model=ssf \
#     lr=8e-3 \
#     epochs=100 \
#     batch_size=16 \
#     "model.val_monitor=val/Dynamic/Mean" \
#     loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}" \
#     val_every=1 \
#     dynamic_classifier=dufo \
#     point_cloud_range="[-204.8, -204.8, -3, 204.8, 204.8, 3]" \
#     # "model.target.num_iters=2" \
    
# # ====> model = dogflow trucksc
/home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python train.py \
    slurm_id=$SLURM_JOB_ID \
    wandb_mode=disabled \
    train_data="${SOURCE}/train" \
    val_data="${SOURCE}/val" \
    num_workers=16 \
    model=ssf \
    lr=8e-3 \
    epochs=100 \
    batch_size=8 \
    "model.val_monitor=val/Dynamic/Mean" \
    loss_fn=deflowLoss \
    point_cloud_range="[-204.8, -204.8, -3, 204.8, 204.8, 3]" \
    gt_fraction=0 \
    pseudo_labels=dogflow