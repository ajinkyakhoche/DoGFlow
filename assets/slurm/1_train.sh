#!/bin/bash
#SBATCH -J trucksc
#SBATCH -A berzelius-2024-479 --gpus-per-node 8 --nodes 1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=khoche@kth.se
#SBATCH --output /proj/berzelius-2023-364/users/x_ajikh/repos/OpenSceneFlow/logs/slurm/%J.out
#SBATCH --error  /proj/berzelius-2023-364/users/x_ajikh/repos/OpenSceneFlow/logs/slurm/%J.err

BASE_DIR=/proj/berzelius-2023-364/users/x_ajikh/repos/OpenSceneFlow
cd $BASE_DIR

DATASET="truckscenes" 
SOURCE=/proj/berzelius-2023-364/data/${DATASET}/preprocess/
DEST="/scratch/local/${DATASET}"
SUBDIRS=("train" "val")

start_time=$(date +%s)
for dir in "${SUBDIRS[@]}"; do
    mkdir -p "${DEST}/${dir}"
    find "${SOURCE}/${dir}" -type f -print0 | xargs -0 -n1 -P16 cp -t "${DEST}/${dir}" &
done
wait
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Copy ${SOURCE} to ${DEST} Total time: ${elapsed} seconds"
echo "Start training..."

# # ====> supervised model = deflow trucksc
# /proj/berzelius-2023-364/users/x_ajikh/.conda/envs/opensf/bin/python train.py \
#     slurm_id=${SLURM_JOB_ID} \
#     wandb_mode=online \
#     wandb_project_name=trucksc \
#     train_data="/scratch/local/${DATASET}/train" \
#     val_data="/scratch/local/${DATASET}/val" \
#     num_workers=16 \
#     model=deflow \
#     lr=2.5e-4 \
#     epochs=100 \
#     batch_size=8 \
#     loss_fn=deflowLoss \
#     keyframe_only=True \
#     val_every=1

# # ====> supervised model = ssf trucksc
# /proj/berzelius-2023-364/users/x_ajikh/.conda/envs/opensf/bin/python train.py \
#     slurm_id=${SLURM_JOB_ID:-local} \
#     wandb_mode=online \
#     wandb_project_name=trucksc \
#     train_data="${SOURCE}/train" \
#     val_data="${SOURCE}/val" \
#     num_workers=16 \
#     model=ssf \
#     lr=2e-3 \
#     epochs=100 \
#     batch_size=16 \
#     loss_fn=deflowLoss \
#     keyframe_only=True \
#     val_every=1 \
#     point_cloud_range="[-204.8, -204.8, -3, 204.8, 204.8, 3]"

# ====> leaderboard model = seflow trucksc
/proj/berzelius-2023-364/users/x_ajikh/.conda/envs/opensf/bin/python train.py \
    slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=dogflow train_data="${SOURCE}/train" val_data="${SOURCE}/val" \
    num_workers=16 model=deflow lr=2e-4 epochs=25 batch_size=12 "model.target.num_iters=2" "model.val_monitor=val/Dynamic/Mean" \
    loss_fn=seflowLoss "add_seloss={chamfer_dis: 1.0, static_flow_loss: 1.0, dynamic_chamfer_dis: 1.0, cluster_based_pc0pc1: 1.0}"
    
# # ====> leaderboard model = dogflow trucksc
# /proj/berzelius-2023-364/users/x_ajikh/.conda/envs/opensf/bin/python train.py \
#     slurm_id=$SLURM_JOB_ID wandb_mode=online wandb_project_name=dogf train_data="${SOURCE}/train" val_data="${SOURCE}/val" \
#     num_workers=16 model=ssf lr=2e-3 epochs=100 batch_size=20 "model.val_monitor=val/Dynamic/CAR" \
#     loss_fn=deflowLoss point_cloud_range="[-204.8, -204.8, -3, 204.8, 204.8, 3]" use_pseudo_labels=True
