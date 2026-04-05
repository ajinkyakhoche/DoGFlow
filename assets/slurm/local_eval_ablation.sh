#!/bin/bash

export HYDRA_FULL_ERROR=1
SOURCE="/home/ajinkya/datasets/truckscenes/man-truckscenes/preprocess"

echo "Use this script to run robustness/sensitivity ablations locally..."

# # nominal case: no noise
# /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#     model=dogflow \
#     dataset_path=$SOURCE \
#     save_res=True \
#     weather=all_weather \
#     av2_mode=val \
#     "model.target.dynamic_classification_threshold=0.05" \
#     "model.target.max_association_error=5.0" \
#     "model.target.min_association_error=0.1" \
#     "model.target.thresh_eucledian=3" \
#     "model.target.thresh_speed=1.5" \
#     "model.target.radar_to_lidar_tf_noise=[0,0,0,0,0,0]" \

# List of dynamic_classification_threshold
DY_CLASSIFICATION_THRESH=(0) #(0.1 0.2 0.3 0.4 0.5)

for DY in "${DY_CLASSIFICATION_THRESH[@]}"; do
    echo "Running experiment with dynamic classification threshold = ${DY} ..."

    /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
        model=dogflow \
        dataset_path=$SOURCE \
        save_res=True \
        weather=all_weather \
        av2_mode=val \
        "model.target.dynamic_classification_threshold=${DY}" \
        "model.target.max_association_error=5.0" \
        "model.target.min_association_error=0.1" \
        "model.target.thresh_eucledian=3" \
        "model.target.thresh_speed=1.5" \
        "model.target.radar_to_lidar_tf_noise=[0,0,0,0,0,0]"
done

# # List of roll (rx) noise values in degrees
# ROLL_VALUES=(1 2 3)

# for ROLL in "${ROLL_VALUES[@]}"; do
#     echo "Running experiment with roll noise = ${ROLL} deg ..."

#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[0,0,0,${ROLL},0,0]"
# done

# # List of pitch (ry) noise values in degrees
# PITCH_VALUES=(1 2 3)

# for PITCH in "${PITCH_VALUES[@]}"; do
#     echo "Running experiment with pitch noise = ${PITCH} deg ..."

#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[0,0,0,0,${PITCH},0]"
# done

# # List of yaw (rz) noise values in degrees
# YAW_VALUES=(1 2 3)

# for YAW in "${YAW_VALUES[@]}"; do
#     echo "Running experiment with yaw noise = ${YAW} deg ..."

#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[0,0,0,0,0,${YAW}]"
# done

# # List of translation (tx) noise values in meters
# TX_VALUES=(0.2 0.4 0.6 0.8 1.0)

# for TX in "${TX_VALUES[@]}"; do
#     echo "Running experiment with translation noise = ${TX} m ..."

#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[${TX},0,0,0,0,0]"
# done

# # List of translation (ty) noise values in meters
# TY_VALUES=(0.2 0.4 0.6 0.8 1.0)

# for TY in "${TY_VALUES[@]}"; do
#     echo "Running experiment with translation noise = ${TY} m ..."

#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[0,${TY},0,0,0,0]"
# done

# # List of translation (ty) noise values in meters
# TZ_VALUES=(0.2 0.4 0.6 0.8 1.0)

# for TZ in "${TZ_VALUES[@]}"; do
#     echo "Running experiment with translation noise = ${TZ} m ..."

#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[0,0,${TZ},0,0,0]"
# done

# # List of eucledian dist threshold in meters
# THRESH_EUCLEDIAN=(1 2 4)

# for THRESH in "${THRESH_EUCLEDIAN[@]}"; do
#     echo "Running experiment with eucledian distance threshold = ${THRESH} m ..."
#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=${THRESH}" \
#         "model.target.thresh_speed=1.5" \
#         "model.target.radar_to_lidar_tf_noise=[0,0,0,0,0,0]"
# done

# # List of speed threshold in m/s
# THRESH_SPEED=(0.5 1.0 2.0)
# for THRESH in "${THRESH_SPEED[@]}"; do
#     echo "Running experiment with speed threshold = ${THRESH} m/s ..."
#     /home/neo/miniforge-pypy3/envs/opensf-pt3d/bin/python eval.py \
#         model=dogflow \
#         dataset_path=$SOURCE \
#         save_res=True \
#         weather=all_weather \
#         av2_mode=val \
#         "model.target.dynamic_classification_threshold=0.05" \
#         "model.target.max_association_error=5.0" \
#         "model.target.min_association_error=0.1" \
#         "model.target.thresh_eucledian=3" \
#         "model.target.thresh_speed=${THRESH}" \
#         "model.target.radar_to_lidar_tf_noise=[0,0,0,0,0,0]"
# done