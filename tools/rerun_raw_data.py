from __future__ import annotations

import argparse
import pathlib
from typing import Any, Final

import matplotlib
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
# from nuscenes import nuscenes
from truckscenes import TruckScenes
from truckscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from pyquaternion import Quaternion
import os, sys
BASE_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from src.utils.mics import flow_to_rgb

DESCRIPTION = """
# TruckScenes
Author: Ajinkya Khoche  (khoche@kth.se, ajinkya.khoche@scania.com)
Visualize the [TruckScenes dataset](https://www.nuscenes.org/) including lidar, radar, images, and bounding boxes data.

The code is modified from rerun example
[on GitHub](https://github.com/rerun-io/rerun/blob/latest/examples/python/nuscenes_dataset).
"""

EXAMPLE_DIR: Final = pathlib.Path(__file__).parent.parent
DATASET_DIR: Final = "/home/ajinkya/datasets/truckscenes/man-truckscenes/"

# currently need to calculate the color manually
# see https://github.com/rerun-io/rerun/issues/4409
cmap = matplotlib.colormaps["turbo"]
cmap_lidar_id = {
    0: (1.0, 0.0, 0.0, 1.0),  # Red
    1: (0.0, 1.0, 0.0, 1.0),  # Green
    2: (0.0, 0.0, 1.0, 1.0),  # Blue
    3: (1.0, 1.0, 0.0, 1.0),  # Yellow
    4: (1.0, 0.0, 1.0, 1.0),  # Magenta
    5: (0.0, 1.0, 1.0, 1.0)   # Cyan
}
norm = matplotlib.colors.Normalize(
    vmin=3.0,
    vmax=75.0,
)
norm_speed = matplotlib.colors.Normalize(
        vmin=0,
        vmax=70.0,
    )
point_size = 0.4

def nuscene_sensor_names(nusc: TruckScenes, scene_name: str) -> set[str]:
    """Return all sensor names in the scene."""

    sensor_names = set()

    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    first_sample = nusc.get("sample", scene["first_sample_token"])
    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        if sample_data["sensor_modality"] == "camera":
            current_camera_token = sample_data_token
            while current_camera_token != "":
                sample_data = nusc.get("sample_data", current_camera_token)
                sensor_name = sample_data["channel"]
                sensor_names.add(sensor_name)
                current_camera_token = sample_data["next"]

    return sensor_names


def log_nuscenes(nusc: TruckScenes, scene_name: str, max_time_sec: float) -> None:
    """Log TruckScenes scene."""

    scene = next(s for s in nusc.scene if s["name"] == scene_name)

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    first_sample_token = scene["first_sample_token"]
    first_sample = nusc.get("sample", scene["first_sample_token"])

    first_lidar_token = ""
    first_lidar_tokens = []
    first_radar_tokens = []
    first_camera_tokens = []
    for sample_data_token in first_sample["data"].values():
        sample_data = nusc.get("sample_data", sample_data_token)
        log_sensor_calibration(sample_data, nusc)

        if sample_data["sensor_modality"] == "lidar":
            first_lidar_token = sample_data_token
            first_lidar_tokens.append(sample_data_token)
        elif sample_data["sensor_modality"] == "radar":
            first_radar_tokens.append(sample_data_token)
        elif sample_data["sensor_modality"] == "camera":
            first_camera_tokens.append(sample_data_token)

    first_timestamp_us = nusc.get("sample_data", first_lidar_token)["timestamp"]
    max_timestamp_us = first_timestamp_us + 1e6 * max_time_sec

    first_ego_pose = log_lidar_and_ego_pose(first_lidar_tokens, nusc, max_timestamp_us)
    log_cameras(first_camera_tokens, nusc, max_timestamp_us)
    log_radars(first_radar_tokens, nusc, max_timestamp_us, first_ego_pose)
    log_annotations(first_sample_token, nusc, max_timestamp_us, first_ego_pose)


def log_lidar_and_ego_pose(first_lidar_tokens: list[str], nusc: TruckScenes, max_timestamp_us: float) -> None:
    """Log lidar data and vehicle pose."""
    first_ego_pose = None
    # current_lidar_token = first_lidar_token
    for sensor_id, first_lidar_token in enumerate(first_lidar_tokens):
        current_lidar_token = first_lidar_token
        while current_lidar_token != "":
            sample_data = nusc.get("sample_data", current_lidar_token)
            sensor_name = sample_data["channel"]

            if max_timestamp_us < sample_data["timestamp"]:
                break

            # timestamps are in microseconds
            rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)

            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            if first_ego_pose == None:
                first_ego_pose = ego_pose
            rotation_xyzw = np.roll(ego_pose["rotation"], shift=-1)  # go from wxyz to xyzw
            rr.log(
                "world/ego_vehicle",
                rr.Transform3D(
                    translation=[a-b for a,b in zip(ego_pose["translation"], first_ego_pose["translation"])],
                    rotation=rr.Quaternion(xyzw=rotation_xyzw),
                    from_parent=False,
                ),
            )
            current_lidar_token = sample_data["next"]

            data_file_path = nusc.dataroot / sample_data["filename"]
            pointcloud = LidarPointCloud.from_file(str(data_file_path))
            points = pointcloud.points[:3].T  # shape after transposing: (num_points, 3)
            point_distances = np.linalg.norm(points, axis=1)
            point_colors = cmap(norm(point_distances))
            point_colors = np.ones((points.shape[0], 4)) * cmap_lidar_id[sensor_id]
            rr.log(f"world/ego_vehicle/{sensor_name}", rr.Points3D(points, colors=point_colors, radii=np.ones((points.shape[0],))*point_size/4))
    return first_ego_pose


def log_cameras(first_camera_tokens: list[str], nusc: TruckScenes, max_timestamp_us: float) -> None:
    """Log camera data."""
    for first_camera_token in first_camera_tokens:
        current_camera_token = first_camera_token
        while current_camera_token != "":
            sample_data = nusc.get("sample_data", current_camera_token)
            if max_timestamp_us < sample_data["timestamp"]:
                break
            sensor_name = sample_data["channel"]
            rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)
            data_file_path = nusc.dataroot / sample_data["filename"]
            rr.log(f"world/ego_vehicle/{sensor_name}", rr.EncodedImage(path=data_file_path))
            current_camera_token = sample_data["next"]


def log_radars(first_radar_tokens: list[str], nusc: TruckScenes, max_timestamp_us: float, first_ego_pose: None) -> None:
    """Log radar data."""
    for first_radar_token in first_radar_tokens:
        current_radar_token = first_radar_token
        prev_ego_pose = None
        while current_radar_token != "":
            sample_data = nusc.get("sample_data", current_radar_token)
            if max_timestamp_us < sample_data["timestamp"]:
                break
            sensor_name = sample_data["channel"]
            rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)

            radar_cs_record = nusc.get('calibrated_sensor',
                                                     sample_data['calibrated_sensor_token'])
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
            rotation_xyzw = np.roll(ego_pose["rotation"], shift=-1)  # go from wxyz to xyzw
            rr.log(
                "world/ego_vehicle",
                rr.Transform3D(
                    translation=[a-b for a,b in zip(ego_pose["translation"], first_ego_pose["translation"])],
                    rotation=rr.Quaternion(xyzw=rotation_xyzw),
                    from_parent=False,
                ),
            )

            data_file_path = nusc.dataroot / sample_data["filename"]
            pointcloud = RadarPointCloud.from_file(str(data_file_path))
            points = pointcloud.points[:3].T  # shape after transposing: (num_points, 3)
            point_distances = np.linalg.norm(points, axis=1)
            point_colors = cmap(norm(point_distances))
            
            # velocities = pointcloud.points[3:5, :]
            # velocities = np.vstack((velocities, np.zeros(pointcloud.points.shape[1])))
            # velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix,
            #                     velocities)
            # velocities = np.dot(Quaternion(ego_pose['rotation']).rotation_matrix.T,
            #                     velocities)
            # velocities[2, :] = pointcloud.points[5, :] #np.zeros(pointcloud.points.shape[1])
            velocities = pointcloud.points[3:6, :]
            
            # color_speed = cmap(norm_speed(np.linalg.norm(pointcloud.points.T[:, 3:6], axis=1)))
            # labels = ["{:.2f} {:.2f} {:.2f}".format(x, y, z) for x, y, z in pointcloud.points.T[:, 3:6].round(2)]
            # color_speed = cmap(norm_speed(np.linalg.norm(velocities.T, axis=1)))
            color_speed = flow_to_rgb(velocities.T)
            labels = ["vel_raw={:.2f},{:.2f},{:.2f}".format(x, y, z) for x, y, z in velocities.T.round(2)]
            rr.log(
                f"world/ego_vehicle/{sensor_name}",
                rr.Points3D(points, colors=color_speed, radii=np.ones((points.shape[0],))*point_size, labels=labels),
            )
            current_radar_token = sample_data["next"]


def log_annotations(first_sample_token: str, nusc: TruckScenes, max_timestamp_us: float, first_ego_pose: None) -> None:
    """Log 3D bounding boxes."""
    label2id: dict[str, int] = {}
    current_sample_token = first_sample_token
    while current_sample_token != "":
        sample_data = nusc.get("sample", current_sample_token)
        if max_timestamp_us < sample_data["timestamp"]:
            break
        rr.set_time_seconds("timestamp", sample_data["timestamp"] * 1e-6)
        ann_tokens = sample_data["anns"]
        sizes = []
        centers = []
        quaternions = []
        class_ids = []
        for ann_token in ann_tokens:
            ann = nusc.get("sample_annotation", ann_token)

            rotation_xyzw = np.roll(ann["rotation"], shift=-1)  # go from wxyz to xyzw
            width, length, height = ann["size"]
            sizes.append((length, width, height))  # x, y, z sizes
            centers.append([a-b for a,b in zip(ann["translation"], first_ego_pose["translation"])])
            quaternions.append(rr.Quaternion(xyzw=rotation_xyzw))
            if ann["category_name"] not in label2id:
                label2id[ann["category_name"]] = len(label2id)
            class_ids.append(label2id[ann["category_name"]])

        rr.log(
            "world/anns",
            rr.Boxes3D(
                sizes=sizes,
                centers=centers,
                quaternions=quaternions,
                class_ids=class_ids,
                # fill_mode=rr.components.FillMode.Solid,
            ),
        )
        current_sample_token = sample_data["next"]

    # skipping for now since labels take too much space in 3D view (see https://github.com/rerun-io/rerun/issues/4451)
    # annotation_context = [(i, label) for label, i in label2id.items()]
    # rr.log("world/anns", rr.AnnotationContext(annotation_context), static=True)


def log_sensor_calibration(sample_data: dict[str, Any], nusc: TruckScenes) -> None:
    """Log sensor calibration (pinhole camera, sensor poses, etc.)."""
    sensor_name = sample_data["channel"]
    calibrated_sensor_token = sample_data["calibrated_sensor_token"]
    calibrated_sensor = nusc.get("calibrated_sensor", calibrated_sensor_token)
    rotation_xyzw = np.roll(calibrated_sensor["rotation"], shift=-1)  # go from wxyz to xyzw
    rr.log(
        f"world/ego_vehicle/{sensor_name}",
        rr.Transform3D(
            translation=calibrated_sensor["translation"],
            rotation=rr.Quaternion(xyzw=rotation_xyzw),
            from_parent=False,
        ),
        static=True,
    )
    if len(calibrated_sensor["camera_intrinsic"]) != 0:
        rr.log(
            f"world/ego_vehicle/{sensor_name}",
            rr.Pinhole(
                image_from_camera=calibrated_sensor["camera_intrinsic"],
                width=sample_data["width"],
                height=sample_data["height"],
            ),
            static=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualizes the nuScenes dataset using the Rerun SDK.")
    parser.add_argument(
        "--root-dir",
        type=pathlib.Path,
        default=DATASET_DIR,
        help="Root directory of nuScenes dataset",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="scene-13f4b71b1bd04a9e88747ad8f58a3f67-4", #"scene-3f542f89ec5241b6a4e30ca743adcf34-29",#"scene-0061",
        help="Scene name to visualize (typically of form 'scene-xxxx')",
    )
    parser.add_argument("--dataset-version", type=str, default="v1.0-mini", help="Scene id to visualize")
    parser.add_argument(
        "--seconds",
        type=float,
        default=float("inf"),
        help="If specified, limits the number of seconds logged",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()

    trucksc = TruckScenes(args.dataset_version, args.root_dir, True)

    # Set up the Rerun Blueprint (how the visualization is organized):
    sensor_space_views = [
        rrb.Spatial2DView(
            name=sensor_name,
            origin=f"world/ego_vehicle/{sensor_name}",
        )
        # for sensor_name in nuscene_sensor_names(nusc, args.scene_name)
        for sensor_name in nuscene_sensor_names(trucksc, args.scene_name)
    ]
    blueprint = rrb.Vertical(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="3D",
                origin="world",
                # Default for `ImagePlaneDistance` so that the pinhole frustum visualizations don't take up too much space.
                defaults=[rr.components.ImagePlaneDistance(4.0)],
                background=(255,255,255),
                # Transform arrows for the vehicle shouldn't be too long.
                overrides={"world/ego_vehicle": [rr.components.AxisLength(5.0)]},
            ),
            rrb.TextDocumentView(origin="description", name="Description"),
            column_shares=[3, 1],
        ),
        rrb.Grid(*sensor_space_views),
        row_shares=[4, 2],
    )

    rr.script_setup(args, "rerun_example_truckscenes", default_blueprint=blueprint)

    rr.log(
        "description",
        rr.TextDocument(DESCRIPTION, media_type=rr.MediaType.MARKDOWN),
        timeless=True,
    )

    log_nuscenes(trucksc, args.scene_name, max_time_sec=args.seconds)

    rr.script_teardown(args)


if __name__ == "__main__":
    main()