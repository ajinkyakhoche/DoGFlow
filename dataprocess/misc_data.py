import numpy as np
import pickle, h5py, os, time
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, cast

NDArrayInt = np.typing.NDArray[np.int64]

def create_meta_index(desc_dict, input_val, data_dir: Path):
    # desc_dict is a dictionary with scene_id as key and a string of description as value
    # input_val is a list of scene_ids to be processed
    # filter desc_dict to only include scenes from input_val
    # save filtered desc_dict to data_dir as 'meta_index.pkl'
    start_time = time.time()
    meta_index = {}
    for scene_id in tqdm(input_val, ncols=100):
        if scene_id in desc_dict:
            meta_index[scene_id] = desc_dict[scene_id]
    with open(data_dir/'meta_index.pkl', 'wb') as f:
        pickle.dump(meta_index, f)
    print(f"Create meta index Successfully, cost: {time.time() - start_time:.2f} s")

def create_reading_index(data_dir: Path):
    start_time = time.time()
    data_index = []
    is_keyframe_index = []
    for file_name in tqdm(os.listdir(data_dir), ncols=100):
        if not file_name.endswith(".h5"):
            continue
        scene_id = file_name.split(".")[0]
        timestamps = []
        is_keyframe_list = []
        with h5py.File(data_dir/file_name, 'r') as f:
            timestamps.extend(f.keys())
            is_keyframe_list.extend(['flow' in f[t] for t in timestamps])
        
        # Combine timestamps and is_keyframe_list
        combined = list(zip(timestamps, is_keyframe_list))
        # Sort combined list based on timestamps
        combined.sort(key=lambda x: int(x[0]))
        # Separate sorted timestamps and is_keyframe_list
        timestamps, is_keyframe_list = zip(*combined)
        
        for timestamp in timestamps:
            data_index.append([scene_id, timestamp])
        for is_keyframe in is_keyframe_list:
            is_keyframe_index.append(is_keyframe)

    with open(data_dir/'is_keyframe_index_total.pkl', 'wb') as f:
        pickle.dump(is_keyframe_index, f)
    with open(data_dir/'index_total.pkl', 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create reading index Successfully, cost: {time.time() - start_time:.2f} s")

def find_closest_integer_in_ref_arr(
    query_int: int, ref_arr: NDArrayInt
) -> Tuple[int, int, int]:
    """Find the closest integer to any integer inside a reference array, and the corresponding difference.

    In our use case, the query integer represents a nanosecond-discretized timestamp, and the
    reference array represents a numpy array of nanosecond-discretized timestamps.

    Instead of sorting the whole array of timestamp differences, we just
    take the minimum value (to speed up this function).

    Args:
        query_int: query integer,
        ref_arr: Numpy array of integers

    Returns:
        integer, representing the closest integer found in a reference array to a query
        integer, representing the integer difference between the match and query integers
    """
    closest_ind = np.argmin(np.absolute(ref_arr - query_int))
    closest_int = cast(
        int, ref_arr[closest_ind]
    )  # mypy does not understand numpy arrays
    int_diff = np.absolute(query_int - closest_int)
    return closest_ind, closest_int, int_diff

class SE2:

    def __init__(self, rotation: np.ndarray, translation: np.ndarray) -> None:
        """Initialize.
        Args:
            rotation: np.ndarray of shape (2,2).
            translation: np.ndarray of shape (2,1).
        Raises:
            ValueError: if rotation or translation do not have the required shapes.
        """
        assert rotation.shape == (2, 2)
        assert translation.shape == (2, )
        self.rotation = rotation
        self.translation = translation
        self.transform_matrix = np.eye(3)
        self.transform_matrix[:2, :2] = self.rotation
        self.transform_matrix[:2, 2] = self.translation

    def transform_point_cloud(self, point_cloud: np.ndarray) -> np.ndarray:
        """Apply the SE(2) transformation to point_cloud.
        Args:
            point_cloud: np.ndarray of shape (N, 2).
        Returns:
            transformed_point_cloud: np.ndarray of shape (N, 2).
        Raises:
            ValueError: if point_cloud does not have the required shape.
        """
        assert point_cloud.ndim == 2
        assert point_cloud.shape[1] == 2
        num_points = point_cloud.shape[0]
        homogeneous_pts = np.hstack([point_cloud, np.ones((num_points, 1))])
        transformed_point_cloud = homogeneous_pts.dot(self.transform_matrix.T)
        return transformed_point_cloud[:, :2]

    def inverse(self) -> "SE2":
        """Return the inverse of the current SE2 transformation.
        For example, if the current object represents target_SE2_src, we will return instead src_SE2_target.
        Returns:
            inverse of this SE2 transformation.
        """
        return SE2(rotation=self.rotation.T,
                   translation=self.rotation.T.dot(-self.translation))

    def inverse_transform_point_cloud(self,
                                      point_cloud: np.ndarray) -> np.ndarray:
        """Transform the point_cloud by the inverse of this SE2.
        Args:
            point_cloud: Numpy array of shape (N,2).
        Returns:
            point_cloud transformed by the inverse of this SE2.
        """
        return self.inverse().transform_point_cloud(point_cloud)

    def compose(self, right_se2: "SE2") -> "SE2":
        """Multiply this SE2 from right by right_se2 and return the composed transformation.
        Args:
            right_se2: SE2 object to multiply this object by from right.
        Returns:
            The composed transformation.
        """
        chained_transform_matrix = self.transform_matrix.dot(
            right_se2.transform_matrix)
        chained_se2 = SE2(
            rotation=chained_transform_matrix[:2, :2],
            translation=chained_transform_matrix[:2, 2],
        )
        return chained_se2
    