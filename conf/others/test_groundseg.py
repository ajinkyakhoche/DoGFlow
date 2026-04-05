import open3d as o3d
import numpy as np
from linefit import ground_seg
import os, sys
PARENT_DIR = os.path.abspath(os.path.join( os.path.dirname( __file__ ), '..'))
sys.path.append(PARENT_DIR)

GROUNDSEG_config = f"conf/others/truckscenes.toml"

mygroundseg = ground_seg(GROUNDSEG_config)

pc0 = np.load(f"{PARENT_DIR}/others/pc.npy")
is_ground_0 = np.array(mygroundseg.run(pc0[:, :3].tolist()))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc0[:,:3])
pcd_color = np.ones_like(pc0[:,:3]) * [0, 0, 1]
pcd_color[is_ground_0] = [1, 0, 0]
pcd.colors = o3d.utility.Vector3dVector(pcd_color)
o3d.visualization.draw_geometries([pcd])