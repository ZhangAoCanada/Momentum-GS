# pylint: disable=[E0402]

import os
import json

from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from pathlib import Path

from tools.read_write_model import (
    Camera, Image, write_cameras_text, write_images_text, write_points3D_text, write_cameras_binary, write_images_binary, write_points3D_binary
)
from tools.utils import list_images, get_camera_id, list_jsons


def mat2quat(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat
    K = np.array([
        [Qxx - Qyy - Qzz, 0,               0,               0              ],
        [Qyx + Qxy,       Qyy - Qxx - Qzz, 0,               0              ],
        [Qzx + Qxz,       Qzy + Qyz,       Qzz - Qxx - Qyy, 0              ],
        [Qyz - Qzy,       Qzx - Qxz,       Qxy - Qyx,       Qxx + Qyy + Qzz]]
        ) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q *= -1
    return q


option = "test"
scene_dir = f"/data/zhangao/bdaibdai___MatrixCity/small_city/blockA_fusion/{option}"
pose_dir = scene_dir.replace(f'{option}', 'pose/block_A')
poses_file = os.path.join(pose_dir, f'transforms_{option}.json')
MATRIX_CITY_TO_COLMAP = torch.FloatTensor([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])
TO_MANHATTAN_WORLD = torch.FloatTensor([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
])



def read_pose(
    scene_path: str, json_path: str, images: Dict, cameras: Dict
) -> Tuple[Dict, Dict]:
    with open(json_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file)

    num_frames = len(metadata["frames"])
    camera_model = metadata["camera_model"]

    count = 0
    for i in tqdm(range(num_frames)):
        frame = metadata["frames"][i]
        fx, fy = frame["fl_x"], frame["fl_y"]
        cx, cy = frame["cx"], frame["cy"]
        w, h = frame["w"], frame["h"]
        assert fx == fy, "Invalid PINHOLE params!"

        fx, fy, cx, cy, w, h = int(fx), int(fy), int(cx), int(cy), int(w), int(h)

        params = [fx, cx, cy]
        camera = Camera(id=-1, model=camera_model, width=w, height=h, params=params)
        camera_id = get_camera_id(cameras, camera)
        camera = Camera(id=camera_id, model=camera_model, width=w, height=h, params=params)
        cameras[camera_id] = camera

        image_path = frame["file_path"]

        ############# NOTE: copy image to colmap input folder ############
        if not os.path.exists(image_path):
            print(image_path)
        image_name = '%04d.png' % i
        os.system("cp {} {}".format(image_path, os.path.join(scene_path, 'input', image_name)))
        ##################################################################
        
        image_id = i
        c2w = torch.from_numpy(np.array(frame["transform_matrix"])).float()
        # c2w[:3, :3] = TO_MANHATTAN_WORLD @ c2w[:3, :3] @ MATRIX_CITY_TO_COLMAP
        # c2w[:3, :3] = TO_MANHATTAN_WORLD @ c2w[:3, :3]
        c2w[:3, :3] = c2w[:3, :3] @ MATRIX_CITY_TO_COLMAP
        w2c = torch.inverse(c2w)
        qvec = R.from_matrix(w2c[:3, :3].numpy()).as_quat()
        tvec = w2c[:3, 3]
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
        tvec = np.array([tvec[0], tvec[1], tvec[2]])

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[],
        )

    return images, cameras



images, cameras = {}, {}
read_pose(
    scene_dir, poses_file, images, cameras
)

colmap_dir = os.path.join(scene_dir, "sparse/0")
os.makedirs(colmap_dir, exist_ok=True)

print(f'num images: {len(images)}')

# print(f'colmap data written to:{colmap_dir}')
# write_cameras_binary(cameras, os.path.join(colmap_dir, "cameras.bin"))
# write_images_binary(images, os.path.join(colmap_dir, "images.bin"))
# write_points3D_binary({}, os.path.join(colmap_dir, "points3D.bin"))
