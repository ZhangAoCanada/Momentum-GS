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
    Camera, Point3D, Image, write_cameras_text, write_images_text, write_points3D_text, write_cameras_binary, write_images_binary, write_points3D_binary, read_points3D_binary
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
scene_dir = f"/data/zhangao/bdaibdai___MatrixCity/small_city/blockA_fusion_small/{option}"
pose_dir = scene_dir.replace(f'{option}', 'pose/block_A')
poses_file = os.path.join(pose_dir, f'transforms_{option}.json')
point3D_raw_path = "/data/zhangao/bdaibdai___MatrixCity/small_city/aerial/train/block_all/sparse/0/points3D.bin"
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
    scene_path: str, json_path: str, images: Dict, cameras: Dict, points3d_new: Dict, point3D_raw_path: str = None
) -> Tuple[Dict, Dict]:
    with open(json_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file)

    num_frames = len(metadata["frames"])
    camera_model = metadata["camera_model"]
    points3d = read_points3D_binary(point3D_raw_path)
    points_xyz = np.array([point.xyz for point in points3d.values()])
    mask = np.zeros(points_xyz.shape[0], dtype=bool)

    scene_image_path = os.path.join(scene_path, 'input')
    scene_depth_path = os.path.join(scene_path, 'depth')
    scene_normal_path = os.path.join(scene_path, 'normal')
    os.makedirs(scene_image_path, exist_ok=True)
    os.makedirs(scene_depth_path, exist_ok=True)
    os.makedirs(scene_normal_path, exist_ok=True)

    x_max = -1 # range: -1.2, mean: -4.1
    x_min = -4 # range: -10.0, mean: -4.1
    y_max = -2 # range: 0., mean: -2.3
    y_min = -4 # range: -6.3, mean: -2.3
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
        
        image_id = i
        c2w = torch.from_numpy(np.array(frame["transform_matrix"])).float()
        # c2w[:3, :3] = TO_MANHATTAN_WORLD @ c2w[:3, :3] @ MATRIX_CITY_TO_COLMAP
        # c2w[:3, :3] = TO_MANHATTAN_WORLD @ c2w[:3, :3]
        c2w[:3, :3] = c2w[:3, :3] @ MATRIX_CITY_TO_COLMAP
        ################ NOTE: get a small region of the city ############ 
        camera_center = c2w[:3, 3]
        if (camera_center[0] > x_max or camera_center[0] < x_min) or (camera_center[1] > y_max or camera_center[1] < y_min):
            continue
        ### NOTE: copy images
        image_abs_path = os.path.abspath(image_path)
        image_abs_split = image_abs_path.split('/')
        image_parent_path = '/'.join(image_abs_split[:-2])
        depth_path = os.path.join(image_parent_path.replace("small_city", "small_city_depth"), f"{image_abs_split[-2]}_depth", image_abs_split[-1].replace("png", "exr"))
        normal_path = os.path.join(image_parent_path.replace("small_city", "small_city_normal"), f"{image_abs_split[-2]}_normal", image_abs_split[-1].replace("png", "exr"))
        if not os.path.exists(image_path):
            print(image_path)
            raise FileNotFoundError
        if not os.path.exists(depth_path):
            print(depth_path)
            raise FileNotFoundError
        if not os.path.exists(normal_path):
            print(normal_path)
            raise FileNotFoundError
        image_name = '%04d.png' % i
        depth_name = '%04d.exr' % i
        normal_name = '%04d.exr' % i
        os.system("cp {} {}".format(image_path, os.path.join(scene_image_path, image_name)))
        os.system("cp {} {}".format(depth_path, os.path.join(scene_depth_path, depth_name)))
        os.system("cp {} {}".format(normal_path, os.path.join(scene_normal_path, normal_name)))
        ## NOTE: remove the camera pose if it is othogonal to the z-axis ##
        # rotation_matrix = c2w[:3, :3]
        # if np.abs(rotation_matrix[0, 2]) < 1e-3 and np.abs(rotation_matrix[1, 2]) < 1e-3:
        #     continue
        ###################################################################
        w2c = torch.inverse(c2w)
        qvec = R.from_matrix(w2c[:3, :3].numpy()).as_quat()
        tvec = w2c[:3, 3]
        qvec = np.array([qvec[3], qvec[0], qvec[1], qvec[2]])
        tvec = np.array([tvec[0], tvec[1], tvec[2]])
        ################# NOTE: check if points are in the image ##########
        if "aerial" in image_path:
            points3d_incam = w2c[:3, :3] @ points_xyz.T + w2c[:3, 3].reshape(3, 1)
            z_mask = points3d_incam.cpu().numpy()[2] > 0
            points2d = np.array([fx * points3d_incam[0] / points3d_incam[2] + cx, fy * points3d_incam[1] / points3d_incam[2] + cy]) 
            current_mask = np.logical_and(points2d[0] > 0, np.logical_and(points2d[0] < w, np.logical_and(points2d[1] > 0, points2d[1] < h)))
            current_mask = np.logical_and(current_mask, z_mask)
            mask = np.logical_or(mask, current_mask)
        ###################################################################

        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=image_name,
            xys=[],
            point3D_ids=[],
        )

    ##### NOTE: remove the points that are not in the image #####
    print("[INFO] num points filtered: ", np.sum(mask))
    for i, (point_id, point) in tqdm(enumerate(points3d.items())):
        if mask[i]:
            assert point_id == point.id
            points3d_new[point_id] = Point3D(
                id=point.id,
                xyz=point.xyz,
                rgb=point.rgb,
                error=point.error,
                image_ids=point.image_ids,
                point2D_idxs=point.point2D_idxs,
            )

    return images, cameras, points3d_new



images, cameras, point3d_new = {}, {}, {}
read_pose(
    scene_dir, poses_file, images, cameras, point3d_new, point3D_raw_path
)

colmap_dir = os.path.join(scene_dir, "sparse/0")
os.makedirs(colmap_dir, exist_ok=True)

print(f'num images: {len(images)}')

print(f'colmap data written to:{colmap_dir}')
write_cameras_binary(cameras, os.path.join(colmap_dir, "cameras.bin"))
write_images_binary(images, os.path.join(colmap_dir, "images.bin"))
write_points3D_binary(point3d_new, os.path.join(colmap_dir, "points3D.bin"))
