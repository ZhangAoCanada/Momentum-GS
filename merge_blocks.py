#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch
import numpy as np

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.partition_utils import contract_to_unisphere
import matplotlib.pyplot as plt


def merge_blocks(dataset : ModelParams, iteration : int):
    out_dir = dataset.model_path
    merged_gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    block_num = dataset.block_num

    print(f'### block_num = {block_num}')

    with torch.no_grad():
        for idx in range(block_num):
            gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)

            pretrained_pc_path = os.path.join(dataset.model_path, 'point_cloud', f'iteration_{iteration}' , f'point_cloud_block{idx}.ply')
            print(f'### Loading pretrained point cloud from {pretrained_pc_path}')
            gaussians.load_ply_sparse_gaussian(pretrained_pc_path)

            # calculate the boundry of each block
            dataset.aabb = torch.tensor(dataset.aabb, dtype=torch.float32, device=gaussians._anchor.device)

            block_id_z = idx // (dataset.block_dim[0] * dataset.block_dim[1]) # 0,0,1,1
            block_id_y = (idx % (dataset.block_dim[0] * dataset.block_dim[1])) // dataset.block_dim[0] # 0,0,0,0
            block_id_x = (idx % (dataset.block_dim[0] * dataset.block_dim[1])) % dataset.block_dim[0] # 0,1,0,1

            xyz = contract_to_unisphere(gaussians._anchor, dataset.aabb, ord=torch.inf)
            min_x, max_x = float(block_id_x) / dataset.block_dim[0], float(block_id_x + 1) / dataset.block_dim[0]
            min_y, max_y = float(block_id_y) / dataset.block_dim[1], float(block_id_y + 1) / dataset.block_dim[1]
            min_z, max_z = float(block_id_z) / dataset.block_dim[2], float(block_id_z + 1) / dataset.block_dim[2]

            block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                        & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                        & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)
            num_gs = block_mask.sum()

            print(f"[Block {idx}] After filter, the number of voxel: {gaussians._anchor.shape[0]} -> {num_gs}")
            
            if len(merged_gaussians._anchor) == 0:
                merged_gaussians._anchor = gaussians._anchor[block_mask]
                merged_gaussians._anchor_feat = gaussians._anchor_feat[block_mask]
                merged_gaussians._offset = gaussians._offset[block_mask]
                merged_gaussians._scaling = gaussians._scaling[block_mask]
                merged_gaussians._rotation = gaussians._rotation[block_mask]
                merged_gaussians._opacity = gaussians._opacity[block_mask]
            else:
                merged_gaussians._anchor = torch.cat([merged_gaussians._anchor, gaussians._anchor[block_mask]], dim=0)
                merged_gaussians._anchor_feat = torch.cat([merged_gaussians._anchor_feat, gaussians._anchor_feat[block_mask]], dim=0)
                merged_gaussians._offset = torch.cat([merged_gaussians._offset, gaussians._offset[block_mask]], dim=0)
                merged_gaussians._scaling = torch.cat([merged_gaussians._scaling, gaussians._scaling[block_mask]], dim=0)
                merged_gaussians._rotation = torch.cat([merged_gaussians._rotation, gaussians._rotation[block_mask]], dim=0)
                merged_gaussians._opacity = torch.cat([merged_gaussians._opacity, gaussians._opacity[block_mask]], dim=0)

    save_path = os.path.join(out_dir, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    print(f"Saving merged {len(merged_gaussians._anchor)} point cloud to {save_path}")
    merged_gaussians.save_ply(save_path)
    print('Done')


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Merging " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.custom_test)
    merge_blocks(model.extract(args), args.iteration)
