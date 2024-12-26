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
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
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


def splt_gaussians(model_path, name, iteration, views, gaussians, pipeline, background):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # if not os.path.exists(render_path):
    #     os.makedirs(render_path)
    # if not os.path.exists(gts_path):
    #     os.makedirs(gts_path)

    max_memory = 0
    name_list = []
    per_view_dict = {}
    t_list = []
    mask_vis = torch.zeros(gaussians.get_anchor.shape[0], dtype=torch.bool, device="cuda")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        mask_vis = torch.logical_or(mask_vis, voxel_visible_mask)
        # render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)
        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        max_memory = max(max_memory, forward_max_memory_allocated)

        # rendering = render_pkg["render"]
        # gt = view.original_image[0:3, :, :]
        # name_list.append('{0:05d}'.format(idx) + ".png")
        # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    # t = np.array(t_list[5:])
    # fps = 1.0 / t.mean()
    # print(f'Test Mean FPS: \033[1;35m{fps:.2f}\033[0m')
    # print(f'Max Inference Memory: \033[1;35m{max_memory:.2f} MB \033[0m')    
    new_anchor = gaussians._anchor[mask_vis]
    new_offset = gaussians._offset[mask_vis]
    new_anchor_feat = gaussians._anchor_feat[mask_vis]
    new_opacity = gaussians._opacity[mask_vis]
    new_scaling = gaussians._scaling[mask_vis]
    new_rotation = gaussians._rotation[mask_vis]

    gaussians._anchor = new_anchor
    gaussians._offset = new_offset
    gaussians._anchor_feat = new_anchor_feat
    gaussians._opacity = new_opacity
    gaussians._scaling = new_scaling
    gaussians._rotation = new_rotation

    return gaussians 
     
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, custom_test : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)

        if custom_test:
            dataset.source_path = custom_test
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        gaussians = splt_gaussians(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        point_cloud_path = os.path.join(dataset.model_path, f"point_cloud/iteration_{iteration}_blockA_aera")
        os.makedirs(point_cloud_path, exist_ok=True)
        gaussians.save_mlp_checkpoints(point_cloud_path)
        gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        print("Saved point cloud to " + os.path.join(point_cloud_path, "point_cloud.ply"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.custom_test)
