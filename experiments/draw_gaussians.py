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
from gaussian_renderer import render, prefilter_voxel, render_anchor
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import gsplat


def show_gaussians(model_path, name, iteration, views, gaussians, pipeline, background):
    max_memory = 0
    name_list = []
    per_view_dict = {}
    t_list = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.reset_peak_memory_stats()

        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg_feat = render_anchor(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, scaling_modifier=0.5)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, scaling_modifier=0.2)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)
        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        max_memory = max(max_memory, forward_max_memory_allocated)

        rendering_feat = render_pkg_feat["render"]
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :].cuda()
        name_list.append('{0:05d}'.format(idx) + ".png")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        torchvision.utils.save_image(rendering_feat, os.path.join(current_dir, "tmp", '{0:05d}'.format(idx) + "_feat.png"))
        torchvision.utils.save_image(rendering, os.path.join(current_dir, "tmp", '{0:05d}'.format(idx) + "_render.png"))
        torchvision.utils.save_image(gt, os.path.join(current_dir, "tmp", '{0:05d}'.format(idx) + "_gt.png"))

     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, custom_test : str):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)

        if custom_test:
            dataset.source_path = custom_test
        iteration = str(iteration) + "_blockA_aera"
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        
        show_gaussians(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


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
