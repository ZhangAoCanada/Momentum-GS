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
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))
# os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
from scene.interpolation import GaussianInterpolation
import json
import time
from gaussian_renderer import render, prefilter_voxel, render_anchor,render_gsplat , render_anchor_gsplat
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from scene.interpolation_point import PointInterpolation

import math, cv2
import gsplat

from scene.freq_modifier import FrequencyModifier
import shutil


def grid_upsample(image_tensor, scale):
    upsampled_image = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), scale_factor=scale, mode='nearest-exact')
    return upsampled_image.squeeze(0)


def show_gaussians(args, scene, freq, name, iteration, views, gaussians, pipeline, background, render_with_gsplat=False, render_with_anchor=False):
    save_dir = os.path.join(currentdir, "tmp")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    max_memory = 0
    name_list = []
    per_view_dict = {}
    t_list = []
    depth_threshold = 0.1
    scaling_factor=None
    resolution_scaling_factor=4
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        #######################################################
        # scale_freqs = 1.
        # offset_freqs = 2.
        # freq_scale_raw = freq.anchor_freq(gaussians, mode="anchor_scale")
        # freq_offset_raw = freq.anchor_freq(gaussians, mode="anchor_offset")
        # freq.plot_freq(freq_scale_raw, os.path.join(save_dir, '{0:05d}'.format(idx) + "_freq_scale_raw.png"))
        # freq.plot_freq(freq_offset_raw, os.path.join(save_dir, '{0:05d}'.format(idx) + "_freq_offset_raw.png"))
        # frequency_scale = freq.anchor_freq_smooth(gaussians, freq_scale=scale_freqs, mode="anchor_scale")
        # frequency_offset = freq.anchor_freq_smooth(gaussians, freq_scale=offset_freqs, mode="anchor_offset")
        # frequency_scale_fitlered = freq.filter_smooth(frequency_scale, freq_scale=scale_freqs)
        # frequency_offset_fitlered = freq.filter_smooth(frequency_offset, freq_scale=offset_freqs)
        # freq.plot_freq_compare(frequency_scale, frequency_scale_fitlered, os.path.join(save_dir, '{0:05d}'.format(idx) + "_freq_scale.png"))
        # freq.plot_freq_compare(frequency_offset, frequency_offset_fitlered, os.path.join(save_dir, '{0:05d}'.format(idx) + "_freq_offset.png"))
        # freq_filter_mask = torch.logical_and(frequency_scale_fitlered["freq_mask"], frequency_offset_fitlered["freq_mask"])
        # voxel_visible_mask = voxel_visible_mask * freq_filter_mask
        #######################################################
        gt = view.original_image[0:3, :, :].cuda()
        # upsample the ground truth with grid_sample
        depth_raw = view.original_depth.cuda()
        normal = view.original_normal.cuda()
        gt = grid_upsample(gt, resolution_scaling_factor)
        depth_raw = grid_upsample(depth_raw, resolution_scaling_factor)
        normal = grid_upsample(normal, resolution_scaling_factor)
        # depth = (depth_raw.repeat(3, 1, 1) - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
        # gt_show = torch.cat([gt, depth, normal], dim=1)
        if render_with_gsplat:
            if render_with_anchor:
                with torch.no_grad():
                    render_pkg_feat = render_anchor_gsplat(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, scaling_modifier=1, resolution_scaling_factor=resolution_scaling_factor, scaledown_ratio=scaling_factor)
            render_pkg = render_gsplat(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, scaling_modifier=1, resolution_scaling_factor=resolution_scaling_factor, scaledown_ratio=scaling_factor)
        else:
            if render_with_anchor:
                with torch.no_grad():
                    render_pkg_feat = render_anchor(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, scaling_modifier=1, resolution_scaling_factor=resolution_scaling_factor, scaledown_ratio=scaling_factor)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask, scaling_modifier=1, resolution_scaling_factor=resolution_scaling_factor, scaledown_ratio=scaling_factor)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)
        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        max_memory = max(max_memory, forward_max_memory_allocated)

        if render_with_anchor:
            rendering_feat = render_pkg_feat["render"]
        rendering = render_pkg["render"]
        rendering_diff = torch.abs(rendering - gt).mean(dim=0, keepdim=True).repeat(3, 1, 1)
        rendering_diff = torch.where(rendering_diff > 0.1, 1., 0.)
        rendering = torch.cat([gt, rendering, rendering_diff], dim=1)
        if render_with_anchor:
            rendering_feat_depth_raw = render_pkg_feat["depth"]
            rendering_feat_depth = (rendering_feat_depth_raw - rendering_feat_depth_raw.min()) / (rendering_feat_depth_raw.max() - rendering_feat_depth_raw.min())
            gt_depth = depth_raw / depth_raw.max()
            rendering_feat_depth = torch.cat([gt_depth, rendering_feat_depth], dim=1)
        rendering_depth = render_pkg["depth"]
        gt_depth = depth_raw
        depth_diff = torch.abs(rendering_depth - gt_depth) / gt_depth.max()
        depth_diff = torch.where(depth_diff > depth_threshold, 1., 0.)
        rendering_depth = rendering_depth / rendering_depth.max()
        gt_depth = gt_depth / gt_depth.max()
        rendering_depth_with_diff = torch.cat([gt_depth, rendering_depth, depth_diff], dim=1)

        ################################################################
        # interpolation = PointInterpolation(args, scene, pipeline, gaussians, tb_writer=None, depth_threshold=0.1, color_threshold=0.1, voxel_stride_portion=0.8, interpolate_interval=1, train_iterations=200)
        # interpolation()
        ################################################################
            
        name_list.append('{0:05d}'.format(idx) + ".png")
        # torchvision.utils.save_image(gt_show, os.path.join(save_dir, '{0:05d}'.format(idx) + "_gt.png"))
        if render_with_anchor:
            torchvision.utils.save_image(rendering_feat, os.path.join(save_dir, '{0:05d}'.format(idx) + "_feat.png"))
        torchvision.utils.save_image(rendering, os.path.join(save_dir, '{0:05d}'.format(idx) + "_render.png"))
        if render_with_anchor:
            torchvision.utils.save_image(rendering_feat_depth, os.path.join(save_dir, '{0:05d}'.format(idx) + "_feat_depth.png"))
            # torchvision.utils.save_image(rendering_feat_alpha, os.path.join(save_dir, '{0:05d}'.format(idx) + "_feat_alpha.png"))
        torchvision.utils.save_image(rendering_depth_with_diff, os.path.join(save_dir, '{0:05d}'.format(idx) + "_render_depth.png"))
        # torchvision.utils.save_image(rendering_alpha, os.path.join(save_dir, '{0:05d}'.format(idx) + "_render_alpha.png"))
        torch.cuda.empty_cache()

     
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
        
        freq = FrequencyModifier(dataset, scene, pipeline, None, None)
        
        show_gaussians(dataset, scene, freq, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)


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
