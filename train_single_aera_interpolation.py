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
import copy
import torch
import torchvision
import json
import wandb
import time
import random
import numpy as np
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, render_with_consistency_loss, prefilter_voxel_gsplat, render_gsplat, render_with_consistency_loss_gsplat, render_gsplat_xyzonly
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.distributed_utils import init_distributed_mode, dist, cleanup

from scene.interpolation import GaussianInterpolation
from scene.interpolation_point import PointInterpolation
from scene.freq_modifier import FrequencyModifier

torch.set_num_threads(32)


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)
    log_dir = pathlib.Path(__file__).parent.resolve()
    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    print('Backup Finished!')


def set_require_grad(model, is_require_grad):
    for param in model.parameters():
        # param.requires_grad = is_require_grad
        param = param.detach().requires_grad_(is_require_grad)


def replace_model(model_a, model_b):
    # replace model_a with model_b
    with torch.no_grad():
        for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
            param_a.data = param_b.data


def momentum_update(block_mlp, main_mlp=None, m=0.9):
    with torch.no_grad():
        for block_param, main_param in zip(block_mlp.parameters(), main_mlp.parameters()):
            main_param.data = m * main_param.data + (1 - m) * block_param.data


def sync_model_with_rank0(model):
    with torch.no_grad():
        for param in model.parameters():
            dist.broadcast(param.data, 0)

"""
**********************************************************************
**********************************************************************
**********************************************************************
*************************** Training script **************************
**********************************************************************
**********************************************************************
**********************************************************************
"""
def training(dataset, opt, pipe, dataset_name, saving_iterations, debug_from, wandb=None, logger=None, ply_path=None, testing_freq=1000, debug=True, render_with_gsplat=True, absgrad=False, train_with_scaledown=True, antialiasing=False):
    first_iter = 0
    # num_blocks = dataset.block_num
    num_blocks = 1
    num_gpus = dist.get_world_size()
    assert num_blocks % num_gpus == 0, "Number of blocks must be divisible by number of GPUs"
    num_blocks_per_gpu = num_blocks // num_gpus
    multi_block_per_gpu = num_blocks_per_gpu > 1
    rank = dist.get_rank()
    device = torch.device("cuda", rank)

    gaussians_list, scene_list, optimizer_state_list, block_id_list = [], [], [], []
    block_psnr_list, block_ssim_list = [], []
    block_psnr_momentum, block_ssim_momentum = [0 for _ in range(num_blocks_per_gpu)], [0 for _ in range(num_blocks_per_gpu)] 
    checkpoint_tmp_dir = dataset.checkpoint_tmp_dir
    if not os.path.exists(checkpoint_tmp_dir):
        os.makedirs(checkpoint_tmp_dir, exist_ok=True)

    # for block_id in range(rank * num_blocks_per_gpu, (rank + 1) * num_blocks_per_gpu):
    #     print(f"### Start initializing Block {block_id} on rank {rank}")

    block_id = 0
    print(f"### Start initializing Block {block_id} on rank {rank}")

    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, block_id=-1, load_test_scenes=True)
    # iteration = str(opt.iterations) + "_blockA_aera"
    # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, block_id=-1, load_test_scenes=True)

    # if not the first block, sync mlp data with previous block
    if multi_block_per_gpu and rank == 0 and block_id != rank * num_blocks_per_gpu:
        replace_model(gaussians.mlp_color, gaussians_list[0].mlp_color)
        replace_model(gaussians.mlp_cov, gaussians_list[0].mlp_cov)
        replace_model(gaussians.mlp_opacity, gaussians_list[0].mlp_opacity)

    ### sync mlp data with rank 0
    sync_model_with_rank0(gaussians.mlp_color)
    sync_model_with_rank0(gaussians.mlp_cov)
    sync_model_with_rank0(gaussians.mlp_opacity)

    gaussians.training_setup(opt)
    gaussians.train()

    gaussians_list.append(gaussians)
    scene_list.append(scene)
    block_id_list.append(block_id)
    optimizer_state_list.append(gaussians.optimizer.state_dict())

    block_psnr_list.append([])
    block_ssim_list.append([])

    if rank == 0:
        tb_writer = prepare_output_and_logger(dataset)
        iter_start = torch.cuda.Event(enable_timing = True)
        iter_end = torch.cuda.Event(enable_timing = True)
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    # Initialize Momentum Gaussian Decoder
    momentum_mlp_color = copy.deepcopy(gaussians.mlp_color).to(device)
    momentum_mlp_cov = copy.deepcopy(gaussians.mlp_cov).to(device)
    momentum_mlp_opacity = copy.deepcopy(gaussians.mlp_opacity).to(device)

    # No gradient required
    set_require_grad(momentum_mlp_color, False)
    set_require_grad(momentum_mlp_cov, False)
    set_require_grad(momentum_mlp_opacity, False)
    
    viewpoint_stack_list = [None for _ in range(num_blocks_per_gpu)]
    last_gaussians = None
    first_iter += 1
    dist.barrier()

    ######################################################################
    freq = FrequencyModifier(dataset, scene, opt, pipe, tb_writer)
    freq_topk = 1000
    if debug:
        freq_plot_dir = os.path.join(dataset.model_path, "freq_plot")
        os.makedirs(freq_plot_dir, exist_ok=True)
    ######################################################################
  
    iteration = first_iter
    while iteration <= opt.iterations:

        end_iter = iteration + opt.block_training_interval

        for idx in range(num_blocks_per_gpu):
            # move gaussians to GPU
            torch.cuda.empty_cache()
            gaussians = gaussians_list[idx]
            scene = scene_list[idx]
            block_id = block_id_list[idx]
            viewpoint_stack = viewpoint_stack_list[idx]

            if multi_block_per_gpu and iteration != first_iter:
                checkpoint = checkpoint_tmp_dir + "chkpnt_" + str(block_id) + ".pth"
                model_params = torch.load(checkpoint)
                gaussians.restore(model_params, opt)
                gaussians.train()
                torch.cuda.synchronize()
                dist.barrier()
                torch.cuda.empty_cache()

            if end_iter == opt.iterations and idx != 0:
                if multi_block_per_gpu:
                    replace_model(gaussians.mlp_color, last_gaussians.mlp_color)
                    replace_model(gaussians.mlp_cov, last_gaussians.mlp_cov)
                    replace_model(gaussians.mlp_opacity, last_gaussians.mlp_opacity)
                gaussians.freezen_mlp()

            for cur_iter in range(iteration, end_iter):
                if not gaussians.freeze_all_mlp:
                    # replace old mlp with the current one
                    if multi_block_per_gpu and cur_iter == iteration and last_gaussians is not None:
                        replace_model(gaussians.mlp_color, last_gaussians.mlp_color)
                        replace_model(gaussians.mlp_cov, last_gaussians.mlp_cov)
                        replace_model(gaussians.mlp_opacity, last_gaussians.mlp_opacity)

                    torch.cuda.synchronize()
                    dist.barrier()

                    # update Momentum Gaussian Decoder
                    if rank == 0:
                        momentum_update(gaussians.mlp_color, momentum_mlp_color, opt.momentum_coefficient)
                        momentum_update(gaussians.mlp_cov, momentum_mlp_cov, opt.momentum_coefficient)
                        momentum_update(gaussians.mlp_opacity, momentum_mlp_opacity, opt.momentum_coefficient)

                    torch.cuda.synchronize()
                    dist.barrier()

                    # sync main mlp with rank0
                    sync_model_with_rank0(momentum_mlp_color)
                    sync_model_with_rank0(momentum_mlp_cov)
                    sync_model_with_rank0(momentum_mlp_opacity)

                if rank == 0:
                    iter_start.record()

                gaussians.update_learning_rate(cur_iter)

                bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                
                # Pick a view randomly
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))


                # Render
                if (cur_iter - 1) == debug_from:
                    pipe.debug = True
                if render_with_gsplat:
                    voxel_visible_mask = prefilter_voxel_gsplat(viewpoint_cam, gaussians, pipe, background, absgrad=absgrad, antialias=antialiasing)
                else:
                    voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

                retain_grad = (cur_iter < opt.update_until and cur_iter >= 0)
                if gaussians.freeze_all_mlp:
                    if render_with_gsplat:
                        render_pkg = render_gsplat(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, absgrad=absgrad, antialias=antialiasing)
                    else:
                        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
                else:
                    if render_with_gsplat:
                        render_pkg = render_with_consistency_loss_gsplat(viewpoint_cam, gaussians, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, absgrad=absgrad,  antialias=antialiasing)
                    else:
                        render_pkg = render_with_consistency_loss(viewpoint_cam, gaussians, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
                
                image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

                if render_with_gsplat:
                    alpha = render_pkg["alpha"]
                    depth = render_pkg["depth"]

                gt_image = viewpoint_cam.original_image.cuda()
                gt_depth = viewpoint_cam.original_depth.cuda()
                gt_normal = viewpoint_cam.original_normal.cuda()
                
                Ll1 = l1_loss(image, gt_image)
                ssim_loss = (1.0 - ssim(image, gt_image))
                scaling_reg = scaling.prod(dim=1).mean()
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg
                ###############################################################
                ###############################################################
                # Ll1 = l1_loss(image, gt_image)
                # ssim_loss = (1.0 - ssim(image, gt_image))
                # scaling_mask = scaling.detach().clone().max(dim=1).values // dataset.voxel_size
                # scaling_mask = scaling_mask > 500
                # scaling_reg = scaling.prod(dim=1) * scaling_mask
                # scaling_reg = scaling_reg.mean()
                # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

                if render_with_gsplat:
                    # if train_with_scaledown:
                    #     # ratio = random.random()
                    #     # render_pkg_scaledown = render_gsplat_xyzonly(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=True,scaledown_ratio=ratio)
                    #     # render_pkg_scaledown = render_gsplat(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=True,scaledown_ratio=ratio)
                    #     depth = render_pkg_scaledown["depth"]
                    #     gt_depth = gt_depth * render_pkg_scaledown["alpha"].detach().clone()
                    Ll1_depth = l1_loss(depth, gt_depth) * (1 - opt.lambda_dssim)
                    # Ll1_depth = l1_loss(depth, gt_depth) * (1 - opt.lambda_dssim) * 0.1
                    loss += Ll1_depth
                ###############################################################
                ###############################################################
            
                if not gaussians.freeze_all_mlp:
                    # consistency loss
                    consistency_loss = render_pkg['consistency_loss']

                    # Reconstruction-guided block weighting
                    with torch.no_grad():
                        if cur_iter <= 1000:
                            recons_weight = torch.tensor(1.0)
                        else:
                            cur_recons_idx = len(block_psnr_list[idx]) - 1
                            cur_psnr = block_psnr_list[idx][cur_recons_idx]
                            cur_ssim = block_ssim_list[idx][cur_recons_idx]

                            momentum_psnr = block_psnr_momentum[idx] if block_psnr_momentum[idx] != 0 else cur_psnr
                            momentum_psnr = momentum_psnr * 0.9 + cur_psnr * 0.1
                            momentum_ssim = block_ssim_momentum[idx] if block_ssim_momentum[idx] != 0 else cur_ssim
                            momentum_ssim = momentum_ssim * 0.9 + cur_ssim * 0.1

                            block_psnr_momentum[idx] = momentum_psnr
                            block_ssim_momentum[idx] = momentum_ssim
                            
                            max_psnr = momentum_psnr
                            max_ssim = momentum_ssim

                            for block_idx in range(num_blocks_per_gpu):
                                max_psnr = max(max_psnr, block_psnr_momentum[block_idx])
                                max_ssim = max(max_ssim, block_ssim_momentum[block_idx])

                            max_psnr_all = torch.zeros(num_gpus, device=device)
                            max_ssim_all = torch.zeros(num_gpus, device=device)

                            max_psnr_all[rank] = max_psnr
                            max_ssim_all[rank] = max_ssim

                            torch.cuda.synchronize()
                            dist.barrier()
                            dist.all_reduce(max_psnr_all)
                            dist.all_reduce(max_ssim_all)

                            cur_max_psnr = max_psnr_all.max()
                            cur_max_ssim = max_ssim_all.max()

                            recons_weight = torch.tensor(2.0) - torch.exp(-((cur_max_psnr - momentum_psnr)**2 + (cur_max_ssim * 10 - momentum_ssim * 10)**2) / (2 * opt.adaptive_sigma * opt.adaptive_sigma))

                    loss = (loss + consistency_loss * opt.consistency_loss_weight) * recons_weight

                loss.backward()

                if not gaussians.freeze_all_mlp:
                    torch.cuda.synchronize()
                    dist.barrier()
                    with torch.no_grad():
                        # all reduce mlp grad
                        for param in gaussians.mlp_opacity.parameters():
                            torch.distributed.all_reduce(param.grad)
                            torch.cuda.synchronize()
                            dist.barrier()
                            param.grad = param.grad / num_gpus

                        for param in gaussians.mlp_color.parameters():
                            torch.distributed.all_reduce(param.grad)
                            torch.cuda.synchronize()
                            dist.barrier()
                            param.grad = param.grad / num_gpus

                        for param in gaussians.mlp_cov.parameters():
                            torch.distributed.all_reduce(param.grad)
                            torch.cuda.synchronize()
                            dist.barrier()
                            param.grad = param.grad / num_gpus

                if rank == 0:
                    iter_end.record()

                with torch.no_grad():
                    # Progress bar
                    if rank == 0:
                        ema_loss_for_log = 0.99 * loss.item() + 0.01 * ema_loss_for_log

                        if cur_iter % 10 == 0:
                            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                            if 10 % num_blocks_per_gpu == 0:
                                progress_bar.update(10 // num_blocks_per_gpu)
                            else:
                                if idx != 0:
                                    progress_bar.update(10 // num_blocks_per_gpu)
                                else:
                                    remain = 10 - (10 // num_blocks_per_gpu) * (num_blocks_per_gpu - 1)
                                    progress_bar.update(remain)
                        if cur_iter == opt.iterations:
                            progress_bar.close()

                    # Log and validation
                    if rank == 0:
                        training_report(tb_writer, dataset_name, cur_iter, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), scene, block_id, render, (pipe, background), wandb, logger, testing_freq=testing_freq, block_psnr_list=block_psnr_list, block_ssim_list=block_ssim_list, block_idx=idx)
                    else:
                        training_report(None, dataset_name, cur_iter, Ll1, loss, l1_loss, None, scene, block_id, render, (pipe, background), testing_freq=testing_freq, block_psnr_list=block_psnr_list, block_ssim_list=block_ssim_list, block_idx=idx)

                    # Save 
                    if (cur_iter in saving_iterations):
                        time.sleep(30 * rank)
                        if rank == 0:
                            logger.info("\n[ITER {}] Block_{} Saving Gaussians".format(cur_iter, block_id))
                        else:
                            print("\n[ITER {}] Block_{} Saving Gaussians".format(cur_iter, block_id))
                        scene.save(cur_iter, block_id=block_id)
                    
                    #######################################################
                    #######################################################
                    # prune
                    if cur_iter < opt.update_until and cur_iter > opt.start_stat:
                        # add statis
                        gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, render_with_gsplat, absgrad=absgrad)
                        # densification
                        if cur_iter > opt.update_from and cur_iter % opt.update_interval == 0:
                            gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity, render_with_gsplat=render_with_gsplat)
                        
                    # if cur_iter < opt.update_until and cur_iter > opt.start_stat:
                    #     gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, render_with_gsplat)
                    #     if cur_iter > opt.update_from and cur_iter % opt.update_interval == 0:
                    #         gaussians.pure_prune_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, min_opacity=opt.min_opacity)
                    #######################################################
                    #######################################################
                    elif cur_iter == opt.update_until:
                        print("### Stop densification.")
                        gaussians.opacity_accum = None
                        gaussians.offset_gradient_accum = None
                        gaussians.offset_denom = None
                        torch.cuda.empty_cache()
                            
                    # Optimizer step
                    if cur_iter < opt.iterations:
                        gaussians.optimizer.step()
                        gaussians.optimizer.zero_grad(set_to_none = True)

                ############################################################
                ############################################################
                ### NOTE: frequency modifier ###
                # # if cur_iter < opt.update_from:
                # #     freq_topk = 3000
                # if cur_iter > opt.update_from and cur_iter < opt.update_until and cur_iter % 1000 == 0:
                #     with torch.no_grad():
                #         frequency = freq.anchor_freq(gaussians, mode="anchor_scale")
                #         frequency_filtered = freq.filter_freq_direct(frequency,  k=freq_topk)
                #         freq.plot_freq_compare(frequency, frequency_filtered, os.path.join(freq_plot_dir, '{0:05d}'.format(cur_iter) + "_freq_compare.png"))
                #         gaussians_prune_mask = ~frequency_filtered["freq_mask"]
                #         gaussians.prune_anchor_withmask(gaussians_prune_mask)
                #     freq_topk = freq_topk + 1000
                ### NOTE: frequency modifier ###
                # if cur_iter > opt.update_from and cur_iter < opt.update_until and cur_iter % 1000 == 0:
                #     with torch.no_grad():
                #         anchorscale_freqscale = 1.
                #         anchoroffset_freqscale = 2.
                #         freq_scale = freq.anchor_freq_smooth(gaussians, freq_scale=anchorscale_freqscale, mode="anchor_scale")
                #         freq_offset = freq.anchor_freq_smooth(gaussians, freq_scale=anchoroffset_freqscale, mode="anchor_offset")
                #         freq_scale_filtered = freq.filter_smooth(freq_scale, freq_scale=anchorscale_freqscale)
                #         freq_offset_filtered = freq.filter_smooth(freq_offset, freq_scale=anchoroffset_freqscale)
                #         freq.plot_freq_compare(freq_scale, freq_scale_filtered, os.path.join(freq_plot_dir, '{0:05d}'.format(cur_iter) + "_freq_scale.png"))
                #         freq.plot_freq_compare(freq_offset, freq_offset_filtered, os.path.join(freq_plot_dir, '{0:05d}'.format(cur_iter) + "_freq_offset.png"))
                #         ### NOTE: prune oversized ###
                #         # freq_mask = torch.logical_and(freq_scale_filtered["freq_mask"], freq_offset_filtered["freq_mask"])
                #         # gaussians_prune_mask = ~freq_mask
                #         # gaussians.prune_anchor_withmask(gaussians_prune_mask)
                #         ### NOTE: scale oversized ###
                #         gaussians.scale_anchor_withmaskratio(freq_scale_filtered['freq_mask'], freq_scale_filtered['freq_scale_ratio'], mode="anchor_scale")
                #         gaussians.scale_anchor_withmaskratio(freq_offset_filtered['freq_mask'], freq_offset_filtered['freq_scale_ratio'], mode="anchor_offset")
                #     freq_topk = freq_topk + 1000
                ### NOTE: interpolation ###
                if cur_iter > opt.update_from and cur_iter < opt.update_until and cur_iter % 1000 == 0:
                    torch.cuda.empty_cache()
                    # gaussian_interpolate = GaussianInterpolation(
                    #     dataset, scene, opt, pipe, gaussians, tb_writer,
                    #     depth_threshold=0.1, color_threshold=0.1, 
                    #     voxel_stride_portion=0.8, interpolate_interval=40, 
                    #     train_iterations=200
                    #     ) # interpolate_interval=10
                    gaussian_interpolate = PointInterpolation(
                        dataset, scene, opt, pipe, gaussians, tb_writer,
                        depth_threshold=0.1, color_threshold=0.1, 
                        voxel_stride_portion=0.8, interpolate_interval=1, 
                        train_iterations=0
                        # train_iterations=200
                        )
                    iterpolated = gaussian_interpolate(resolution_scaling=1, scaledown_ratio=1)
                    torch.cuda.empty_cache()
                ############################################################
                ############################################################

            viewpoint_stack_list[idx] = viewpoint_stack
            last_gaussians = gaussians

            if multi_block_per_gpu:
                gaussians.eval()
                torch.save(gaussians.capture(), checkpoint_tmp_dir + "chkpnt_" + str(block_id) + ".pth")
                del gaussians._anchor
                del gaussians._anchor_feat
                del gaussians._offset
                del gaussians._scaling
                del gaussians._rotation
                del gaussians._opacity
                del gaussians.max_radii2D
                del gaussians.optimizer
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                del gaussians.anchor_demon
                del gaussians.spatial_lr_scale
                torch.cuda.synchronize()
                dist.barrier()

            torch.cuda.empty_cache()

        iteration += opt.block_training_interval
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, elapsed, scene : Scene, block_id, renderFunc, renderArgs, wandb=None, logger=None, testing_freq=1000, block_psnr_list=None, block_ssim_list=None, block_idx=None):
    if rank == 0 and tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)
    if rank == 0 and wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, })

    if iteration % testing_freq == 0:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, len(scene.getTrainCameras()), 8)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                
                if rank == 0 and wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if rank == 0 and tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_freq:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])

                block_psnr_list[block_idx].append(psnr_test)
                block_ssim_list[block_idx].append(ssim_test)

                if rank == 0:      
                    logger.info("\n[ITER {}] Evaluating Block{} {}: L1 {} PSNR {} SSIM {}".format(iteration, block_id, config['name'], l1_test, psnr_test, ssim_test))
                else:
                    #TODO add other ranks to logger
                    print("\n[ITER {}] Evaluating Block{} {}: L1 {} PSNR {} SSIM {}".format(iteration, block_id, config['name'], l1_test, psnr_test, ssim_test))

                if rank == 0 and tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if rank == 0 and wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if rank == 0 and tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()
    

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # "--pretrained_path", "outputs/pretrianed/matrixcity", 
    torch.cuda.reset_peak_memory_stats()

    # Distributed training
    init_distributed_mode(args=args)
    rank = args.rank
    device = torch.device(args.device)

    dataset = args.source_path.split('/')[-1]

    if rank == 0:
        # enable logging
        model_path = args.model_path
        os.makedirs(model_path, exist_ok=True)

        logger = get_logger(model_path)
        logger.info(f'args: {args}')

        try:
            saveRuntimeCode(os.path.join(args.model_path, 'backup'))
        except:
            logger.info(f'save code failed~')
    
        exp_name = args.model_path.split('/')[-2]
    
        if args.use_wandb:
            wandb.login()
            run = wandb.init(
                # Set the project where this run will be logged
                project=f"Momentum-GS-{dataset}",
                name=exp_name,
                # Track hyperparameters and run metadata
                settings=wandb.Settings(start_method="fork"),
                config=vars(args)
            )
        else:
            wandb = None
    
        logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    if rank == 0:
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset, args.save_iterations, args.debug_from, wandb, logger)
    else:
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset, args.save_iterations, args.debug_from)

    max_memory = torch.cuda.max_memory_allocated()
    print(f"[rank {rank}] max vram={max_memory / (1024**2)} MB\n")

    if rank == 0:
        logger.info("\nTraining complete.")

    cleanup()
