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
# append parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from utils.loss_utils import l1_loss, ssim, l2_loss
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

import dataio, utility, training, loss_functions, modules, sdf_meshing
from torch.utils.data import DataLoader
import open3d as o3d
from mesh_utils import post_process_mesh, to_cam_open3d

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
def training(dataset, opt, pipe, dataset_name, saving_iterations, debug_from, wandb=None, logger=None, ply_path=None, testing_freq=1000, debug=True, render_with_gsplat=False, train_with_depth=True, absgrad=True, train_with_scaledown=True, antialiasing=False):
    first_iter = 0
    # num_blocks = dataset.block_num
    num_blocks = 1
    num_gpus = dist.get_world_size()
    assert num_blocks % num_gpus == 0, "Number of blocks must be divisible by number of GPUs"
    num_blocks_per_gpu = num_blocks // num_gpus
    multi_block_per_gpu = num_blocks_per_gpu > 1
    rank = dist.get_rank()
    device = torch.device("cuda", rank)

    checkpoint_tmp_dir = dataset.checkpoint_tmp_dir
    if not os.path.exists(checkpoint_tmp_dir):
        os.makedirs(checkpoint_tmp_dir, exist_ok=True)

    # for block_id in range(rank * num_blocks_per_gpu, (rank + 1) * num_blocks_per_gpu):
    #     print(f"### Start initializing Block {block_id} on rank {rank}")

    block_id = 0
    print(f"### Start initializing Block {block_id} on rank {rank}")

    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False, block_id=-1, load_test_scenes=False)
    # iteration = str(opt.iterations) + "_blockA_aera"
    # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, block_id=-1, load_test_scenes=True)

    gaussians.training_setup(opt)
    gaussians.train()

    batch_size = 10000

    tb_writer = prepare_output_and_logger(dataset)
    iter_start = torch.cuda.Event(enable_timing = True)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    debug_dir = os.path.join(current_dir, "tmp")
    os.makedirs(debug_dir, exist_ok=True)
    viewpoint_processor = dataio.ViewPointPostProcess(points=gaussians.get_anchor.clone().detach(), batch_size=batch_size, bound_extend_ratio=0.01)

    """
    ****************************************************************
    ************** NOTE: single scene ******************************
    ****************************************************************
    """
    # single_scene = scene.getTrainCameras().copy()[0]
    # single_scene_image = single_scene.original_image.cuda()
    # single_scene_depth = single_scene.original_depth.cuda()
    # single_scene_normal = single_scene.original_normal.cuda()

    # points, mask = viewpoint_processor.depth2points(single_scene_depth, single_scene)
    # normals = single_scene_normal.permute(1, 2, 0).view(-1, 3)[mask]
    # colors = single_scene_image.permute(1, 2, 0).view(-1, 3)[mask]
    # points, normals = viewpoint_processor.transform2w(points, normals, single_scene)
    # pnts = torch.cat([points, normals, colors], dim=1)
    # del viewpoint_processor
    # sdf_dataset = dataio.PointCloudPlus(pnts.cpu().numpy(), on_surface_points=batch_size)
    # dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)

    # #########################################################################
    # ############################# NOTE: build mesh ##########################
    # #########################################################################
    # # voxel_size = 0.001
    # # sdf_trunc = voxel_size * 5
    # # depth_trunc = 3
    # # volume = o3d.pipelines.integration.ScalableTSDFVolume(
    # #     voxel_length=voxel_size,
    # #     sdf_trunc=sdf_trunc,
    # #     color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    # # )
    # # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    # #     o3d.geometry.Image(np.asarray(np.clip(single_scene_image.permute(1, 2, 0).cpu().numpy(), 0, 1) * 255, order="C", dtype=np.uint8)),
    # #     o3d.geometry.Image(np.asarray(single_scene_depth.permute(1, 2, 0).cpu().numpy(), order="C")),
    # #     depth_trunc=depth_trunc, convert_rgb_to_intensity=False,
    # #     depth_scale=1.0,
    # # )

    # # camera = to_cam_open3d([single_scene])[0]

    # # volume.integrate(rgbd, intrinsic=camera.intrinsic, extrinsic=camera.extrinsic)
    # # mesh = volume.extract_triangle_mesh()
    # # name = "mesh_debug.ply"
    # # o3d.io.write_triangle_mesh(os.path.join(debug_dir, name), mesh)
    # # print("mesh saved at {}".format(os.path.join(debug_dir, name)))
    # # # post-process the mesh and save, saving the largest N clusters
    # # mesh_post = post_process_mesh(mesh, cluster_to_keep=10000)
    # # o3d.io.write_triangle_mesh(os.path.join(debug_dir, name.replace('.ply', '_post.ply')), mesh_post)
    # # print("mesh post processed saved at {}".format(os.path.join(debug_dir, name.replace('.ply', '_post.ply'))))
    # #########################################################################
    # #########################################################################
    # #########################################################################


    # # model = modules.SingleBVPNet(type="sine", in_features=3)
    # model = modules.SimpleNet(type="sine", in_features=3, out_features=4)
    # model.cuda()
    # # loss_fn = loss_functions.sdf
    # # summary_fn = utility.write_sdf_summary
    # loss_fn = loss_functions.sdf_extent
    # summary_fn = utility.write_sdf_summary_extent
    # optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

    # epochs = 1000
    # total_step = 0
    # with tqdm(total=len(dataloader) * epochs) as pbar:
    #     for epoch in range(epochs):
    #         for step, (model_input, gt) in enumerate(dataloader):
    #             model_input = {key: value.cuda() for key, value in model_input.items()}
    #             gt = {key: value.cuda() for key, value in gt.items()}
    #             model_output = model(model_input)
    #             losses = loss_fn(model_output, gt)
    #             train_l = 0.
    #             for l_name, l in losses.items():
    #                 single_l = l.mean()
    #                 tb_writer.add_scalar(l_name, single_l, total_step)
    #                 train_l += single_l
    #             tb_writer.add_scalar("total_train_l", train_l, total_step)
    #             summary_fn(model, model_input, gt, model_output, tb_writer, total_step)
    #             optim.zero_grad()
    #             train_l.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    #             optim.step()
    #             pbar.update(1)
    #             total_step += 1
    # torch.save(model.state_dict(), os.path.join(dataset.model_path, "model.pth"))
    """
    ****************************************************************
    ****************************************************************
    ****************************************************************
    """


    total_iterations = 10000
    batch_size = 10000
    out_features = 4

    progress_bar = tqdm(range(first_iter, total_iterations), desc="Training progress")
    # model = modules.SimpleNet(type="sine", in_features=3, out_features=4)
    model = modules.SimpleNet(type="sine", mode="fft", in_features=3, out_features=out_features, fft_mode="simple")
    model.cuda()
    if out_features > 1:
        loss_fn = loss_functions.sdf_extent
        summary_fn = utility.write_sdf_summary_extent
    else:
        loss_fn = loss_functions.sdf
        summary_fn = utility.write_sdf_summary
    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())


    single_scene = scene.getTrainCameras().copy()[0]
    single_scene_depth = single_scene.original_depth.cuda()
    single_scene_normal = single_scene.original_normal.cuda()
    points, mask = viewpoint_processor.depth2points(single_scene_depth, single_scene)
    normals = single_scene_normal.permute(1, 2, 0).view(-1, 3)[mask]
    points, normals = viewpoint_processor.transform2w(points, normals, single_scene)
    del viewpoint_processor
    viewpoint_processor = dataio.ViewPointPostProcess(points=points.clone().detach(), batch_size=batch_size, bound_extend_ratio=0.01)

    iteration = first_iter
    while iteration <= total_iterations:
        progress_bar.update(1)
        block_id = -1
        viewpoint_stack = None
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # # Pick a view randomly
        # if not viewpoint_stack:
        #     viewpoint_stack = scene.getTrainCameras().copy()
        # viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # single scene
        viewpoint_cam = single_scene
        gt_image = viewpoint_cam.original_image.cuda()
        gt_depth = viewpoint_cam.original_depth.cuda()
        gt_normal = viewpoint_cam.original_normal.cuda()
        gt_opacities = torch.ones_like(gt_depth)
        model_input, gt = viewpoint_processor.propogate_data(gt_depth, gt_normal, gt_image, gt_opacities, viewpoint_cam)
        model_input = {key: value.cuda() for key, value in model_input.items()}
        gt = {key: value.cuda() for key, value in gt.items()}
        model_output = model(model_input)
        losses = loss_fn(model_output, gt)
        train_l = 0.
        for l_name, l in losses.items():
            single_l = l.mean()
            tb_writer.add_scalar(l_name, single_l, iteration)
            train_l += single_l
        tb_writer.add_scalar("total_train_l", train_l, iteration)
        summary_fn(model, model_input, gt, model_output, tb_writer, iteration)
        optim.zero_grad()
        train_l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optim.step()
        iteration += 1
        torch.cuda.empty_cache()
    torch.save(model.state_dict(), os.path.join(dataset.model_path, "model.pth"))
    
    ################################################################
    ################ NOTE: test sdf with mesh ######################
    ################################################################
    class SDFDecoder(torch.nn.Module):
        def __init__(self, model_path):
            super().__init__()
            # self.model = modules.SingleBVPNet(type="sine", in_features=3)
            # self.model = modules.SimpleNet(type="sine", in_features=3, out_features=4)
            self.model = modules.SimpleNet(type="sine", mode="fft", in_features=3, out_features=out_features, fft_mode="simple")
            self.model.load_state_dict(torch.load(model_path))
            self.model.cuda()

        def forward(self, coords):
            model_in = {'coords': coords}
            return self.model(model_in)['model_out']
    
    sdf_decoder = SDFDecoder(os.path.join(dataset.model_path, "model.pth"))
    sdf_meshing.create_mesh_colors(sdf_decoder, os.path.join(dataset.model_path, f"mesh__"), N=512, lower_bound=viewpoint_processor.lower_bound, upper_bound=viewpoint_processor.upper_bound)
    ################################################################
    ################################################################
    ################################################################
            

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    # if os.path.exists(args.model_path):
    #     shutil.rmtree(args.model_path)
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
