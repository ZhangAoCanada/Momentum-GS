import os, sys
import random
import numpy as np
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render, render_with_consistency_loss, prefilter_voxel_gsplat, render_gsplat, render_with_consistency_loss_gsplat, render_gsplat_xyzonly
from scene.gaussian_interpolation_model import GaussianInterpolationModel
import shutil
from tqdm import tqdm


class PointInterpolation(nn.Module):
    def __init__(self, args, scene, opt, pipe, trained_gaussians, tb_writer=None, depth_threshold=0.1, color_threshold=0.1, voxel_stride_portion=0.5, interpolate_interval=10, train_iterations=500): # interval 10 iter 500
        super().__init__()
        # setups for Gaussians
        self.args = args
        self.scene = scene
        self.opt = opt
        self.pipe = pipe
        self.gaussians = GaussianInterpolationModel(args.feat_dim, args.n_offsets, args.voxel_size, args.update_depth, args.update_init_factor, args.update_hierachy_factor, args.use_feat_bank, args.appearance_dim, args.ratio, args.add_opacity_dist, args.add_cov_dist, args.add_color_dist)
        self.trained_gaussians = trained_gaussians
        self.iterations = train_iterations
        self.tb_writer = tb_writer
        self.interpolate_interval = interpolate_interval 

        # setups for interpolation 
        self.voxel_size = args.voxel_size
        self.depth_threshold = depth_threshold
        self.color_threshold = color_threshold
        self.voxel_stride_portion = voxel_stride_portion
    
    def compute_voxel_stride(self):
        trained_voxel_stride = torch.sort(self.trained_gaussians.get_scaling[:, :3].max(dim=1).values / self.voxel_size).values
        # trained_voxel_stride = torch.sort(self.trained_gaussians.get_scaling[:, 3:].max(dim=1).values / self.voxel_size).values
        self.voxel_stride = torch.ceil(trained_voxel_stride[:int(len(trained_voxel_stride) * self.voxel_stride_portion)].mean()).cpu().item()

    def voxelize_distance(self, data=None, stride=None):
        if stride is None:
            voxel_size = self.args.voxel_size * self.voxel_stride
        else:
            # voxel_size = self.args.voxel_size * stride * self.voxel_stride
            voxel_size = self.args.voxel_size * stride
        data = torch.unique(torch.round(data/voxel_size)*voxel_size, dim=0)
        return data
    
    def gaussians_setup(self, points, voxel_visible_mask):
        self.gaussians.fix_trained(self.trained_gaussians, voxel_visible_mask)
        self.gaussians.create_from_points(points, self.scene.cameras_extent, self.voxel_stride)
        self.gaussians.training_setup(self.opt)
        self.gaussians.eval()
        return
    
    def depth_mask(self, rendered_depth, gt_depth):
        # depth_diff = torch.abs(rendered_depth - gt_depth) / gt_depth.max()
        depth_diff = torch.abs(rendered_depth - gt_depth)
        mask = depth_diff
        ### NOTE: no too far points ###
        farpoint_mask = gt_depth > 3
        mask = torch.where(farpoint_mask, 0., mask)
        ###############################
        mask = torch.where(mask > self.depth_threshold, 1., 0.).bool()
        return mask
    
    def color_mask(self, rendered_image, gt_image):
        color_diff = torch.abs(rendered_image - gt_image).max(dim=0).values
        mask = torch.where(color_diff > self.color_threshold, 1., 0.)
        return mask
    
    def update_gaussians(self,):
        self.trained_gaussians.update_anchor(self.gaussians)
        self.gaussians.clean_attributes()
    
    def upsample_tensor(self, tensor, scale):
        upsampled = torch.nn.functional.interpolate(tensor.unsqueeze(0), scale_factor=scale, mode='nearest-exact').squeeze(0)
        return upsampled

    def depth2viewpoints(self, depth_map, viewpoint_camera, resolution_scaling_factor=1.):
        # viewpoint attributes
        fovx = viewpoint_camera.FoVx
        fovy = viewpoint_camera.FoVy
        W = int(viewpoint_camera.image_width * resolution_scaling_factor)
        H = int(viewpoint_camera.image_height * resolution_scaling_factor)
        fx = W / (2 * math.tan(fovx / 2))
        fy = H / (2 * math.tan(fovy / 2))
        K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda")
        w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
        c2w = torch.inverse(w2c)

        assert depth_map.shape == (1, H, W)
        # transfer depth_map to (x, y, z) in viewspace
        x = torch.arange(W, device="cuda").repeat(H, 1)
        y = torch.arange(H, device="cuda").repeat(W, 1).transpose(0, 1)
        z = depth_map[0].clone().cuda()
        x = x.reshape(-1)
        y = y.reshape(-1)
        z = z.reshape(-1)
        points = torch.stack([x, y, z], dim=1)
        points[:, 0] = (points[:, 0] - W / 2) * z / fx
        points[:, 1] = (points[:, 1] - H / 2) * z / fy
        mask = points[:, 2] > 0
        points = points[mask]

        new_z = z[mask]
        point_voxel_stride = torch.round(new_z / fx / self.args.voxel_size) + 1
        assert point_voxel_stride.min() > 0
        point_voxel_stride = point_voxel_stride.unsqueeze(1).cuda()

        points = self.voxelize_distance(points, stride=point_voxel_stride)
        
        # transfer points to world space
        points = torch.cat([points, torch.ones(points.shape[0], 1, device="cuda")], dim=1)
        points = torch.matmul(c2w, points.t()).t()
        points = points[:, :3]
        return points

    def forward(self, debug=True, resolution_scaling=1, scaledown_ratio=1.):
        bg_color = [1, 1, 1] if self.args.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        viewpoint_stack = self.scene.getTrainCameras().copy()

        if scaledown_ratio > 0. and scaledown_ratio < 1.:
            self.trained_gaussians.scale_down(scaledown_ratio)

        if debug:
            debug_dir = "experiments/debug" if debug else None
            if debug_dir:
                if os.path.exists(debug_dir):
                    shutil.rmtree(debug_dir)
                os.makedirs(debug_dir, exist_ok=True)

        num_frames = len(viewpoint_stack) // self.interpolate_interval
        num_frames = num_frames if len(viewpoint_stack) % self.interpolate_interval == 0 else num_frames + 1
        start_frame = randint(0, self.interpolate_interval - 1)
        resolution_scaleup = resolution_scaling > 1.

        all_points = []
        self.compute_voxel_stride()
        for fid in range(num_frames):
            viewpoint_id = fid * self.interpolate_interval + start_frame
            viewpoint_id = min(viewpoint_id, len(viewpoint_stack) - 1)
            viewpoint_cam = viewpoint_stack[viewpoint_id]

            # render the trained gaussians
            with torch.no_grad():
                voxel_visible_mask_trained = prefilter_voxel_gsplat(viewpoint_cam, self.trained_gaussians, self.pipe, background)
                render_pkg_pretrained = render_gsplat(viewpoint_cam, self.trained_gaussians, self.pipe, background, visible_mask=voxel_visible_mask_trained, retain_grad=False, absgrad=True, resolution_scaling_factor=resolution_scaling)
                rendered_image, rendered_depth = render_pkg_pretrained["render"], render_pkg_pretrained["depth"]
                voxel_visible_mask_trained = voxel_visible_mask_trained.detach()
                rendered_image = rendered_image.detach()
                rendered_depth = rendered_depth.detach()

            gt_image = viewpoint_cam.original_image.cuda()
            gt_depth = viewpoint_cam.original_depth.cuda()
            gt_normal = viewpoint_cam.original_normal.cuda()
            if resolution_scaleup:
                gt_image = self.upsample_tensor(gt_image, resolution_scaling)
                gt_depth = self.upsample_tensor(gt_depth, resolution_scaling)
                gt_normal = self.upsample_tensor(gt_normal, resolution_scaling)

            mask = self.depth_mask(rendered_depth, gt_depth)
            color_mask = self.color_mask(rendered_image, gt_image)
            mask = torch.logical_or(mask, color_mask)
            gt_depth_masked = gt_depth * mask
            points = self.depth2viewpoints(gt_depth_masked, viewpoint_cam, resolution_scaling_factor=resolution_scaling)
            assert len(points.shape) == 2 and points.shape[1] == 3
            all_points.append(points)
            torch.cuda.empty_cache()
        
        if len(all_points) == 0:
            print("[DEBUG] No valid points for interpolation")
            return
        points = torch.cat(all_points, dim=0)
        points = self.voxelize_distance(points)

        self.gaussians_setup(points, voxel_visible_mask_trained)

        # train self.gaussians until converge
        for iteration in range(self.iterations):
            self.gaussians.update_learning_rate(iteration) # TODO: change config

            voxel_visible_mask = prefilter_voxel_gsplat(viewpoint_cam, self.gaussians, self.pipe, background)
            # retain_grad = (iteration < self.opt.update_until and iteration >= 0)
            retain_grad = True
            assert self.gaussians.freeze_all_mlp
            render_pkg = render_gsplat(viewpoint_cam, self.gaussians, self.pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, absgrad=False, interpolation=True, resolution_scaling_factor=resolution_scaling, override_training=True)
            
            image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

            alpha = render_pkg["alpha"]
            depth = render_pkg["depth"]

            Ll1 = l1_loss(image, gt_image)
            ssim_loss = (1.0 - ssim(image, gt_image))
            scaling_reg = scaling.prod(dim=1).mean()
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

            Ll1_depth = l1_loss(depth, gt_depth) * (1 - self.opt.lambda_dssim) * 0.1
            loss += Ll1_depth
            
            loss.backward()
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            count = fid * self.iterations + iteration
            if self.tb_writer:
                self.tb_writer.add_scalar(f'interpolation/l1_loss', Ll1, count)
                self.tb_writer.add_scalar(f'interpolation/depth_l1_loss', Ll1_depth, count)
                self.tb_writer.add_scalar(f'interpolation/total_loss', loss, count)

            if debug and iteration == self.iterations - 1:
                rgbs = torch.cat([gt_image, image, torch.abs(gt_image - image)], dim=1)
                depths = torch.cat([gt_depth, depth, torch.abs(gt_depth - depth)], dim=1)
                depths = torch.clamp(depths / gt_depth.max(), 0., 1.).repeat(3, 1, 1)
                debug_image = torch.cat([rgbs, depths], dim=2)
                if self.tb_writer:
                    self.tb_writer.add_images(f'interpolation/debug_image_{fid}', debug_image.unsqueeze(0), iteration)
                else:
                    torchvision.utils.save_image(debug_image, os.path.join(debug_dir, f"debug_image_{fid}_{iteration}.png"))
            iteration += 1

        print(f"[DEBUG] anchors: {self.trained_gaussians.get_anchor.shape[0]}, interpolation: {self.gaussians.get_anchor.shape[0]}, stride: {self.voxel_stride}")
        self.update_gaussians()
        torch.cuda.empty_cache()
        return