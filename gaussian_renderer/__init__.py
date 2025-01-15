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
import torch
import einops
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel

import gsplat


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False, interpolation=False, scaledown_ratio=None):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask] if not interpolation else pc.get_anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask] if not interpolation else pc.get_offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    #############################################################
    if scaledown_ratio is not None and scaledown_ratio > 0.0 and scaledown_ratio < 1.0:
        min_scale = pc.voxel_size * 1.0
        grid_scaling[:, 3:] = grid_scaling[:, 3:] * scaledown_ratio
        grid_scaling[:, 3:] = torch.where(grid_scaling[:, 3:] < min_scale, min_scale, grid_scaling[:, 3:])
    #############################################################

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7])
    
    # offsets
    offsets = grid_offsets.view([-1, 3])
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot
    

def generate_neural_gaussians_and_momentum_gaussians(viewpoint_camera, pc : GaussianModel, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, visible_mask=None, is_training=False, interpolation=False, scaledown_ratio=None):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask] if not interpolation else pc.get_anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask] if not interpolation else pc.get_offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    #############################################################
    if scaledown_ratio is not None and scaledown_ratio > 0.0 and scaledown_ratio < 1.0:
        min_scale = pc.voxel_size * 1.0
        grid_scaling[:, 3:] = grid_scaling[:, 3:] * scaledown_ratio
        grid_scaling[:, 3:] = torch.where(grid_scaling[:, 3:] < min_scale, min_scale, grid_scaling[:, 3:])
    #############################################################

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]

    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
        neural_opacity_main = momentum_mlp_opacity(cat_local_view)
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)
        neural_opacity_main = momentum_mlp_opacity(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    # select opacity 
    opacity = neural_opacity[mask]

    # main mlp opacity mask generation
    neural_opacity_main = neural_opacity_main.reshape([-1, 1])

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            color_main = momentum_mlp_color(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
            color_main = momentum_mlp_color(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
            color_main = momentum_mlp_color(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
            color_main = momentum_mlp_color(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])
    color_main = color_main.reshape([anchor.shape[0]*pc.n_offsets, 3])

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
        scale_rot_main = momentum_mlp_cov(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
        scale_rot_main = momentum_mlp_cov(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7])
    scale_rot_main = scale_rot_main.reshape([anchor.shape[0]*pc.n_offsets, 7])
    
    # offsets
    offsets = grid_offsets.view([-1, 3])
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, color_main, scale_rot, scale_rot_main, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, color_main, scale_rot, scale_rot_main, offsets = masked.split([6, 3, 3, 3, 7, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    scaling_main = scaling_repeat[:,3:] * torch.sigmoid(scale_rot_main[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])
    rot_main = torch.nn.functional.normalize(scale_rot_main[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, color_main, opacity, scaling, scaling_main, rot, rot_main, neural_opacity, neural_opacity_main, mask
    else:
        return xyz, color, opacity, scaling, rot
    

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, xyz=None, color=None, opacity=None, scaling=None, rot=None, resolution_scaling_factor=1.0, interpolation=False, scaledown_ratio=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training

    if xyz is None:
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    ############### NOTE: for rendering depth map #############
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, 0:1])], dim=1)
    xyz2c = torch.matmul(w2c, xyz_h.transpose(0, 1)).transpose(0, 1)
    z2c = xyz2c[:, 2]
    color = torch.cat([color, z2c.unsqueeze(1)], dim=1)
    bg_color = torch.cat([bg_color, torch.tensor(bg_color[0], device=bg_color.device).unsqueeze(0)], dim=0)
    ###########################################################

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height * resolution_scaling_factor),
        image_width=int(viewpoint_camera.image_width * resolution_scaling_factor),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    ############### NOTE: for rendering depth map #############
    rendered_depth = rendered_image[3:4, ...]
    rendered_image = rendered_image[:3, ...]
    ###########################################################
    
    if is_training:
        return {"render": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "xyz": xyz.detach().clone(),
                }
    else:
        return {"render": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "xyz": xyz.detach().clone(),
                }


def render_with_consistency_loss(viewpoint_camera, pc : GaussianModel, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, xyz=None, color=None, opacity=None, scaling=None, rot=None, resolution_scaling_factor=1.0, interpolation=False, scaledown_ratio=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    consistency_loss, color_loss, rot_loss, scaling_loss, opacity_loss = 0, 0, 0, 0, 0

    if xyz is None:
        if is_training:
            xyz, color, color_main, opacity, scaling, scaling_main, rot, rot_main, neural_opacity, neural_opacity_main, mask = generate_neural_gaussians_and_momentum_gaussians(viewpoint_camera, pc, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

            color_loss = torch.nn.functional.mse_loss(color, color_main)
            rot_loss = torch.nn.functional.mse_loss(rot, rot_main)
            scaling_loss = torch.nn.functional.mse_loss(scaling, scaling_main)
            opacity_loss = torch.nn.functional.mse_loss(neural_opacity, neural_opacity_main)

            consistency_loss = color_loss + rot_loss + scaling_loss + opacity_loss
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0 # [672276, 3]
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    ############### NOTE: for rendering depth map #############
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, 0:1])], dim=1)
    xyz2c = torch.matmul(w2c, xyz_h.transpose(0, 1)).transpose(0, 1)
    z2c = xyz2c[:, 2]
    color = torch.cat([color, z2c.unsqueeze(1)], dim=1)
    bg_color = torch.cat([bg_color, torch.tensor(bg_color[0], device=bg_color.device).unsqueeze(0)], dim=0)
    ###########################################################

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height * resolution_scaling_factor),
        image_width=int(viewpoint_camera.image_width * resolution_scaling_factor),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    ############### NOTE: for rendering depth map #############
    rendered_depth = rendered_image[3:4, ...]
    rendered_image = rendered_image[:3, ...]
    ###########################################################

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "consistency_loss": consistency_loss,
                "color_loss": color_loss,
                "rot_loss": rot_loss,
                "scaling_loss": scaling_loss,
                "opacity_loss": opacity_loss,
                "xyz": xyz.detach().clone(),
                }
    else:
        return {"render": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "xyz": xyz.detach().clone(),
                }
    

def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, resolution_scaling_factor=1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height * resolution_scaling_factor),
        image_width=int(viewpoint_camera.image_width * resolution_scaling_factor), 
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0


def render_anchor(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, override_color=None, resolution_scaling_factor=1.0, scaledown_ratio=None):

    # acquire the anchor attributes
    xyz = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    opacity = pc.opacity_activation(pc._opacity)[visible_mask]

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height * resolution_scaling_factor),
        image_width=int(viewpoint_camera.image_width * resolution_scaling_factor),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_anchor
    scales = None
    rotations = None
    cov3D_precomp = None
    # if pipe.compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    #     cov3D_precomp = cov3D_precomp[visible_mask]
    # else:
    scales = pc.get_scaling
    rotations = pc.get_rotation
    scales = scales[visible_mask]
    rotations = rotations[visible_mask]
    
    ###############################################################
    if scaledown_ratio is not None and scaledown_ratio > 0.0 and scaledown_ratio < 1.0:
        min_scale = pc.voxel_size * 1.0
        scales[:, :3] = scales[:, :3] * scaledown_ratio
        scales[:, :3] = torch.where(scales[:, :3] < min_scale, min_scale, scales[:, :3])
    ###############################################################

    num_per_row = 8
    image_row_list = []
    image_list = []
    for i in range(feat.shape[1]):
        color = feat[:, i:i+1].repeat(1, 3)
        screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        rendered_image, radii = rasterizer(
            means3D = xyz,
            means2D = screenspace_points,
            shs = None,
            colors_precomp = color,
            opacities = opacity,
            scales = scales[:, :3],
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        if len(image_row_list) == num_per_row:
            image_list.append(torch.cat(image_row_list, dim=2))
            image_row_list = []
        image_row_list.append(rendered_image)

    if len(image_row_list) > 0:
        image_list.append(torch.cat(image_row_list, dim=2))
    image_list = torch.cat(image_list, dim=1)
    
    return {"render": image_list}



def render_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, xyz=None, color=None, opacity=None, scaling=None, rot=None, absgrad=False, override_training=False, resolution_scaling_factor=1.0, interpolation=False, scaledown_ratio=None, antialias=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    if override_training:
        is_training = override_training

    if xyz is None:
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

    # viewpoint attributes
    fovx = viewpoint_camera.FoVx
    fovy = viewpoint_camera.FoVy
    W = int(viewpoint_camera.image_width * resolution_scaling_factor)
    H = int(viewpoint_camera.image_height * resolution_scaling_factor)
    fx = W / (2 * math.tan(fovx / 2))
    fy = H / (2 * math.tan(fovy / 2))
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda").unsqueeze(0)
    w2c = viewpoint_camera.world_view_transform.unsqueeze(0)

    # rendering all features
    """
    meta.keys() = {'camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics', 'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids', 'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras'}
    """
    if antialias:
        rasterize_mode = "antialiased"
    else:
        rasterize_mode = "classic"
    rendered_image, rendered_alphas, meta = gsplat.rendering.rasterization(
        xyz,
        rot,
        scaling,
        opacity.squeeze(1),
        color,
        w2c.transpose(1, 2),
        K,
        W,
        H,
        render_mode="RGB+D",
        packed=True, 
        absgrad=absgrad,
        rasterize_mode=rasterize_mode
    )
    rendered_depth = rendered_image.squeeze(0)[..., 3:4].permute(2, 0, 1)
    rendered_image = rendered_image.squeeze(0)[..., :3].permute(2, 0, 1)
    rendered_alphas = rendered_alphas.squeeze(0).permute(2, 0, 1)

    visibility_filter = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    visibility_filter[meta['gaussian_ids']] = True

    if not absgrad:
        meta['means2d'].retain_grad()
    if is_training:
        return {"render": rendered_image,
                "alpha": rendered_alphas,
                "depth": rendered_depth,
                # "viewspace_points": meta["means2d"],
                "viewspace_points": meta,
                "visibility_filter" : visibility_filter,
                "radii": meta["radii"],
                "selection_mask": mask, 
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "xyz": xyz.detach().clone(),
                }
    else:
        return {"render": rendered_image,
                "alpha": rendered_alphas,
                "depth": rendered_depth,
                # "viewspace_points": meta["means2d"],
                "viewspace_points": meta, 
                "visibility_filter" : meta["radii"] > 0,
                "radii": meta["radii"],
                "xyz": xyz.detach().clone(),
                }



def render_with_consistency_loss_gsplat(viewpoint_camera, pc : GaussianModel, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, absgrad=False, xyz=None, color=None, opacity=None, scaling=None, rot=None, resolution_scaling_factor=1.0, interpolation=False, scaledown_ratio=None, antialias=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
    consistency_loss, color_loss, rot_loss, scaling_loss, opacity_loss = 0, 0, 0, 0, 0

    if xyz is None:
        if is_training:
            xyz, color, color_main, opacity, scaling, scaling_main, rot, rot_main, neural_opacity, neural_opacity_main, mask = generate_neural_gaussians_and_momentum_gaussians(viewpoint_camera, pc, momentum_mlp_color, momentum_mlp_cov, momentum_mlp_opacity, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

            color_loss = torch.nn.functional.mse_loss(color, color_main)
            rot_loss = torch.nn.functional.mse_loss(rot, rot_main)
            scaling_loss = torch.nn.functional.mse_loss(scaling, scaling_main)
            opacity_loss = torch.nn.functional.mse_loss(neural_opacity, neural_opacity_main)

            consistency_loss = color_loss + rot_loss + scaling_loss + opacity_loss
        else:
            xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

    # viewpoint attributes
    fovx = viewpoint_camera.FoVx
    fovy = viewpoint_camera.FoVy
    W = int(viewpoint_camera.image_width * resolution_scaling_factor)
    H = int(viewpoint_camera.image_height * resolution_scaling_factor)
    fx = W / (2 * math.tan(fovx / 2))
    fy = H / (2 * math.tan(fovy / 2))
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda").unsqueeze(0)
    w2c = viewpoint_camera.world_view_transform.unsqueeze(0)

    # rendering all features
    """
    meta.keys() = {'camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics', 'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids', 'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras'}
    """
    if antialias:
        rasterize_mode = "antialiased"
    else:
        rasterize_mode = "classic"
    rendered_image, rendered_alphas, meta = gsplat.rendering.rasterization(
        xyz,
        rot,
        scaling,
        opacity.squeeze(1),
        color,
        w2c.transpose(1, 2),
        K,
        W,
        H,
        render_mode="RGB+D",
        packed=True, 
        absgrad=absgrad, 
        rasterize_mode=rasterize_mode
    )
    rendered_depth = rendered_image.squeeze(0)[..., 3:4].permute(2, 0, 1)
    rendered_image = rendered_image.squeeze(0)[..., :3].permute(2, 0, 1)
    rendered_alphas = rendered_alphas.squeeze(0).permute(2, 0, 1)

    visibility_filter = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    visibility_filter[meta['gaussian_ids']] = True
    
    if not absgrad:
        meta['means2d'].retain_grad()
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "alpha": rendered_alphas,
                "depth": rendered_depth,
                # "viewspace_points": meta['means2d'],
                "viewspace_points": meta, 
                "visibility_filter" : visibility_filter,
                "radii": meta['radii'],
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "consistency_loss": consistency_loss,
                "color_loss": color_loss,
                "rot_loss": rot_loss,
                "scaling_loss": scaling_loss,
                "opacity_loss": opacity_loss,
                "xyz": xyz.detach().clone(),
                }
    else:
        return {"render": rendered_image,
                # "viewspace_points": meta['means2d'],
                "viewspace_points": meta, 
                "visibility_filter" : meta['radii'] > 0,
                "radii": meta['radii'],
                "xyz": xyz.detach().clone(),
                }



def prefilter_voxel_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, resolution_scaling_factor=1.0, absgrad=False, antialias=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    means3D = pc.get_anchor
    covars = None
    quats = pc.get_rotation
    scales = pc.get_scaling

    # viewpoint attributes
    fovx = viewpoint_camera.FoVx
    fovy = viewpoint_camera.FoVy
    W = int(viewpoint_camera.image_width * resolution_scaling_factor)
    H = int(viewpoint_camera.image_height * resolution_scaling_factor)
    fx = W / (2 * math.tan(fovx / 2))
    fy = H / (2 * math.tan(fovy / 2))
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda").unsqueeze(0)
    w2c = viewpoint_camera.world_view_transform.unsqueeze(0)

    if antialias:
        rasterize_mode = "antialiased"
    else:
        rasterize_mode = "classic"
    proj_results = gsplat.fully_fused_projection(
        means3D,
        covars,
        quats,
        scales[:, :3] * scaling_modifier,
        w2c.transpose(1, 2),
        K,
        W, 
        H, 
        packed=True,
        calc_compensations=(rasterize_mode=="antialiased"),
    )

    (
        camera_ids,
        gaussian_ids,
        radii,
        means2d,
        depths,
        conics,
        compensations,
    ) = proj_results
    
    mask = torch.zeros(means3D.shape[0], dtype=torch.bool, device=means3D.device)
    mask[gaussian_ids] = True

    return mask



def render_anchor_gsplat(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, override_color=None, resolution_scaling_factor=1.0, scaledown_ratio=None, antialias=False):
    # gaussian attributes
    means = pc.get_anchor[visible_mask]
    feat = pc._anchor_feat[visible_mask]
    # opacity = pc.opacity_activation(pc._opacity)[visible_mask]
    opacity = torch.ones_like(pc._opacity[visible_mask], device=pc.get_anchor.device) * 0.5
    scales = pc.get_scaling[visible_mask]
    quats = pc.get_rotation[visible_mask]

    ###############################################################
    if scaledown_ratio is not None and scaledown_ratio > 0.0 and scaledown_ratio < 1.0:
        min_scale = pc.voxel_size * 1.0
        scales[:, :3] = scales[:, :3] * scaledown_ratio
        scales[:, :3] = torch.where(scales[:, :3] < min_scale, min_scale, scales[:, :3])
    ###############################################################

    # viewpoint attributes
    fovx = viewpoint_camera.FoVx
    fovy = viewpoint_camera.FoVy
    W = int(viewpoint_camera.image_width * resolution_scaling_factor)
    H = int(viewpoint_camera.image_height * resolution_scaling_factor)
    fx = W / (2 * math.tan(fovx / 2))
    fy = H / (2 * math.tan(fovy / 2))
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda").unsqueeze(0)
    w2c = viewpoint_camera.world_view_transform.unsqueeze(0)
    
    # rendering all features
    """
    meta.keys() = {'camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics', 'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids', 'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras'}
    """
    C = feat.shape[1]
    if antialias:
        rasterize_mode = "antialiased"
    else:
        rasterize_mode = "classic"
    rendered_image, rendered_alphas, meta = gsplat.rendering.rasterization(
        means,
        quats, 
        scales[:, :3] * scaling_modifier, 
        opacity.squeeze(1),
        feat,
        w2c.transpose(1, 2),
        K,
        W, 
        H, 
        render_mode="RGB+D",
        packed=False, 
        rasterize_mode=rasterize_mode
    )

    rendered_depth = rendered_image.squeeze(0)[..., C:C+1].permute(2, 0, 1)
    rendered_image = rendered_image.squeeze(0)[..., :C].permute(2, 0, 1)
    rendered_alphas = rendered_alphas.squeeze(0).permute(2, 0, 1)

    num_per_row = 8
    channels, height, width = rendered_image.shape
    assert channels == C
    image_list = rendered_image.reshape(num_per_row, -1, height, width)
    image_list = einops.rearrange(image_list, 'n1 n2 h w -> (n1 h) (n2 w)')
    image_list = image_list.unsqueeze(0)
    return {"render": image_list,
            "alpha": rendered_alphas,
            "depth": rendered_depth}



def render_gsplat_xyzonly(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, xyz=None, color=None, opacity=None, scaling=None, rot=None, absgrad=False, override_training=False, resolution_scaling_factor=1.0, interpolation=False, scaledown_ratio=None, antialias=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=False, interpolation=interpolation, scaledown_ratio=scaledown_ratio)

    color = color.detach().clone()
    opacity = opacity.detach().clone()
    scaling = scaling.detach().clone()
    rot = rot.detach().clone()

    # viewpoint attributes
    fovx = viewpoint_camera.FoVx
    fovy = viewpoint_camera.FoVy
    W = int(viewpoint_camera.image_width * resolution_scaling_factor)
    H = int(viewpoint_camera.image_height * resolution_scaling_factor)
    fx = W / (2 * math.tan(fovx / 2))
    fy = H / (2 * math.tan(fovy / 2))
    K = torch.tensor([[fx, 0, W / 2], [0, fy, H / 2], [0, 0, 1]], device="cuda").unsqueeze(0)
    w2c = viewpoint_camera.world_view_transform.unsqueeze(0)

    # rendering all features
    """
    meta.keys() = {'camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'conics', 'opacities', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids', 'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras'}
    """
    if antialias:
        rasterize_mode = "antialiased"
    else:
        rasterize_mode = "classic"
    rendered_image, rendered_alphas, meta = gsplat.rendering.rasterization(
        xyz,
        rot,
        scaling,
        opacity.squeeze(1),
        color,
        w2c.transpose(1, 2),
        K,
        W,
        H,
        render_mode="RGB+D",
        packed=False, 
        absgrad=absgrad, 
        rasterize_mode=rasterize_mode
    )
    rendered_depth = rendered_image.squeeze(0)[..., 3:4].permute(2, 0, 1)
    rendered_image = rendered_image.squeeze(0)[..., :3].permute(2, 0, 1)
    rendered_alphas = rendered_alphas.squeeze(0).permute(2, 0, 1)

    visibility_filter = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    visibility_filter[meta['gaussian_ids']] = True

    return {"render": rendered_image,
            "alpha": rendered_alphas,
            "depth": rendered_depth,
            }
