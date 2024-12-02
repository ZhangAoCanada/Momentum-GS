import os
import torch
import numpy as np
import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
from gaussian_renderer import render, prefilter_voxel
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import ssim
from utils.partition_utils import contract_to_unisphere, get_default_aabb


def block_partitioning(dataset, pipe, iteration, scale=1.0, quiet=False, disable_inblock=False, ply_path=None):
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    cameras = scene.getTrainCameras()
    xyz_org = gaussians._anchor

    num_blocks = args.num_blocks
    block_dims = args.block_dims

    print(f"### Block number: {num_blocks}, block_dims: {block_dims}")  

    print(f'### camera numbers={len(cameras)}')

    if args.scene_aabb == [-1, -1, -1, -1, -1, -1]:
        torch.cuda.empty_cache()
        aabb = get_default_aabb(dataset, cameras, xyz_org, scale)
        print(f'### aabb={aabb}')
        np.save(os.path.join(dataset.source_path, "data_partitions", f"aabb.npy"), np.array(aabb.detach().cpu()))
    else:
        assert len(args.scene_aabb) == 6, "Unknown aabb format!"
        aabb = torch.tensor(args.scene_aabb, dtype=torch.float32, device=xyz_org.device)

    camera_mask = torch.zeros((len(cameras), num_blocks), dtype=torch.bool, device=xyz_org.device)

    with torch.no_grad():
        for block_id in range(num_blocks):

            block_id_z = block_id // (block_dims[0] * block_dims[1]) 
            block_id_y = (block_id % (block_dims[0] * block_dims[1])) // block_dims[0] 
            block_id_x = (block_id % (block_dims[0] * block_dims[1])) % block_dims[0] 

            xyz = contract_to_unisphere(xyz_org, aabb, ord=torch.inf)
            min_x, max_x = float(block_id_x) / block_dims[0], float(block_id_x + 1) / block_dims[0]
            min_y, max_y = float(block_id_y) / block_dims[1], float(block_id_y + 1) / block_dims[1]
            min_z, max_z = float(block_id_z) / block_dims[2], float(block_id_z + 1) / block_dims[2]

            org_min_x, org_max_x, org_min_y, org_max_y, org_min_z, org_max_z = min_x, max_x, min_y, max_y, min_z, max_z

            block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                        & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                        & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)

            block_mask = ~block_mask

            choose_inbox, choose_ssim = 0, 0

            for idx in tqdm(range(len(cameras)), desc=f"Block {block_id} / {num_blocks}"):
                bg_color = [1,1,1] if args.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device=xyz_org.device)
                c = cameras[idx]
                contract_cam_center = contract_to_unisphere(c.camera_center, aabb, ord=torch.inf)

                if (not disable_inblock) and contract_cam_center[0] > org_min_x and contract_cam_center[0] < org_max_x \
                    and contract_cam_center[1] > org_min_y and contract_cam_center[1] < org_max_y \
                    and contract_cam_center[2] > org_min_z and contract_cam_center[2] < org_max_z :
                    camera_mask[idx, block_id] = True
                    choose_inbox += 1
                    continue

                voxel_visible_mask = prefilter_voxel(c, gaussians, pipe, background)
                render_pkg_block = render(c, gaussians, pipe, background, visible_mask=voxel_visible_mask)
                org_image_block = render_pkg_block["render"]

                anchor_backup = gaussians._anchor
                anchor_feat_backup = gaussians._anchor_feat
                offset_backup = gaussians._offset
                scaling_backup = gaussians._scaling
                rotation_backup = gaussians._rotation
                opacity_backup = gaussians._opacity

                gaussians._anchor = gaussians._anchor[block_mask]
                gaussians._anchor_feat = gaussians._anchor_feat[block_mask]
                gaussians._offset = gaussians._offset[block_mask]
                gaussians._scaling = gaussians._scaling[block_mask]
                gaussians._rotation = gaussians._rotation[block_mask]
                gaussians._opacity = gaussians._opacity[block_mask]

                voxel_visible_mask = prefilter_voxel(c, gaussians, pipe, background)
                render_pkg_block = render(c, gaussians, pipe, background, visible_mask=voxel_visible_mask)
                image_block = render_pkg_block["render"]

                gaussians._anchor = anchor_backup
                gaussians._anchor_feat = anchor_feat_backup
                gaussians._offset = offset_backup
                gaussians._scaling = scaling_backup
                gaussians._rotation = rotation_backup
                gaussians._opacity = opacity_backup

                loss = 1.0 - ssim(image_block, org_image_block)
                if loss > args.ssim_threshold:
                    camera_mask[idx, block_id] = True
                    choose_ssim += 1

            print(f"### Block {block_id} / {num_blocks} has {choose_inbox} cameras in box, {choose_ssim} cameras selected by SSIM.")
    
    if not quiet:
        for block_id in range(num_blocks):
            print(f"Block {block_id} / {num_blocks} has {camera_mask[:, block_id].sum()} cameras.")
            with open(f"{args.source_path}/data_partitions/block{num_blocks}_ssim{args.ssim_threshold}.npy.log", "a") as f:
                f.write(f"Block {block_id} / {num_blocks} has {camera_mask[:, block_id].sum()} cameras.\n")
                
    return camera_mask


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ssim_threshold", default=0.1, type=float)
    parser.add_argument("--num_blocks", type=int, default=-1, help="Number of blocks to divide the scene into")
    parser.add_argument("--block_dims", type=int, nargs=3, default=[-1, -1, -1], help="Dimensions of each block (x, y, z)")
    parser.add_argument("--scene_aabb", type=int, nargs=6, default=[-1, -1, -1, -1, -1, -1], help="Axis-aligned bounding box (min_x, min_y, min_z, max_x, max_y, max_z)")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default = '-1')
    args = get_combined_args(parser)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    ssim_threshold = args.ssim_threshold
    print(f"SSIM threshold: {ssim_threshold}")
    
    if not os.path.exists(os.path.join(args.source_path, f"data_partitions")):
        os.makedirs(os.path.join(args.source_path, f"data_partitions"))

    camera_mask = block_partitioning(lp.extract(args), pp.extract(args), args.iteration)
    camera_mask = camera_mask.cpu().numpy()

    num_blocks = args.num_blocks

    np.save(os.path.join(args.source_path, f"data_partitions/camera_mask.npy"), camera_mask)

    print("### Block partitioning finished.")
