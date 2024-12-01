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

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


# From Mip-NeRF 360: https://github.com/google-research/multinerf/blob/main/internal/image.py#L81
# The following code is converted from original JAX code
def color_correct(img, ref, num_iters=5, eps=0.5 / 255, block_size=256):
    """Warp `img` to match the colors in `ref_img` using block-wise and per-channel processing."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f"img's {img.shape[-1]} and ref's {img.shape[-1]} channels must match"
        )

    # print(f'DEBUG: img.shape={img.shape}, ref.shape={ref.shape}')

    num_channels = img.shape[-1]
    h, w = img.shape[:2]

    def is_unclipped(z):
        return (z >= eps) & (z <= (1 - eps))

    # img = torch.from_numpy(img)
    # ref = torch.from_numpy(ref)

    # corrected_img = torch.zeros_like(torch.from_numpy(img))
    corrected_img = torch.zeros_like(img)

    #DEBUG
    # print(f'DEBUG: h={h}, type={type(h)}, w = {w}, type={type(w)}, block_size={block_size}, type={type(block_size)}')

    h, w = int(h), int(w)
    block_size = int(block_size)

    for i in range(0, h, block_size):
        # print('@')
        for j in range(0, w, block_size):
            # print(f'#')
            i, j, block_size = torch.tensor(i), torch.tensor(j), torch.tensor(block_size)
            img_block = img[i:i+block_size, j:j+block_size]
            ref_block = ref[i:i+block_size, j:j+block_size]

            img_mat = img_block.reshape(-1, num_channels)
            ref_mat = ref_block.reshape(-1, num_channels)

            mask0 = is_unclipped(img_mat)

            for _ in range(num_iters):
                for c in range(num_channels):
                    a_mat = []
                    # 构造a_mat，包括二次项和线性项
                    a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])
                    a_mat.append(img_mat[:, c:(c + 1)])  # 线性项
                    # a_mat.append(torch.ones_like(torch.from_numpy(img_mat[:, :1])))  # 常数项
                    a_mat.append(torch.ones_like(img_mat[:, :1]))  # 常数项
                    a_mat = torch.cat(a_mat, dim=-1)

                    warp = []
                    b = ref_mat[:, c]
                    mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
                    
                    # 避免一次性生成大的掩码矩阵
                    ma_mat = torch.zeros_like(a_mat)
                    mb = torch.zeros_like(b).unsqueeze(-1)

                    # 应用掩码逐步处理
                    ma_mat[mask] = a_mat[mask]
                    mb[mask] = b[mask].unsqueeze(-1)

                    # 解决线性方程组，使用更平滑的调整
                    weight = torch.linalg.lstsq(ma_mat, mb).solution.squeeze()
                    
                    # 调整修正幅度，避免过度修正
                    weight = weight * 0.1  # 控制修正权重，避免输出图像过度接近参考图像

                    # 更新图像矩阵当前通道，进行颜色修正
                    img_mat[:, c] = torch.clamp(torch.matmul(a_mat, weight), 0, 1)

            corrected_img[i:i+block_size, j:j+block_size] = img_mat.reshape(img_block.shape)

    return corrected_img




########### 改的跟gt几乎一致了
# def color_correct(img, ref, num_iters=5, eps=0.5 / 255, block_size=256):
#     """Warp `img` to match the colors in `ref_img` using block-wise and per-channel processing."""
#     with torch.no_grad():
#         if img.shape[-1] != ref.shape[-1]:
#             raise ValueError(
#                 f"img's {img.shape[-1]} and ref's {img.shape[-1]} channels must match"
#             )
        
#         num_channels = img.shape[-1]
#         h, w = img.shape[:2]

#         def is_unclipped(z):
#             return (z >= eps) & (z <= (1 - eps))

#         corrected_img = torch.zeros_like(img)

#         for i in range(0, h, block_size):
#             for j in range(0, w, block_size):
#                 img_block = img[i:i+block_size, j:j+block_size]
#                 ref_block = ref[i:i+block_size, j:j+block_size]

#                 img_mat = img_block.reshape(-1, num_channels)
#                 ref_mat = ref_block.reshape(-1, num_channels)

#                 mask0 = is_unclipped(img_mat)

#                 for _ in range(num_iters):
#                     for c in range(num_channels):
#                         a_mat = []
#                         # 逐通道构造a_mat，只对当前通道进行操作
#                         a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])
#                         a_mat.append(img_mat[:, c:(c + 1)])  # 加入线性项
#                         a_mat.append(torch.ones_like(img_mat[:, :1]))  # 加入常数项
#                         a_mat = torch.cat(a_mat, dim=-1)

#                         warp = []
#                         # 构建线性方程右侧
#                         b = ref_mat[:, c]
#                         mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
                        
#                         # 减少显存占用：逐步处理mask中的元素
#                         ma_mat = torch.zeros_like(a_mat)
#                         mb = torch.zeros_like(b).unsqueeze(-1)

#                         # 对掩码逐行应用
#                         ma_mat[mask] = a_mat[mask]
#                         mb[mask] = b[mask].unsqueeze(-1)
                        
#                         # 解决线性系统
#                         w = torch.linalg.lstsq(ma_mat, mb).solution.squeeze()
#                         warp.append(w)

#                         # 更新img_mat当前通道
#                         img_mat[:, c] = torch.clamp(torch.matmul(a_mat, w), 0, 1)

#                 corrected_img[i:i+block_size, j:j+block_size] = img_mat.reshape(img_block.shape)

#     return corrected_img











############## OOM version ##############
# def color_correct(img, ref, num_iters=5, eps=0.5 / 255):
#     """Warp `img` to match the colors in `ref_img`."""
#     if img.shape[-1] != ref.shape[-1]:
#         raise ValueError(
#             f"img's {img.shape[-1]} and ref's {ref.shape[-1]} channels must match"
#         )
    
#     num_channels = img.shape[-1]
#     img_mat = img.reshape(-1, num_channels)
#     ref_mat = ref.reshape(-1, num_channels)

#     def is_unclipped(z):
#         return (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps]

#     mask0 = is_unclipped(img_mat)

#     for _ in range(num_iters):
#         a_mat = []
#         for c in range(num_channels):
#             a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
#         a_mat.append(img_mat)  # Linear term.
#         a_mat.append(torch.ones_like(img_mat[:, :1]))  # Bias term.
#         a_mat = torch.cat(a_mat, dim=-1)

#         warp = []
#         for c in range(num_channels):
#             b = ref_mat[:, c]
#             mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
#             ma_mat = torch.where(mask.unsqueeze(1), a_mat, torch.zeros_like(a_mat))
#             mb = torch.where(mask, b, torch.zeros_like(b)).unsqueeze(-1)
#             # 使用 torch.linalg.lstsq 代替 torch.lstsq
#             w = torch.linalg.lstsq(ma_mat, mb).solution.squeeze()
#             assert torch.all(torch.isfinite(w))
#             warp.append(w)

#         warp = torch.stack(warp, dim=-1)

#         img_mat = torch.clamp(
#             torch.matmul(a_mat, warp), 0, 1
#         )

#     corrected_img = img_mat.reshape(img.shape)
#     return corrected_img


