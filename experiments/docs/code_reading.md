# Code Reading

## Table of Contents

- [Data Partition](#data-partition)
    - [Bound the unbounded scenes](#bound-the-unbounded-scenes)
    - [Select the cam views](#select-the-cam-views)
- [Train partitioned scene](#train-partitioned-scene)
    - [Scaffold-gs Basic Setup](#scaffold-gs-basic-setup)
    - [Scaffold-gs Densification](#scaffold-gs-densification)
    - [Momentum MLP](#momentum-mlp)
- [Interpolation (Draw from exisiting gs-sr)](#interpolation-draw-from-exisiting-gs-sr)
- [Diffusion (Draw from nerf-sr or IRSR)](#diffusion-draw-from-nerf-sr-or-irsr)


## Data Partition

### Bound the unbounded scenes

`data_partition.py` plays a crucial role in the data partitioning process. To enable the data partitioning, give the correct arguments to the script after `train_coarse.sh`. An example is shown as below
```bash
python data_partition.py -m ${COARSE_MODEL_FOLDER} --ssim_threshold 0.05 --iteration 30000 --num_blocks 4 --block_dims 2 1 2 --scene_aabb -140 -100 0 -10 900 250
```
The arguments are explained as follows:
```yaml
num_blocks: "total block number to be partitioned" # normally 8
block_dims: "block number in x, y, z direction" # normally 2 1 4, make sure the product equals to num_blocks
aabb: "normalization mapping, contrain the inner foreground region with linear space mapping" # [min_x, min_y, min_z, max_x, max_y, max_z]
```
Specifically, `aabb` is the $p_{min}$ and $p_{max}$ in the following equation:
```math
\hat{p}_k = 2(\frac{p_k - p_{min}}{p_{max} - p_{min}}) - 1
```
with $\hat{p}_k$ defined, the normalization mapping is defined as follows:
```math
\text{contract}(\hat{p}_k) =  \left \{ 
    \begin{array}{ll}
        \hat{p}_k, & \text{if } ||\hat{p}_k||_{\infty} \leq 1 \\
        \big( 2 - \frac{1}{||\hat{p}_k||_{\infty}} \big) \frac{\hat{p}_k}{||\hat{p}_k||_{\infty}}, & \text{if } ||\hat{p}_k||_{\infty} > 1
    \end{array}
\right.
```
with this, all coarse points will be normalized to the range of $[-1, 1]$.


### Select the cam views

To understand it in an easier way:
```python
if cam_view in block_i:
    return True
else:
    rendered_image = render(cam_view, gaussians)
    rendered_image_wo_block_i = render(cam_view, gaussians[not block_i])
    if 1 - ssim(rendered_image, rendered_image_wo_block_i) > ssim_threshold:
        return True
    else:
        return False
```

## Train partitioned scene

### Scaffold-gs Basic Setup

The parameters of Gaussian Splatting model directly duplicated from Scaffold-gs. 
<!-- The Basic configuration setups are:
```python
self.update_init_factor = 128
self.update_hirarchy_factor = 4
self.update_depth = 3
``` -->
Each anchor contains the following attributes parameters, 
```python
self._anchor = (N, 3)
self._offset = (N, 10, 3) # 10 offset points
self._anchor_feat = (N, 32) # feat_dim = 32
self._scaling = (N, 6)
self._rotation = (N, 4)
self._opacity = (N, 1)
```
and $3$ MLPs.
```python
self.mlp_opacity
self.mlp_conv
self.mlp_color
```
<!-- and the following gradient parameters for densification:
```python
self.opacity_accum
self.max_radii2D
self.offset_gradient_accum
self.offset_denom
self.anchor_denom
```
 -->
**NOTE**: All blocks share the same MLP parameters. At the beginning of the training stage, the parameters are copied from the previous model.


### Scaffold-gs Densification

Explained in the paper, for a given voxel size $\epsilon_g^{(m)}$ with $m=\{1,2,3\}$ indicating multi-resolution, the average gradients of the Gaussian balls in each voxel are computed over $N$ iterations, denoted as $\triangledown_g^{(m)}$. New anchors are generated w.r.t the Gaussian balls when $\triangledown_g^{(m)} > \tau_g^{(m)}$.
```math
\epsilon_g^{(m)} = \epsilon_g / 4^{(m-1)}
\tau_g^{(m)} = \tau_g \cdot 2^{(m-1)}
```
where, $\epsilon_g$ is the pre-defined voxel size and $\tau_g$ is the pre-defined threshold.

The `anchor_growing` function in `gaussian_model.py` implements the process above. Key parameters are listed below:
```python
cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
cur_size = self.voxel_size*size_factor
# find the satisfied offsets (>threshold) that are not in the 
# current anchor set, and initialize new anchors accordingly.
candidate_mask = (grads >= cur_threshold)
candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size
```

For anchor pruning, simply use,
```python
# Prune the anchors with opacity < min_opacity
prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
# anchors that have been trained enough time
anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
prune_mask = torch.logical_and(prune_mask, anchors_mask)
```

### Momentum MLP

Momentum MLPs, which are also called Teacher Model, are initialized from the coarse `gaussians.mlp_opacity`, `gaussians.mlp_cov` and `gaussians.mlp_color`.
```python
momentum_mlp_color = copy.deepcopy(gaussians.mlp_color).to(device)
momentum_mlp_cov = copy.deepcopy(gaussians.mlp_cov).to(device)
momentum_mlp_opacity = copy.deepcopy(gaussians.mlp_opacity).to(device)
```
*The backpropagation of momentum MLPs are disabled*. The update policy of the momentum MLPs (Teacher Model) is defined as in the paper:
```math
\theta_t \leftarrow m \cdot \theta_t + (1 - m) \cdot \theta_s
```
where, $\theta_t$ is the parameters of the **Teacher Decoder $D_m$** and $\theta_s$ is the parameters of the **Student Decoder $D_s$**. In the implementation, $m=0.9$.
In order to keep the consistency of the teacher decoder $D_m$ (global) and the student decoder $D_s$ (local), a consistency loss is introduced to encourage $D_s$ align with $D_m$.
```math
\mathcal{L}_{consistency} = || D_m(f_b, v_b; \theta_t) - D_o(f_b, v_b; \theta_s) ||_2
```
where, $f_b$ is the anchor feature and $v_b$ is the view direction.
Specifically, the implementation of the consistency loss is conducted during rendering. The details are shown as follows:
```python
color_loss = F.mse_loss(color, color_main)
rot_loss = F.mse_loss(rot, rot_main)
scaling_loss = F.mse_loss(scaling, scaling_main)
opacity_loss = F.mse_loss(opacity, opacity_main)
consistency_loss = color_loss + rot_loss + scaling_loss + opacity_loss
```
where, all `_main` are derived from the teacher model.


Therefore, the total loss is defined in `python` as follows:
```python
loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg 
loss = (loss + consistency_loss * opt.consistency_loss_weight) * recons_weight
```

`recons_weight` is defined with `momentum_psnr, momentum_ssim` (Momentum update metrics). The momentum update coefficient is defined as $0.9$. In details,
```python
momentum_psnr = 0.9 * momentum_psnr + 0.1 * cur_psnr
momentum_ssim = 0.9 * momentum_ssim + 0.1 * cur_ssim
```

Finding the max value of `momentum_psnr` and `momentum_ssim` with all trianing process in all parallel GPUs to get `max_psnr` and `max_ssim`. The deviation of PSNR and SSIM are derived from `max_psnr - momentum_psnr` and `max_ssim - momentum_ssim`. `recons_weight` is then defined as,
```python
recons_weight = torch.tensor(2.0) - torch.exp(-((cur_max_psnr - momentum_psnr)**2 + (cur_max_ssim * 10 - momentum_ssim * 10)**2) / (2 * opt.adaptive_sigma * opt.adaptive_sigma))
```
Exactly the same as the paper.
```math
\omega_i = 2 - \exp{\Big( \frac{\delta_p^2 + \lambda \cdot \delta_s^2}{-2\sigma^2} \Big)}
```
where, $\sigma$, a.k.a `adaptive_sigma` is set to $9.0$.


## Interpolation (Draw from exisiting gs-sr)

## Diffusion (Draw from nerf-sr or IRSR)
