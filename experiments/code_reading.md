# Code Reading

> Questions with config:
> - `aabb`
> - `block_dim`
> - `consistency_loss_weight`
> - `adaptive_sigma`

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