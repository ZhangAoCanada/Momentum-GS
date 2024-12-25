# Developing Harder

## Reference Code Reading

### [GussianSR](https://arxiv.org/abs/2407.18046)

No public repository available. The pseudocode is offered as follow.

> **Algorithm 1** A Simple Example of 2D Gaussian Splatting \
> **Require**: $\Sigma_x$, $\Sigma_y$, $\rho$ (covariance param), $coords$ (point coordinates), $color$ (point colors), $image\_size$ \
> **Ensure**: $final\_image$ (rendered image) \
> 1: Compute the covariance matrix $\Sigma$ using $\Sigma_x$, $\Sigma_y$, and $\rho$ \
> 2: Check if $\Sigma$ is positive definite \
> 3: Compute the inverse of $\Sigma$: $\Sigma^{-1}$ \
> 4: Create a 2D grid $x, y$ in the range $[-5, 5]$ \
> 5: Compute the Gaussian kernel $K$ using $x, y$, and $\Sigma^{-1}$ \
> 6: Normalize $K$ to $[0, 1]$ range \
> 7: Repeat $K$ along the channel dimension to match colors \
> 8: Pad $K$ with zeros to match $image\_size$ \
> 9: Create a batch of 2D affine transformation matrices $\Theta$ using $coords$ \
> 10: Apply affine transformations to $K$ using $\Theta$ to obtain $K_{translated}$ \
> 11: Multiply $K_{translated}$ with $color$ to get image layers \
> 12: Sum the image layers to obtain $final\_image$ \
> 13: Clamp $final\_image$ to $[0, 1]$ range \
> 14: Permute $final\_image$ to match the channel order \
> **Return** $final\_image$


## Exploration Experiments

### Data Processing

The MatrixCity block A is composed of totally `1063` **aerial** frames and `4075` **street** frames. The total number of the entire MatrixCity SfM points is `3826641`. 

The entire MatrixCity points         |  Camera trajectory of MatrixCity block A
:-----------------------------------:|:-----------------------------------------:
<!-- ![all](/experiments/assets/all.png)  |  ![ba](/experiments/assets/ba.png) -->
<img src="../assets/all.png" width="450" height="300">  | <img src="../assets/ba.png" width="450" height="300"> 

To extract a small port of data for the convenience of development, we select a square area of the MatrixCity block A, which is shown as follows.

<!-- ![small_aera](/experiments/assets/small_aera.png) -->
<img src="../assets/small_aera.png" width="490" height="300">

The selected region contains totally `1229` frames, detailed below
- **aerial**: `189` frames
- **street**: `1040` frames
- **SfM points**: `760348` points