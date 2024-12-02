### mill19-rubble
# python data_partition.py -m $1 --ssim_threshold 0.05 --iteration 30000 --num_blocks 8 --block_dims 2 1 4 --scene_aabb -50 -100 -135 50 300 -5

### mill19-building
python data_partition.py -m $1 --ssim_threshold 0.05 --iteration 30000 --num_blocks 4 --block_dims 2 1 2 --scene_aabb -140 -100 0 -10 900 250

### urbanscend3d-residence
# python data_partition.py -m $1 --ssim_threshold 0.05 --iteration 30000 --num_blocks 8 --block_dims 2 1 4 --scene_aabb -25 -200 -270 175 200 60

### urbanscend3d-sciart
# python data_partition.py -m $1 --ssim_threshold 0.05 --iteration 30000 --num_blocks 8 --block_dims 2 1 4 --scene_aabb -110 -500 -205 55 100 90

### matrixcity-aerial
# python data_partition.py -m $1 --ssim_threshold 0.05 --iteration 60000 --num_blocks 16 --block_dims 4 4 1 --aabb -3.5 -4 -10 4.5 2 10