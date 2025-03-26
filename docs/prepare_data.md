# Prepare data
Create data folder
```bash
mkdir data
```

### Colmap

For Colmap, we use the same results as [CityGaussian](https://github.com/DekuLiuTesla/CityGaussian). The following results are sourced from [this link](https://github.com/DekuLiuTesla/CityGaussian/blob/main/doc/data_preparation.md).

+ [Download from Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/dcda048066d946e98598/?dl=1)
+ [Download from Google Drive](https://drive.google.com/file/d/1bebYr7v9AwRju6cT3Zg23jmDjReevLH1/view?usp=drive_link)

Please download and unzip the Colmap results into `data/`.

### Scene partition
We adopt the similar divide-and-conquer strategy as CityGaussian. The following file divide each scene into **8 blocks**. Please download and unzip into `data/`.

+ [Download from Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/198d7b5909a8469bae09/?dl=1)
+ [Download from Google Drive](https://drive.google.com/file/d/1a0N4YJyeMiiRSGYkYe9OK6GUlIvixora/view?usp=drive_link)

[Optional] If you wish to divide the scene into a different number of blocks, you can follow these instructions (take scene `Building` as an example):

1. Complete all the subsequent data preparation first before starting scene partition.

2. Train a coarse model for SSIM-based view filtering.

```bash
bash script/train/train-building-coarse.sh
```

3. Divide the scene

```bash
bash data_partition.sh <COARSE_MODEL_FOLDER>
```

4. Modify `block_num`, `partition_name`, and `block_dim`  in corresponding training script.


### Mill 19 (Building, Rubble)
Please download these two scenes from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf) and extract them into the `data/mill19/` directory. You may refer to the folder structure at the bottom of this page for guidance.
+ [Dowload Building](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz)
+ [Dowload Rubble](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz)

Next, run following script to preprocess images. Please make sure you are in the `Momentum-GS/` directory.
```bash
bash script/data_preparation/preprocess_mill19.sh
```


### UrbanScene 3D (Residence, Sci-Art)
Please download these two scenes from [UrbanScene3D](https://github.com/Linxius/UrbanScene3D) and extract them into the `data/urbanscene3d` directory.
Besides, please download the refined camera poses from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf).
+ [Download Residence Camera Pose](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm.tgz)
+ [Download Sci-Art Camera Pose](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm.tgz)

Next, run the following script to preprocess images.
```bash
bash script/data_preparation/preprocess_urbanscene3d.sh
```

### MatrixCity (SmallCity-Aerial)
Please download the small_city-aerial scene from [MatrixCity](https://github.com/city-super/MatrixCity) into the `data/matrix_city/aerial` directory.

Next, run the following script to preprocess images.

```bash
bash script/data_preparation/preprocess_matrixcity.sh
```

### Cache images

Following previous methods (e.g. MegaNeRF, VastGaussian, CityGaussian), we downsample all images by a factor of 4 (except for MatrixCity, which will be dawnsampled to 1600\*900). This downsampling will be performed during each camera loading process. To enhance efficiency, we will downsample all images in advance.
```bash
bash script/data_preparation/downsample.sh
```





### Folder structure
```
├── data
│   ├── matrix_city
|   |   ├── aerial
|   │   │   ├── train
|   |   |   |   ├── block_all
|   |   │   │   │   ├── data_partitions
|   |   |   |   |   ├── input
|   |   |   |   |   ├── input_cached
|   |   │   │   │   ├── sparse
|   |   │   │   │   │   ├── 0
|   |   │   │   │   │   │   ├── cameras.bin
|   |   │   │   │   │   │   ├── points3D.bin
|   |   │   │   │   │   │   ├── images.bin
|   │   │   ├── test
|   |   |   |   ├── block_all
|   |   │   │   │   ├── input
|   |   │   │   │   ├── input_cached
|   |   │   │   │   ├── sparse
|   |   |   |   │   │   ├── ...
│   ├── mill19
|   │   ├── building-pixsfm
|   │   │   ├── train
|   |   │   │   ├── data_partitions
|   |   │   │   ├── images
|   |   │   │   ├── images_4
|   |   │   │   ├── sparse
|   |   |   │   │   ├── ...
|   |   |   ├── val
|   |   |   │   ├── images
|   |   |   │   ├── images_4
|   |   |   │   ├── sparse
|   |   |   │   │   ├── ...
|   │   ├── rubble-pixsfm
|   │   │   ├── train
|   |   |   |   ... (Similar to building-pixsfm)
|   |   |   ├── val
|   |   |   |   ... 
|   ├── urbanscene3d
|   │   ├── residence
|   │   │   ├── train
|   |   │   │   ├── ...
|   |   │   ├── val
|   |   │   │   ├── ...
|   |   ├── sci-art
|   |   │   ├── train
|   |   │   │   ├── ...
|   |   │   ├── val
|   |   │   │   ├── ...
```
