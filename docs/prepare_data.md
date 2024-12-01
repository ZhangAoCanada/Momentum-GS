# Prepare data

```bash
# Create data folder
mkdir data
```

### Checkpoints
TODO

### Mill19 (Building, Rubble)
Please download these two scenes from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf) and extract them into the `data/mill19/` directory. You may refer to the folder structure at the bottom of this page for guidance.
+ [Dowload Building](https://storage.cmusatyalab.org/mega-nerf-data/building-pixsfm.tgz)
+ [Dowload Rubble](https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz)



### UrbanScene3D (Residence, Sci-Art)
Please download these two scenes from [UrbanScene3D](https://github.com/Linxius/UrbanScene3D) and extract them into the `data/urbanscene3d` directory.
Besides, please download the refined camera poses from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf)
+ [Download Residence Camera Pose](https://storage.cmusatyalab.org/mega-nerf-data/residence-pixsfm.tgz)
+ [Download Sci-Art Camera Pose](https://storage.cmusatyalab.org/mega-nerf-data/sci-art-pixsfm.tgz)


```bash
# preprocessing
python scripts/data_preparation/copy_images.py --image_path $RAW_PHOTO_PATH --dataset_path $CAMERA_POSE_PATH
```

### MatrixCity (SmallCity-Aerial)
Please download the small_city-aerial part from [MatrixCity](https://github.com/city-super/MatrixCity).
```bash
# preprocessing
bash scripts/data_preparation/untar_matrixcity_train.sh
bash scripts/data_preparation/untar_matrixcity_test.sh
bash scripts/data_preparation/data_proc_mc.sh
```

### Cache images

downsample 4X
TODO


### Colmap

For Colmap, we use the same results as [CityGaussian](https://github.com/DekuLiuTesla/CityGaussian). The following results are sourced from [this link](https://github.com/DekuLiuTesla/CityGaussian/blob/main/doc/data_preparation.md).
- **Google Drive**: https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1zX34zftxj07dCM1x5bzmbA?pwd=1t6r

Please download and unzip the COLMAP results into the corresponding scene folders under `./data`, as shown below:


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
