mv data/urbanscene3d/residence-pixsfm data/urbanscene3d/residence
mv data/urbanscene3d/sci-art-pixsfm data/urbanscene3d/sci-art

# Copy Colmap results
cp -r data/colmap_results/residence/train/sparse data/urbanscene3d/residence/train/
cp -r data/colmap_results/residence/val/sparse data/urbanscene3d/residence/val/
cp -r data/colmap_results/sciart/train/sparse data/urbanscene3d/sci-art/train/
cp -r data/colmap_results/sciart/val/sparse data/urbanscene3d/sci-art/val/

echo "Copy Colmap results done."

# Copy partition files
cp -r data/partition/residence/data_partitions data/urbanscene3d/residence/train/
cp -r data/partition/sciart/data_partitions data/urbanscene3d/sci-art/train/

echo "Copy partition files done."

python copy_images.py --image_path data/urbanscene3d/Residence/photos --dataset_path data/urbanscene3d/residence
python copy_images.py --image_path data/urbanscene3d/Sci-Art/photos --dataset_path data/urbanscene3d/sci-art

echo "Preprocess Urbanscene3D (Residence, Sci-Art) done."