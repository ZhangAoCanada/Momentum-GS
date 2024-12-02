# Rename image folders
mv data/mill19/building-pixsfm/train/rgbs data/mill19/building-pixsfm/train/images
mv data/mill19/building-pixsfm/val/rgbs data/mill19/building-pixsfm/val/images
mv data/mill19/rubble-pixsfm/train/rgbs data/mill19/rubble-pixsfm/train/images
mv data/mill19/rubble-pixsfm/val/rgbs data/mill19/rubble-pixsfm/val/images

echo "Rename image folders done."

# Copy Colmap results
cp -r data/colmap_results/building/train/sparse data/mill19/building-pixsfm/train/
cp -r data/colmap_results/building/val/sparse data/mill19/building-pixsfm/val/
cp -r data/colmap_results/rubble/train/sparse data/mill19/rubble-pixsfm/train/
cp -r data/colmap_results/rubble/val/sparse data/mill19/rubble-pixsfm/val/

echo "Copy Colmap results done."

# Copy partition files
cp -r data/partition/building/data_partitions data/mill19/building-pixsfm/train/
cp -r data/partition/rubble/data_partitions data/mill19/rubble-pixsfm/train/

echo "Copy partition files done."

# Transform pt to txt
python tools/transform_pt2txt.py --source_path data/mill19/building-pixsfm
python tools/transform_pt2txt.py --source_path data/mill19/rubble-pixsfm

echo "Preprocess Mill19 done."