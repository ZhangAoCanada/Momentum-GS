# (1) untar the files
cd data/matrix_city/aerial/train
for num in {1..10}
do  
mkdir block_$num/input
tar -xvf block_$num.tar
mv block_$num/*.png block_$num/input
done 

cd ../test
for num in {1..10}
do  
mkdir block_${num}_test/input
tar -xvf block_${num}_test.tar
mv block_${num}_test/*.png block_${num}_test/input
done 

cd ../../../..

echo "untar the files done."

# (2) preprocessing
mkdir -p data/matrix_city/aerial/train/block_all/input
mkdir -p data/matrix_city/aerial/test/block_all/input
cp data/matrix_city/aerial/pose/block_all/transforms_train.json data/matrix_city/aerial/train/block_all/transforms.json
cp data/matrix_city/aerial/pose/block_all/transforms_test.json data/matrix_city/aerial/test/block_all/transforms.json

python tools/transform_json2txt.py --source_path data/matrix_city/aerial/train/block_all
python tools/transform_json2txt.py --source_path data/matrix_city/aerial/test/block_all

echo 'Transform json to txt done.'

mv data/colmap_results/matrix_city/train/sparse data/matrix_city/aerial/train/block_all/
mv data/colmap_results/matrix_city/test/sparse data/matrix_city/aerial/test/block_all/

echo 'Copy Colmap results done.'

cp -r data/partition/matrix_city/data_partitions data/matrix_city/aerial/train/block_all/

echo 'Copy partition files done.'
echo 'Preprocess MatrixCity done.'