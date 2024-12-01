output_folder=$1
iteration=60000

python merge_blocks.py -m ${output_folder} --iteration ${iteration}
python render.py -m ${output_folder} --iteration ${iteration} --skip_train
python metrics.py -m ${output_folder}

