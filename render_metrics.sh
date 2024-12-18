test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion/test
output_folder=outputs/pretrained/matrixcity
gpu_list=3
iteration=60000

CUDA_VISIBLE_DEVICES=${gpu_list} python render.py -m ${output_folder} --iteration ${iteration} --skip_train --custom_test ${test_path} --resolution 1
# CUDA_VISIBLE_DEVICES=${gpu_list} python metrics.py -m ${output_folder}

