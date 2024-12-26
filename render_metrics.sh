gpu_list=2
test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small/test

output_folder=outputs/pretrianed/matrixcity
iteration=60000

# output_folder=outputs/bdaibdai___MatrixCity/small_city/blockA_fusion_small/train/mc-1gpu-smallaera-densification80kiter/2024-12-25_19:14:01
# iteration=100000

CUDA_VISIBLE_DEVICES=${gpu_list} python render.py -m ${output_folder} --iteration ${iteration} --skip_train --custom_test ${test_path} --images input --resolution 1
# CUDA_VISIBLE_DEVICES=${gpu_list} python metrics.py -m ${output_folder}

