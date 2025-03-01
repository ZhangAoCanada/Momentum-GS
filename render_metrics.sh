gpu_list=1
test_path=data/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial/test
images=input_cached

# output_folder=outputs/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train/mc-1gpu-smallaera-aerial-freq+interpolation/wogsplat-depth-l2-0.1-freq-offset+scale-interpolation1000-10-notrain-scalereg-minmaxsum
output_folder=outputs/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train/mc-1gpu-smallaera-aerial+lilstreet-raw/original
iteration=60000

CUDA_VISIBLE_DEVICES=${gpu_list} python render.py -m ${output_folder} --iteration ${iteration} --skip_train --custom_test ${test_path} --images ${images} --resolution 1
