import numpy as np
import cv2
import os
import shutil
from tqdm import tqdm
import json

select_list = [
[1171, 1192],
[1488, 1502], 
[2012, 2025, 2049, 2072],
[3840, 3857],
[3780, 3799, 3813],
[3714, 3731],
[3641, 3665, 3682],
[3564],
[3516, 3528],
[3461],
[3395, 3406],
[3351],
[3104],
[3091, 3102],
[3024, 3048],
[2984, 3022],
[2505, 2529, 2553],
[2331, 2351, 2374, 2394],
[2188, 2211, 2234],
[2024, 2048, 2072],
]

remove_list = [
   2012, 2025, 2049, 3351, 3395, 2529,  3406, 3564, 2505, 3714, 3731, 3780, 3813, 3840, 3857
]

remove_2nd_list = [
    1502, 3682
]

remove_3rd_list = [
    3799
]

image_list = []
for i in range(len(select_list)):
    for j in range(len(select_list[i])):
        if select_list[i][j] in remove_list or select_list[i][j] in remove_2nd_list:
            continue
        image_list.append(select_list[i][j])

image_dir = "/data/zhangao/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train/input"
this_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(this_dir, "tmp_select")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

for image_id in tqdm(image_list):
    image_path = os.path.join(image_dir, f"{image_id}.png")
    shutil.copy(image_path, output_dir)


json_path = "/data/zhangao/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train/imagename_dict.json"
with open(json_path, "r", encoding="utf-8") as json_file:
    imagename_dict = json.load(json_file)

raw_list = []
for image_id in image_list:
    image_name = '%04d.png' % image_id
    raw_list.append("/".join(imagename_dict[str(image_name)].split('/')[-2:]))
print(raw_list)