import numpy as np
import cv2
from PIL import Image

import os, sys
from glob import glob
from tqdm import tqdm
import shutil


pretrianed_dir = "outputs/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train/mc-1gpu-smallaera-aerial-freq+interpolation/wogsplat-depth-l2-0.1-freq-offset+scale-interpolation1000-10-notrain-scalereg-minmaxsum/test/ours_80000"
wstreet_dir = "outputs/bdaibdai___MatrixCity/small_city/blockA_fusion_small_aerial+somestreet/train/mc-1gpu-smallaera-aerial+lilstreet-raw/original/test/ours_60000"
select_list = ["00025.png", "00031.png", "00038.png", "00134.png"]

gt_sub_dir = "gt"
render_sub_dir = "renders"
save_as_video = False

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
pretrianed_dir = os.path.join(parent_dir, pretrianed_dir)
wstreet_dir = os.path.join(parent_dir, wstreet_dir)

gt_dir = os.path.join(pretrianed_dir, gt_sub_dir)
pretrained_render_dir = os.path.join(pretrianed_dir, render_sub_dir)
wstreet_render_dir = os.path.join(wstreet_dir, render_sub_dir)
save_dir = os.path.join(current_dir, "tmp_compare")
save_video_dir = os.path.join(current_dir, "tmp_video")
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
if os.path.exists(save_video_dir):
    shutil.rmtree(save_video_dir)
os.makedirs(save_dir, exist_ok=True)
if save_as_video:
    os.makedirs(save_video_dir, exist_ok=True)

gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))
print(f"Total {len(gt_files)} images in the {gt_dir} directory.")

image_list = []

for gt_file in tqdm(gt_files):
    pretrained_render_file = os.path.join(pretrained_render_dir, os.path.basename(gt_file))
    wstreet_render_file = os.path.join(wstreet_render_dir, os.path.basename(gt_file))
    base_name = os.path.basename(gt_file)
    if len(select_list) > 0 and base_name not in select_list:
        continue
    gt = cv2.imread(gt_file)
    pretrained_render = cv2.imread(pretrained_render_file)
    wstreet_render = cv2.imread(wstreet_render_file)
    compare = np.concatenate([gt, pretrained_render, wstreet_render], axis=1)
    if save_as_video:
        image_list.append(compare)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(gt_file)), compare)

if save_as_video:
    print("Saving video...")
    fps = 1
    out = cv2.VideoWriter(os.path.join(save_video_dir, "compare.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (compare.shape[1], compare.shape[0]))
    for image in tqdm(image_list):
        out.write(image)
    out.release()

