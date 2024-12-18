import numpy as np
import cv2
from PIL import Image

import os, sys
from glob import glob
from tqdm import tqdm


experiments_dir = "outputs/pretrained/matrixcity/test/ours_60000"
gt_sub_dir = "gt"
render_sub_dir = "renders"
save_sub_dir = "compare"
save_as_video = True

gt_dir = os.path.join(experiments_dir, gt_sub_dir)
render_dir = os.path.join(experiments_dir, render_sub_dir)
save_dir = os.path.join(experiments_dir, save_sub_dir)
save_video_dir = os.path.join(save_dir, "videos")
os.makedirs(save_dir, exist_ok=True)
if save_as_video:
    os.makedirs(save_video_dir, exist_ok=True)

gt_files = sorted(glob(os.path.join(gt_dir, "*.png")))

image_list = []

for gt_file in tqdm(gt_files):
    render_file = os.path.join(render_dir, os.path.basename(gt_file))
    gt = cv2.imread(gt_file)
    render = cv2.imread(render_file)
    compare = np.concatenate([gt, render], axis=1)
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

