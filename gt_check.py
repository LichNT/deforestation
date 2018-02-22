from PIL import Image
import numpy as np
import cv2
import os
import sys

max_x = 52
max_y = 71
subsize = 81
size = 512
gt = cv2.imread("QuangNamRung.tif")
gt = gt[:max_x * subsize, 0 * subsize: max_y * subsize, 0]
threshold = 20

cv2.imwrite("gt_clip.tif", gt)

diff_path = "diff_noarea"
for path in os.listdir(diff_path):
    img = cv2.imread(os.path.join(diff_path, path))
    x, y = list(map(lambda _: int(_), path[4: -4].split("-")))
    if x == 52 or y == 71:
        continue
    sub_gt = gt[x * subsize:(x + 1) * subsize, y * subsize: (y + 1) * subsize]
    sub_gt = cv2.resize(sub_gt, (size, size))
    img[sub_gt < 50] = 255
    cv2.imwrite(os.path.join(diff_path, path), img)
