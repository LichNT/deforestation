import cv2
import numpy as np
from tqdm import tqdm
import sys

# path = "/home/khoi/Desktop/sample_palm.tif"
path = "/home/khoi/Desktop/docs/quangname_basemap_201707_clip_LZW.tif"

big_img = cv2.imread(path)
print(big_img.shape[0], big_img.shape[1])
# sys.exit(0)
width = height = 512
max_y = (big_img.shape[0]) // height
max_x = (big_img.shape[1]) // width

if big_img.shape[0] % height != 0:
    max_y += 1
if big_img.shape[1] % width != 0:
    max_x += 1
print(max_x, max_y)
for i in tqdm(range(max_y)):
    for j in range(max_x):
        img = big_img[i * height: min((i + 1)*height, big_img.shape[0]),
              j * width:min((j + 1) * width, big_img.shape[1])]
        cv2.imwrite("../quangnam_cut/img_{0}-{1}.jpg".format(i, j),img)
