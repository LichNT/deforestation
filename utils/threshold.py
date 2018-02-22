from PIL import Image
import numpy as np
import cv2
import time
from tqdm import tqdm
import os

img_dir = "../quangnam_diff"
paths = os.listdir(img_dir)
start = time.time()

for path in paths:
    grey_threshold = 40
    size_threshold = 1000
    img = cv2.imread(os.path.join(img_dir, path))[..., 0]
    img = 255 - ((img < grey_threshold) * 255).astype(np.uint8)

    kernel = np.ones((11, 11), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    sum_list = stats[..., 4]

    bg_label = np.argmax(sum_list)
    bg = []
    for i in range(num_labels):
        if sum_list[i] > size_threshold and i != bg_label:
            labels[labels == i] = 0
        else:
            labels[labels == i] = 255

    cv2.imwrite("../diff_noarea/{}".format(path), img)

stop = time.time()

print("Time: {}".format(stop - start))
