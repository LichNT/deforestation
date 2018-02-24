from PIL import Image
import numpy as np
import cv2
import time
from tqdm import tqdm
import os

img_dir = "../quangnam/diff"
paths = os.listdir(img_dir)
start = time.time()

for path in paths:
    grey_threshold = 50
    size_threshold = 1000
    img = cv2.imread(os.path.join(img_dir, path))[..., 0]
    img = (img < grey_threshold).astype(np.uint8)

    kernel = np.ones((11, 11), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    sum_list = stats[..., 4]
    bg_label = np.argmax(sum_list)
    for i in range(num_labels):
        if sum_list[i] > size_threshold and i != bg_label:
            labels[labels == i] = 0
        else:
            labels[labels == i] = 255

    img = (labels > grey_threshold).astype(np.uint8)
    output = cv2.connectedComponentsWithStats(img, connectivity, cv2.CV_32S)
    num_labels = output[0]
    labels2 = output[1]
    stats = output[2]
    sum_list = stats[..., 4]
    bg_label = np.argmax(sum_list)
    for i in range(num_labels):
        if i != bg_label:
            labels[labels2 == i] = 0

    cv2.imwrite("../quangnam/diff_th/{}".format(path), labels)

stop = time.time()

print("Time: {}".format(stop - start))
