from PIL import Image
import numpy as np
from infer import to_rgb
import os
import cv2
from tqdm import tqdm

# imgs = [np.asarray(Image.open("../after.jpg")),
#         np.asarray(Image.open("../before.jpg"))]
# imgs = list(map(lambda x: (x - np.min(x)) / np.max(x), imgs))
# diff = to_rgb(imgs[0] - imgs[1])[..., 0]
# # diff = to_rgb(1 - imgs[1]*(1 - imgs[0]))[..., 0]
# result = Image.fromarray(diff.round().astype(np.uint8))
# result.save("../hehe2.jpg")

dirs = ["../quangnam_t7", "../quangnam_t9"]
list_files = [os.listdir(d) for d in dirs]
assert len(list_files[0]) == len(list_files[1])

for l in list_files[0]:
    before = np.asarray(Image.open(os.path.join(dirs[0], l)))[..., 0].astype(np.float)
    after = np.asarray(Image.open(os.path.join(dirs[1], l)))[..., 0].astype(np.float)
    before = to_rgb(before)/255.0
    before[np.isnan(before)] = 0
    after = to_rgb(after)/255.0
    after[np.isnan(after)] = 0
    diff = to_rgb(1 - before*(1 - after))
    diff[np.isnan(diff)] = 255
    # diff[diff > 70] = 255

    diff = Image.fromarray(diff.round().astype(np.uint8))
    diff.save("../quangnam_diff/{}".format(l))
