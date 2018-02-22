import sys
from PIL import Image
import os
from tqdm import tqdm
offset = 512
total_width = 52 * 512
total_height = 71 * 512

new_im = Image.new('L', (total_height, total_width))

for i in tqdm(range(52), unit=" rows"):
    for j in range(71):
        im = Image.open("../diff_noarea/img_{0}-{1}.jpg".format(i,j))
        new_im.paste(im, (j*offset, i*offset))
new_im.save('diff.tif')
