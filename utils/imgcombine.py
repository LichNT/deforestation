import sys
from PIL import Image
import os
from tqdm import tqdm

offset = 512
# quangnam
rows, columns = 53, 72
total_width = (rows - 1) * 512 + 380
total_height = (columns - 1) * 512 + 268
# clip02
# rows, columns = 12, 17
# total_width = (rows - 1) * 512 + 64
# total_height = (columns - 1) * 512 + 28

# clip01
# rows, columns = 6,8
# total_width = (rows - 1) * 512 + 460
# total_height = (columns - 1) * 512 + 272

# clip03
# rows, columns = 15,15
# total_width = (rows - 1) * 512 + 496
# total_height = (columns - 1) * 512 + 328

# clip04
# rows, columns = 16,15
# total_width = (rows - 1) * 512 + 412
# total_height = (columns - 1) * 512 + 144

new_im = Image.new('L', (total_height, total_width))

for i in tqdm(range(rows), unit=" rows"):
    for j in range(columns):
        im = Image.open("../quangnam/diff_th/img_{0}-{1}.jpg".format(i, j))
        new_im.paste(im, (j * offset, i * offset))
new_im.save('../quangnam/diff.tif')
