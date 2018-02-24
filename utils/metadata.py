import numpy as np
import rasterio
from rasterio import crs


image_file = "/home/khoi/Desktop/docs/quangname_basemap_201707_clip_LZW.tif"
output_file = "../quangnam/diff.tif"
with rasterio.open(image_file) as src:
    profile = src.profile

with rasterio.open(output_file) as out:
    image = out.read()

output_image = image
affine = profile['transform']
west = affine[2]
north = affine[5]

new_tranform = rasterio.transform.from_origin(west, north, xsize=affine[0], ysize=-affine[4])

print(new_tranform)
profile.update(transform=new_tranform)

profile.update(dtype=rasterio.uint8, count=3, compress='lzw')
profile.update({'width': output_image.shape[2], 'height': output_image.shape[1], 'count': 3})

profile.update(crs=rasterio.crs.CRS.from_string("+proj=longlat +datum=WGS84 +no_defs"))

with rasterio.open(output_file, 'w', **profile) as dst:
    for i in range(output_image.shape[0]):
        dst.write(output_image[i], i + 1)
    print(dst.profile)