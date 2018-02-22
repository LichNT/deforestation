import cv2
import os
directory = "../quangnam_data"
files = [f for f in os.listdir(directory) if "mask" in f]
for f in files:
    print(f)
    img = cv2.imread(directory+"/{}".format(f))
    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(directory+"/{}".format(f), img[:, :, 0])
