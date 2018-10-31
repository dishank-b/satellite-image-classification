import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import pudb

file = "data/sat/1.tif"
P = tiff.imread(file)
col_img = P[: , :, 0:3]
nir_img = P[: , :, 3]
# pu.db

col_img = cv2.convertScaleAbs(col_img)
nir_img = cv2.convertScaleAbs(nir_img)
col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)

# pu.db
plt.figure(1)
plt.imshow(col_img)
plt.show()
# cv2.waitKey(0)
plt.figure(2)
plt.imshow(nir_img, cmap='gray')
plt.show()