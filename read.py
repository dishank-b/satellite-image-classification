import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pudb

classes = {
	'[0. 0. 0.]': 0,
	'[  0.  0. 150.]': 1,
	'[  0. 125.   0.]': 2,
	'[  0. 255.   0.]': 3,
	'[100. 100. 100.]': 4, 
	'[150.  80.   0.]': 5,
	'[150. 150. 255.]': 6,
	'[255. 255.   0.]': 7,
	'[255. 255. 255.]': 8

}

org_file = "data/sat/3.tif"
gt_file = "data/gt/3.tif"

org_data = rasterio.open(org_file)
gt_data = rasterio.open(gt_file)

r, g, b, ir = org_data.read()
gt_r, gt_g, gt_b = gt_data.read()
r, g, b, ir = r.astype(float), g.astype(float), b.astype(float), ir.astype(float)
gt_r, gt_g, gt_b = gt_r.astype(float), gt_g.astype(float), gt_b.astype(float)

r /= np.max(r)
g/= np.max(g)
b/= np.max(b)
ir /= np.max(ir)

r, g, b, ir = r.flatten(), g.flatten(), b.flatten(), ir.flatten()
gt_r, gt_g, gt_b = gt_r.flatten(), gt_g.flatten(), gt_b.flatten()

org = []
gt = []

for i in range(len(r)):
	org.append([r[i], g[i], b[i], ir[i]])
	gt.append([gt_r[i], gt_g[i], gt_b[i]])

org = np.array(org)
gt = np.array(gt)
pu.db

# col_img = P[: , :, 0:3]
# nir_img = P[: , :, 3]
# # pu.db

# col_img = cv2.convertScaleAbs(col_img)
# # nir_img = cv2.convertScaleAbs(nir_img)
# #col_img = cv2.cvtColor(col_img, cv2.COLOR_BGR2RGB)

# # pu.db
# plt.figure(1)
# plt.imshow(col_img)
# plt.show()
# # cv2.waitKey(0)
# plt.figure(2)
# plt.imshow(nir_img, cmap='gray')
# plt.show()