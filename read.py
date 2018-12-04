import rasterio
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pudb
import sys

classes = {
	'[0 0 0]': 0,			#black
	'[  0  0 150]': 1,		#blue
	'[  0   0 150]': 1,
	'[  0 125   0]': 2,		#Dark green
	'[  0 255   0]': 3,		#Light green
	'[100 100 100]': 4, 	# Grey
	'[150  80   0]': 5,		# Brown
	'[150 150 255]': 6,		# Light blue/purple
	'[255 255   0]': 7,		# Yellow
	'[255 255 255]': 8		# White

}
org = []
gt = []
# file = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 14]	# Adjust this for choosing file name
file = [8, 9, 13]	# Adjust this for choosing file name

for f in file:
	print(f)
	org_file = "data/sat/"+str(f)+".tif"
	gt_file = "data/gt/"+str(f)+".tif"

	org_data = rasterio.open(org_file)
	gt_data = rasterio.open(gt_file)

	r, g, b, ir = org_data.read()
	gt_r, gt_g, gt_b = gt_data.read()
	r, g, b, ir = r.astype(float), g.astype(float), b.astype(float), ir.astype(float)

	img = np.zeros((r.shape[0], r.shape[1], 3))

	r /= np.max(r)
	g/= np.max(g)
	b/= np.max(b)
	ir /= np.max(ir)

	img[...,0] = r
	img[...,1] = g
	img[...,2] = b

	plt.imshow(img)
	plt.show()

	sys.exit(0)

	r, g, b, ir = r.flatten(), g.flatten(), b.flatten(), ir.flatten()
	gt_r, gt_g, gt_b = gt_r.flatten(), gt_g.flatten(), gt_b.flatten()


	for i in range(len(r)):
		org.append([r[i], g[i], b[i], ir[i]])
		temp = np.array([gt_r[i], gt_g[i], gt_b[i]])
		gt.append(classes[str(temp)])

org = np.array(org)
gt = np.array(gt)
np.save("X_val.npy", org)	# Adjust this for out file name
np.save("Y_val.npy", gt)
# pu.db

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