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

	# img = np.zeros((r.shape[0], r.shape[1], 3))

	# r /= np.max(r)
	# g/= np.max(g)
	# b/= np.max(b)
	# ir /= np.max(ir)

	# img[...,0] = r
	# img[...,1] = g
	# img[...,2] = b

	# plt.imshow(img)
	# plt.show()

	# sys.exit(0)

	# r_pad = np.pad(r, 1, 'edge')
	# g_pad = np.pad(g, 1, 'edge')
	# b_pad = np.pad(b, 1, 'edge')
	# ir_pad = np.pad(ir, 1, 'edge')

	num_rows, num_cols = r.shape[0], r.shape[1]
	r, g, b, ir = r.flatten(), g.flatten(), b.flatten(), ir.flatten()

	sigma = np.std(r)
	meu = np.mean(r)
	r += meu - 2*sigma
	r = r / (4*sigma)
	r[r>1.0] = 1.0
	r[r<0.0] = 0.0

	sigma = np.std(g)
	meu = np.mean(g)
	g += meu - 2*sigma
	g = g / (4*sigma)
	g[g>1.0] = 1.0
	g[g<0.0] = 0.0

	sigma = np.std(b)
	meu = np.mean(b)
	b += meu - 2*sigma
	b = b / (4*sigma)
	b[b>1.0] = 1.0
	b[b<0.0] = 0.0

	sigma = np.std(ir)
	meu = np.mean(ir)
	ir += meu - 2*sigma
	ir = ir / (4*sigma)
	ir[ir>1.0] = 1.0
	ir[ir<0.0] = 0.0

	# pu.db
	# r_pad, g_pad, b_pad, ir_pad = r_pad.flatten(), g_pad.flatten(), b_pad.flatten(), ir_pad.flatten()
	gt_r, gt_g, gt_b = gt_r.flatten(), gt_g.flatten(), gt_b.flatten()


	for i in range(len(r)):
		try:
			if (i - 1 >= 0):
				org.append([r[i], g[i], b[i], ir[i],
						r[i - 1], g[i - 1], b[i - 1], ir[i - 1],
						r[i + 1], g[i + 1], b[i + 1], ir[i + 1],
						r[i - num_cols], g[i - num_cols], b[i - num_cols], ir[i - num_cols],
						r[i + num_cols], g[i + num_cols], b[i + num_cols], ir[i + num_cols],
						r[i - num_cols - 1], g[i - num_cols - 1], b[i - num_cols - 1], ir[i - num_cols - 1],
						r[i - num_cols + 1], g[i - num_cols + 1], b[i - num_cols + 1], ir[i - num_cols + 1],
						r[i + num_cols - 1], g[i + num_cols - 1], b[i + num_cols - 1], ir[i + num_cols - 1],
						r[i + num_cols + 1], g[i + num_cols + 1], b[i + num_cols + 1], ir[i + num_cols + 1]])
			else:
				print("Noes 1")
				org.append([r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i],
						r[i], g[i], b[i], ir[i]])
		except:
			print("Noes 2")
			org.append([r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i],
					r[i], g[i], b[i], ir[i]])
					
		temp = np.array([gt_r[i], gt_g[i], gt_b[i]])
		gt.append(classes[str(temp)])

org = np.array(org)
gt = np.array(gt)
np.save("X_nb_val.npy", org)	# Adjust this for out file name
np.save("Y_nb_val.npy", gt)
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