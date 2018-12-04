import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import numpy as np
import pudb
import rasterio

classes = {
	"0": [0, 0, 0],			# Black
	"1": [0, 0, 150],		# Blue
	"2": [0, 125, 0],		# Dark Green
	"3": [0, 255, 0],		# Light Green
	"4": [100, 100, 100], 	# Grey
	"5": [150, 80, 0],		# Brown
	"6": [150, 150, 255],	# Light Blue
	"7": [255, 255, 0],		# Yellow
	"8": [255, 255, 255]	# White
}

model = load_model('models/with_ir.h5')


X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
Y_val = keras.utils.to_categorical(Y_val, num_classes = 9)


score = model.evaluate(X_val, Y_val, batch_size=100000)
print(score)

result = model.predict(X_val)
result = np.argmax(result, axis = 1)

# pu.db
file = [str(i) for i in range(14, 15)]	# Adjust this for choosing file name

for f in file:
	print(f)
	gt_file = "data/gt/"+f+".tif"
	gt_data = rasterio.open(gt_file)

	instance = gt_data.read(1)
	shape_ = instance.shape
	size_ = shape_[0] * shape_[1]
	result_here = result[:size_]
	result = result[size_:]

	r = []
	g = []
	b = []

	for each in result_here:
		# print each
		r.append(classes[str(each)][0])
		g.append(classes[str(each)][1])
		b.append(classes[str(each)][2])

	r = np.reshape(r, shape_)
	g = np.reshape(g, shape_)
	b = np.reshape(b, shape_)

	img = np.zeros((shape_[0], shape_[1], 3))

	img[...,0] = r
	img[...,1] = g
	img[...,2] = b
	img = img.astype(np.uint8)
	# pu.db
	plt.imshow(img)
	# plt.show()
	plt.imsave("./"+f+".png",img)