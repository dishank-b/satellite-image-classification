import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import numpy as np
import pudb
import rasterio
import keras.backend as K

def recall(y_true, y_pred):
    """Recall metric.
     Only computes a batch-wise average of recall.
     Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

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
keras.metrics.recall = recall

X_val = np.load("X.npy")

model = load_model('models/binary_cat_7.h5')
result = model.predict(X_val)
del model
result = result[...,1]

#pu.db
Y_val = np.load("Y.npy")

# pu.db
# result = np.argmax(result_full, axis = 1)

true_list = result == Y_val

acc = float(sum(true_list)) * 100 /sum(Y_val != 8)
print ("overall acc: "+str(acc))

# file = ["8", "9", "14"]	# Adjust this for choosing file name
file = ["1", "2", "3", "4", "5", "6", "7", "10", "11", "12", "13"]	# Adjust this for choosing file name

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
	# pu.db
	for each in result_here:
		# print each
		if each > (np.mean(result_here) + 1.5 * np.std(result_here)):
			r.append(classes["7"][0])
			g.append(classes["7"][1])
			b.append(classes["7"][2])
		else:
			r.append(classes["8"][0])
			g.append(classes["8"][1])
			b.append(classes["8"][2])

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
	plt.imsave("./yellow_"+f+".png",img)

print acc
