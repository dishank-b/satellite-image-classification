import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pudb

classes = {
	"0": [0, 0, 0],
	"1": [0, 0, 150],
	"2": [0, 125, 0],
	"3": [0, 255, 0],
	"4": [100, 100, 100], 
	"5": [150, 80, 0],
	"6": [150, 150, 255],
	"7": [255, 255, 0],
	"8": [255, 255, 0]
}

model = load_model('check.h5')


X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")
Y_val = keras.utils.to_categorical(Y_val, num_classes = 9)

score = model.evaluate(X_val, Y_val, batch_size=100000)
print(score)

result = model.predict(X_val)
result = np.argmax(result, axis = 1)

# pu.db

r = []
g = []
b = []

for each in result:
	r.append(classes[str(each)][0])
	g.append(classes[str(each)][1])
	b.append(classes[str(each)][2])

r = np.reshape(r, (622, 782))
g = np.reshape(g, (622, 782))
b = np.reshape(b, (622, 782))

img = np.zeros((622, 782, 3))

img[...,0] = r
img[...,1] = g
img[...,2] = b

plt.imshow(img)
plt.show()