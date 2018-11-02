import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import numpy as np

BATCH_SIZE = 1000000

X_train = np.load("X.npy")
Y_train = np.load("Y.npy")

X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")

Y_train = keras.utils.to_categorical(Y_train, num_classes = 9)
Y_val = keras.utils.to_categorical(Y_val, num_classes = 9)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=4))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(9, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(monitor='val_loss', filepath="check.h5", verbose=True,
save_best_only = True)

model.fit(X_train, Y_train,
          epochs=20,
          batch_size=BATCH_SIZE, callbacks=[checkpointer])
score = model.evaluate(X_val, Y_val, batch_size=128)
