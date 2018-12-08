import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
# import pudb
import numpy as np
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

LABEL_HERE = 2

BATCH_SIZE = 100000
MAX_ITER = 1000

X_train = np.load("X.npy")
Y_train = np.load("Y.npy")

# pu.db

X_val = np.load("X_val.npy")
Y_val = np.load("Y_val.npy")

for i in xrange(len(Y_train)):
  if (Y_train[i] != LABEL_HERE):
    Y_train[i] = 0
  else:
    Y_train[i] = 1

for i in xrange(len(Y_val)):
  if (Y_val[i] != LABEL_HERE):
    Y_val[i] = 0
  else:
    Y_val[i] = 1

Y_train = keras.utils.to_categorical(Y_train, num_classes = 2)
Y_val = keras.utils.to_categorical(Y_val, num_classes = 2)

# Y_train = keras.utils.to_categorical(Y_train, num_classes = 9)
# Y_val = keras.utils.to_categorical(Y_val, num_classes = 9)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(128, activation='selu', input_dim=4))
model.add(Dropout(0.3))
model.add(Dense(128, activation='selu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='selu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='selu'))
model.add(Dropout(0.3))
model.add(Dense(9, activation='selu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy', recall])

checkpointer = ModelCheckpoint(monitor='val_loss', filepath="binary_cat_"+str(LABEL_HERE)+".h5", verbose=True,
save_best_only = True)

model.fit(X_train, Y_train,
          epochs=MAX_ITER, shuffle=True, verbose=True, validation_data=(X_val, Y_val),
          batch_size=BATCH_SIZE, callbacks=[checkpointer])
score = model.evaluate(X_val, Y_val, batch_size=BATCH_SIZE)
print(score)
