import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

model = Sequential()
model.add(Conv2D(48, kernel_size=(3, 3),
                 input_shape=input_shape))

# First DB
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same'))
model.add(Dropout(0.2))


model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3), padding='same'))
model.add(Dropout(0.2))


model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(112, (3, 3), padding='same'))
model.add(Dropout(0.2))

# First TD
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(112, (1, 1), padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second DB
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(144, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(160, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(176, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Dropout(0.2))

# Second TD
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (1, 1), padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Bottle neck
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(208, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(224, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(240, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(272, (3, 3), padding='same'))
model.add(Dropout(0.2))

# Second TU
model.add(Conv2DTranspose(272, (3, 3), stride = 2))

# Second DB
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(240, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(224, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(208, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3), padding='same'))
model.add(Dropout(0.2))

# First TU
model.add(Conv2DTranspose(192, (3, 3), stride = 2))

# First DB
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(176, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(160, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(144, (3, 3), padding='same'))
model.add(Dropout(0.2))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Dropout(0.2))

# Get classes

model.add(Conv2D(8, (3, 3), padding='same'))



model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
