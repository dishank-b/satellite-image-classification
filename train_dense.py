import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Activation, BatchNormalization
from keras import backend as K
from keras.layers import Input, Concatenate
from keras.models import Model

batch_size = 128
num_classes = 10
epochs = 12

inputs = Input()
x = Conv2D(48, kernel_size=(3, 3), input_shape=input_shape)(inputs)

# First DB
x1 = BatchNormalization()(x)
x1 = Activation('relu')(x1)
x1 = Conv2D(64, (3, 3), padding='same')(x1)
x1 = Dropout(0.2)(x1)

x2 = Concatenate(x, x1)

x2 = BatchNormalization()(x)
x2 = Activation('relu')(x2)
x2 = Conv2D(64, (3, 3), padding='same')(x2)
x2 = Dropout(0.2)(x2)
x3 = Concatenate(x, x2)

# First TD
x3 = BatchNormalization()(x3)
x3 = Activation('relu')(x3)
x3 = Conv2D(112, (1, 1), padding='same')(x3)
x3 = Dropout(0.2)(x3)
x3 = MaxPooling2D(pool_size=(2, 2))(x3)

# Second DB
x4 = BatchNormalization()(x3)
x4 = Activation('relu')(x4)
x4 = Conv2D(128, (3, 3), padding='same')(x4)
x4 = Dropout(0.2)(x4)
x4 = Concatenate(x3, x4)

x5 = BatchNormalization()(x4)
x5 = Activation('relu')(x5)
x5 = Conv2D(128, (3, 3), padding='same')(x5)
x5 = Dropout(0.2)(x5)
x5 = Concatenate(x3, x5)

x6 = Concatenate(x3, x5)


# Second TD
x7 = BatchNormalization()(x6)
x7 = Activation('relu')(x7)
x7 = Conv2D(192, (1, 1), padding='same')(x7)
x7 = Dropout(0.2)(x7)
x7 = MaxPooling2D(pool_size=(2, 2))(x7)

# Bottle neck
x8 = BatchNormalization()(x7)
x8 = Activation('relu')(x8)
x8 = Conv2D(208, (3, 3), padding='same')(x8)
x8 = Dropout(0.2)(x8)

# Second TU
x8 = Conv2DTranspose(272, (3, 3), stride = 2)(x8)

# Second DB

x9 = Concatenate(x6, x9)
x9 = BatchNormalization()(x9)
x9 = Activation('relu')(x9)
x9 = Conv2D(128, (3, 3), padding='same')(x9)
x9 = Dropout(0.2)(x9)
x9 = Concatenate(x8, x9)

x10 = BatchNormalization()(x9)
x10 = Activation('relu')(x10)
x10 = Conv2D(128, (3, 3), padding='same')(x10)
x10 = Dropout(0.2)(x10)

# First TU
x11 = Conv2DTranspose(192, (3, 3), stride = 2)(x10)

# First DB

x11 = Concatenate(x3, x11)
x12 = BatchNormalization()(x11)
x12 = Activation('relu')(x12)
x12 = Conv2D(128, (3, 3), padding='same')(x12)
x12 = Dropout(0.2)(x12)
x13 = Concatenate(x11, x12)

x13 = BatchNormalization()(x13)
x13 = Activation('relu')(x13)
x13 = Conv2D(128, (3, 3), padding='same')(x13)
x13 = Dropout(0.2)(x13)

# Get classes
x14 = Conv2D(8, (3, 3), padding='same')(x13)



x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
