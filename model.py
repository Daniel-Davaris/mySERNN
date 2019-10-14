import os
import numpy as np
import librosa
import keras
import tensorflow as tf
import keras
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
from librosa import display
import h5py



#  read in the variables from the data_prep.py

h5f = h5py.File('data.h5','r')
train_X = h5f['pre'][:]
train_y = h5f['pre2'][:]
# test_X = h5f['spec3'][:]
# test_y = h5f['spec4'][:]

h5f.close()

print(train_X.shape)
print(train_y.shape)
# print(test_X.shape)
# print(test_y.shape)

# reshape the data into testing and training data and also into singals and lables data 

train_X = np.reshape(train_X, [*train_X.shape, 1])
print(train_X.shape)
test_X = train_X[1000:]
train_X = train_X[:1000]
print(test_X.shape)
test_y = train_y[1000:]
train_y = train_y[:1000]


# the neural network model 
# kernal size of 4 by 4 to fit the data arrays
# inp = Input(shape=train_X[0].shape)   # set the input as the x variable training data 
# m = Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same')(inp)
# m = MaxPooling2D(pool_size=(4, 4))(m)
# m = Dropout(0.2)(m)   #low dropout to moderatly prevent overfitting
# m = Conv2D(64, kernel_size=(4, 4), activation='relu')(m)
# m = MaxPooling2D(pool_size=(4, 4))(m)
# m = Dropout(0.2)(m)     # low dropout to moderatly prevent overfitting 
# m = Conv2D(128, kernel_size=(4, 4), activation='relu')(m)
# m = MaxPooling2D(pool_size=(4, 4))(m)
# m = Dropout(0.2)(m)     # low dropout to moderatly prevent overfitting 
# m = Flatten()(m)
# m = Dense(32, activation='relu')(m)
# out = Dense(10, activation='softmax')(m)   # output our data through a softmax function to average our results and impove accuracy 
# model = Model(input=inp, output=out) # the model will initalize with the input varrible and output with the hidden layers + softmax

model = keras.models.Sequential([
    Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same', input_shape=train_X[0].shape),
    Conv2D(32, kernel_size=(4, 4), activation='relu'),
    Conv2D(32, kernel_size=(4, 4), activation='relu'),
    MaxPooling2D(pool_size=(3, 3)),
    Conv2D(64, kernel_size=(4, 4), activation='relu', padding='same'),
    Conv2D(64, kernel_size=(4, 4), activation='relu'),
    Conv2D(64, kernel_size=(4, 4), activation='relu'),

    MaxPooling2D(pool_size=(3, 3)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.7),
    Dense(64, activation='relu'),
    Dropout(0.7),
    Dense(32, activation='relu'),
    Dropout(0.7),
    Dense(10, activation='softmax'),
])

model.summary()   # prints the model information, nmumber of parrameters ect 


model.compile(loss='sparse_categorical_crossentropy',    # complete this model with the sparse_categorical_crossentropy loss function 
              optimizer='adam',
              metrics=['accuracy'],
)
              
history = model.fit(    # fit the model or train the model with the training variables 
    train_X,
    train_y,
    epochs=100,    # set a high amount of epochs to optimise accruacy 
    batch_size=32,
    validation_data=(test_X, test_y),
    callbacks=[keras.callbacks.callbacks.EarlyStopping(monitor = 'val_loss', patience=2)],   # test the model with the testing data variables 
)

model.save("model.h5")    # save the model with the weights asciated so we only have to run it once and can easily apply it to predictions.py
 
# model.evaluate(x=X_test, y=Y_test)


# plot the results of the model after it is finished training 
plt.title("Accuracy")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.title("Loss")
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
