import os
import numpy as np
import librosa
import keras
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten
import matplotlib.pyplot as plt
from librosa import display
import h5py
from sklearn.preprocessing import MinMaxScaler

def pad(wav,const):    
    if wav.shape[0] > const:    
        return wav[0: const]          
    else:   
        return np.hstack((wav, np.zeros(const - wav.shape[0])))   


def lay2(wav, const):                                                            
    if wav.shape[1] > const:
        return wav[:, 0: const] 
    else: 
        return np.hstack((wav, np.zeros((wav.shape[0],const - wav.shape[1]))))

a_p = []
a_digit = []
spec_a = []
a_sp = []


# load in the data directory again 
data_dir = 'C:\\Users\\danie\\Dropbox\\classes\\programming\\conv2\\wav_emotions\\'

# initalize a random file for testing 
ss = '03-01-03-02-02-02-01.wav'
# for i in os.listdir(data_dir):  

    # if i == '03-01-08-02-02-02-01.wav':

# convert that file into wav form 
A, t = librosa.load(data_dir + "Actor_01\\"+ss)
# struct = i.split('_')    
# digit = struct[0]
# pad that file 
a_p.append(pad(A, 30000))
# a_digit.append(struct[0])

# convert that file into a foruier signal 
spec_a = np.abs(librosa.stft(A))

# padd the fourier signal
a_sp.append(lay2(spec_a,40))

# reshape the signal so we can test in on the neural network
train_X = np.reshape(a_sp, (1,1025,40,1))

print("sh",train_X.shape)
# print('digit',digit)


# model = keras.models.load_model("model.h5")


# initalize an array the represent each emotion  
categories = ['0','1','2','3','4','5','6','7']

# h5f = h5py.File('data.h5','r')
# pred = h5f['spec5'][:]
# h5f.close()
# print(pred.shape)
# Xnew = scalar.transform(v)
# ynew = model.predict(pred)

# lpoad in the model from the model.5 , i.e . the model we just trained,.. load it in as a varable
model = tf.keras.models.load_model('model.h5')
# use the keras predict function to run the model on our random file
prediction = model.predict([train_X])
# output the number in the emotions array the coresponds to the index of the prediction result. 
print(categories[int(prediction[0][0])])
print(prediction[0])
