#
# for testing 





# import os
# import numpy as np
# import librosa
# import keras
# import tensorflow as tf
# from keras.layers import Input, Dense
# from keras.models import Model
# import matplotlib.pyplot as plt
# from librosa import display
# import h5py




# a_list = []
# b_list = []
# # c_list = []
# # d_list = []

# emotion1 = []
# emotion2 = []
# # emotion3 = []
# # emotion4 = []



# a_p = []
# a_sp = []
# a_pms = []
# a_pmfcc = []

# b_p = []
# b_sp = []
# b_pms = []
# b_pmfcc = []

# # c_p = []
# # c_sp = []
# # c_pms = []
# # c_pmfcc = []

# # d_p = []
# # d_sp = []
# # d_pms = []
# # d_pmfcc = []

# # count = []

# # data_dir = 'C:\\Users\\danie\\Desktop\\speech-emotion-recognition-master\\start\\free-spoken-digit-dataset\\recordings\\'
# # data_dir = 'C:\\Users\\noahd\\Desktop\\conv\\free-spoken-digit-dataset\\recordings\\'
# data_dir = 'C:\\Users\\danie\\Dropbox\\classes\\programming\\conv2\\wav_emotions\\'

# # # v, t = librosa.load(data_dir + '1_jackson_1.wav')
# # # spec = librosa.amplitude_to_db(np.abs(librosa.stft(v)), ref=np.max)
# # spec = librosa.amplitude_to_db(np.abs(librosa.stft(v)), ref=np.max)
# # # librosa.display.specshow(spec, y_axis='linear')
# # librosa.display.specshow(spec, y_axis='linear')
# # plt.show()


# def pad(wav,const):    
#     if wav.shape[0] > const:    
#         return wav[0: const]          
#     else:   
#         return np.hstack((wav, np.zeros(const - wav.shape[0])))   


# def lay2(wav, const):                                                            
#     if wav.shape[1] > const:
#         return wav[:, 0: const] 
#     else: 
#         return np.hstack((wav, np.zeros((wav.shape[0],const - wav.shape[1]))))



# for i in os.listdir(data_dir):    
#     for x in os.listdir(data_dir + i):
#         struct = x.split('-')   
#         emotion = struct[2]
#         # speaker = struct[6].split('.')
#         # speaker = speaker[0]

#         # if int(speaker) < 18:
#         print(emotion)
#         A,sr = librosa.load(data_dir + i +'\\' + x)  
#         a_p.append(pad(A, 30000))
#         emotion1.append(struct[2])
#         spec_a = np.abs(librosa.stft(A))
#         a_sp.append(lay2(spec_a,40))
#         # mel_spec_a = librosa.feature.melspectrogram(A)
#         # a_pms.append(lay2(mel_spec_a,40))
#         # mfcc_a = librosa.feature.mfcc(A)
#         # a_pmfcc.append(lay2(mfcc_a,40))
#         # else:
#             # B, sr2 = librosa.load(data_dir + i +'\\' + x)
#             # b_p.append(pad(B, 30000))
#             # emotion2.append(struct[2])
#             # spec_b = np.abs(librosa.stft(B))
#             # b_sp.append(lay2(spec_b,40))
#             # mel_spec_b = librosa.feature.melspectrogram(B)
#             # b_pms.append(lay2(mel_spec_b,40))
#             # mfcc_b = librosa.feature.mfcc(B)
#             # b_pmfcc.append(lay2(mfcc_b,40))
#         # elif struct[1] == 'jackson':
#         #     C, sr3 = librosa.load(data_dir + i)
#         #     c_p.append(pad(C, 30000))
#         #     c_digit.append(struct[0])
#         #     spec_c = np.abs(librosa.stft(C))
#         #     c_sp.append(lay2(spec_c,40))
#         #     mel_spec_c = librosa.feature.melspectrogram(C)
#         #     c_pms.append(lay2(mel_spec_c,40))
#         #     mfcc_c = librosa.feature.mfcc(C)
#         #     c_pmfcc.append(lay2(mfcc_c,40))
#         # elif struct[1] == 'theo':
#         #     D, sr4 = librosa.load(data_dir + i)
#         #     d_p.append(pad(D, 30000))
#         #     d_digit.append(struct[0])
#         #     spec_d = np.abs(librosa.stft(D))
#         #     d_sp.append(lay2(spec_d,40))
#         #     mel_spec_d = librosa.feature.melspectrogram(D)
#         #     d_pms.append(lay2(mel_spec_d,40))
#         #     mfcc_d = librosa.feature.mfcc(D)
#         #     d_pmfcc.append(lay2(mfcc_d,40))
        



# # ANP = np.vstack(a_list)


# # BNP = np.vstack(b_list)
# # BDNP = np.array(emotion2)

# # # CNP = np.vstack(c_list)
# # CDNP =np.array(emotion3)

# # # DNP = np.vstack(d_list)
# # DDNP = np.array(emotion4)


# # train_y = np.concatenate((ADNP,BDNP,CDNP))
# # test_X = np.array(d_sp)


# # spec1 = np.array(a_spec)
# # spec2 = np.array(b_spec)
# # spec3 = np.array(c_spec)
# # spec4 = np.array(d_spec)


# # emp = np.empty([1,])

# # train_X = np.vstack((ANP,BNP,CNP))
# # train_X =  np.concatenate((a_spec, b_spec, c_spec))
# # print("train_X",train_X.shape)


# # train_y = np.concatenate((ADNP,BDNP,CDNP))
# # print("train_y",train_y.shape)

# # test_X = np.vstack((DNP))
# # print("test_X",test_X.shape)
# # test_y = DDNP
# # print("test_y",test_y.shape)

# # print(new_x.shape)

# # print('X:', len(x_list))
# # print('NX', new_x.shape)
# # print('y:', len(y_list))
# # print('NY', new_y.shape)

# # print('A',ANP.shape)
# # print('AD',ADNP.shape)
# # print('B',BNP.shape)
# # print('BD',BDNP.shape)
# # print('C',CNP.shape)
# # print('CD',CDNP.shape)
# # print('D',DNP.shape)
# # print('DD',DDNP.shape)

# # print("spec1",type(a_spec), len(a_spec))
# # print("spec1[0]",type(a_spec), len(a_spec[0]))
# # col = np.hstack((a_spec,b_spec,c_spec))
# # first = np.hstack(a_spec)
# # print("spec2",type(b_spec), len(b_spec))
# # print("spec3",type(c_spec), len(c_spec))
# # print("spec4",type(d_spec), len(d_spec))











# # train_O = np.vstack(a_p)
# # print(train_O.shape)
# # train_s = np.concatenate((a_sp,b_sp,c_sp))
# # print(train_s.shape)
# # train_ms = np.array(a_pms)
# # print(train_ms.shape)
# # train_mfcc = np.array(a_pmfcc)
# # print(train_mfcc.shape)

# train_X = np.array(a_sp)
# train_X = train_X[:1000]
# train_y = np.array(emotion1)
# train_y = train_y[:1000]
# test_X = train_X[1000:]
# test_y = train_y[1000:]


# train_X = np.reshape(train_s, (1440,1025,40,1))
# print(train_X.shape)
# test_X = np.reshape(test_X, (500,1025,40,1))
# print(test_X.shape)
# train_y = np.reshape(train_y.astype(int), (1500,1))
# print(train_y.shape)
# test_y = np.reshape(DDNP.astype(int), (500,1))
# print(test_y.shape)
# # prediction = train_X[199:200]

# os.remove('data.h5')
# h5f = h5py.File('data.h5', 'w-')
# h5f.create_dataset('spec1', data=train_X)
# h5f.create_dataset('spec2', data=train_y)
# h5f.create_dataset('spec3', data=test_X)
# h5f.create_dataset('spec4', data=test_y)
# h5f.create_dataset('spec5', data=prediction)

# h5f.close()



