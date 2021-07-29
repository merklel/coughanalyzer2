import keras
import glob
import librosa
from scipy.fft import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as cl
import matplotlib
#from skimage.color import rgb2gray
import os
from keras.preprocessing.image import ImageDataGenerator
# from tensorflow.python.keras.callbacks import TensorBoard
from sklearn.utils import shuffle
import time


# Flag control ############################
flag_pp = False
flag_re_train = True
###########################################



# constants
SAMPLERATE = 16000
CHUNKSIZE = 2 # seconds


FFTS_COUGH = "data/train_ffts/train/ffts_cough.p"
FFTS_NO_COUGH = "data/train_ffts/train/ffts_no_cough.p"
FFTS_CROSSVALID = "data/train_ffts/crossvalid/ffts_crossvalid.p"


## Load from pickle ##############################################################################################
print("* loading data from hdd")

shapeSxx = (16000,)

ffts_cough = pickle.load(open(FFTS_COUGH, "rb"))
ffts_cough = [f[0:4000] for f in ffts_cough]

ffts_cough_Y = [[1,0]]*len(ffts_cough)
ffts_no_cough = pickle.load(open(FFTS_NO_COUGH, "rb"))
ffts_no_cough = [f[0:4000] for f in ffts_no_cough]

ffts_no_cough_Y = [[0,1]]*len(ffts_no_cough)

trainX = np.stack(ffts_cough + ffts_no_cough)
trainY = np.stack(ffts_cough_Y + ffts_no_cough_Y)

ffts_crossvalid = pickle.load(open(FFTS_CROSSVALID, "rb"))
print("n crossvalids: ", len(ffts_crossvalid))
ffts_crossvalid = [fcv for fcv in ffts_crossvalid if fcv.shape == shapeSxx]
ffts_crossvalid = [f[0:4000] for f in ffts_crossvalid]
print("n crossvalids after filter: ", len(ffts_crossvalid))

crossvalids_testX = np.stack(ffts_crossvalid)

# shuffle train files
trainX, trainY = shuffle(trainX, trainY)


print("* slicing train/test images")
n_test = 0.2
idx_train_test = int(n_test*len(trainX))


shapeSxx = (4000,)


testX = trainX[0:idx_train_test]
testY = trainY[0:idx_train_test]
trainX = trainX[idx_train_test:]
trainY = trainY[idx_train_test:]

print("shape testX: ", testX.shape)
print("shape testY: ", testY.shape)
print("shape trainX: ", trainX.shape)
print("shape trainY: ", trainY.shape)

#standardize
#print("* standardize images")
#trainX = (trainX - np.mean(trainX)) / np.std(trainX)
#testX = (testX -np.mean(testX)) / np.std(testX)
#untouched_testX = (untouched_testX - np.mean(untouched_testX)) / np.std(untouched_testX)
#real_testX = (real_testX - np.mean(real_testX)) / np.std(real_testX)
#crossvalids_testX = (crossvalids_testX - np.mean(crossvalids_testX)) / np.std(crossvalids_testX)

# reshape to fit conv2D
print("* reshape images")
trainX = trainX.reshape(trainX.shape + (1,))
testX = testX.reshape(testX.shape + (1,))
crossvalids_testX = crossvalids_testX.reshape(crossvalids_testX.shape + (1,))

#kernel_constraint = keras.constraints.UnitNorm(axis=0)
kernel_constraint = None

activity_reg = keras.regularizers.l2(0.006)
kernel_reg = keras.regularizers.l2(0.0001)
b_reg=keras.regularizers.l2(0.0001)

activity_reg=None
kernel_reg=None
b_reg=None

EPOCHS=40
BATCHSIZE=100
da = 0
m1 = keras.Sequential()

activation = "relu"

m1.add(keras.layers.Conv1D(filters=20, kernel_size=(100), input_shape=(4000,1)))
m1.add(keras.layers.Conv1D(filters=20, kernel_size=(20)))

m1.add(keras.layers.Flatten())
#m1.add(keras.layers.Dense(3000, input_shape=shapeSxx, activation=activation, activity_regularizer=activity_reg, bias_regularizer=b_reg, kernel_regularizer=kernel_reg, kernel_constraint=kernel_constraint))
#m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(50, activation=activation, activity_regularizer=activity_reg, bias_regularizer=b_reg, kernel_regularizer=kernel_reg, kernel_constraint=kernel_constraint))
m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(50, activation=activation, activity_regularizer=activity_reg, bias_regularizer=b_reg, kernel_regularizer=kernel_reg, kernel_constraint=kernel_constraint))
m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(50, activation=activation, activity_regularizer=activity_reg, bias_regularizer=b_reg, kernel_regularizer=kernel_reg, kernel_constraint=kernel_constraint))
m1.add(keras.layers.Dropout(da))
# m1.add(keras.layers.Dense(5, activation=activation, activity_regularizer=activity_reg, bias_regularizer=b_reg, kernel_regularizer=kernel_reg))
# m1.add(keras.layers.Dropout(da))
# # m1.add(keras.layers.Dense(5, activation=activation, activity_regularizer=activity_reg, bias_regularizer=b_reg, kernel_regularizer=kernel_reg))
# m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(2, activation='softmax'))
m1.summary()


#opt=keras.optimizers.SGD(learning_rate=0.03)
#opt=keras.optimizers.Adam(lr=0.003)
opt=keras.optimizers.Adam(lr=0.001)
#opt=keras.optimizers.Adagrad()
#opt=keras.optimizers.Nadam()
m1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

if flag_re_train:
#if False:
    H = m1.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BATCHSIZE, callbacks=[])
    cache_dump = {"H": H,"m1": m1}
    pickle.dump(cache_dump, open( "model_2.p", "wb"))

cache_dump = pickle.load(open( "model_2.p", "rb"))
m1 = cache_dump["m1"]
H = cache_dump["H"]

# plot the training loss and accuracy
plt.style.use("ggplot")
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["categorical_accuracy"], label="train_acc")
plt.plot(N, H.history["val_categorical_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.ylim([0,1])
plt.legend()
plt.savefig("erg.pdf")

#### predict real manually found cough
# print("##### predict manually found cough #########")
# H2 = m1.test_on_batch(real_testX, real_testY)
# print(m1.metrics_names)
# print(H2)
# for idx, line in enumerate(m1.predict(real_testX)):
#     print(idx, line)


# print("##### predict unbearbeitet #########")
# # H3 = m1.predict(untouched_testX)
# # print(H3)
# f=open("untouched_labels.txt", "w")
# for idx, line in enumerate(m1.predict(untouched_testX)):
#     print(idx, line)
#     if line[1] > 0.5:
#         f.write("{}\t{}\t{}\n".format(idx*2, (idx*2)+2, line[1]))
# f.close()

print("##### predict crossvalid #########")
# H3 = m1.predict(untouched_testX)
# print(H3)
f=open("untouched_labels_crossvalid.txt", "w")
for idx, line in enumerate(m1.predict(crossvalids_testX)):
    print(idx, line)
    #if line[1] > 0.6:
    f.write("{}\t{}\t{}\t{}\n".format(crossvalid_chunk_numbers[idx]*CHUNKSIZE, (crossvalid_chunk_numbers[idx]*CHUNKSIZE)+2, line[0], line[1]))
f.close()




