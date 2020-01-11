import keras
import glob
import librosa
from scipy.fft import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Flag control ############################
flag_pp = True
flag_re_train = True
###########################################

# constants
SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds

# load data
COUGH_FOLDER="data/cough_added"
NO_COUGH_FOLDER="data/no_cough"
MANUAL_COUGH="data/manual_cough"

cough_files = glob.glob(COUGH_FOLDER+"/*")
no_cough_files = glob.glob(NO_COUGH_FOLDER+"/*")
manual_cough_files = glob.glob(NO_COUGH_FOLDER+"/*")

train_files = cough_files[0:1300] + no_cough_files[0:1300]
np.random.shuffle(train_files)

shapex = 66000
# shapex = 24347
shapeSxx = (251, 97, 1)

if flag_pp:
    trainY = np.empty(shape=(2,))
    trainXs = []

    ################## Syntetic cough training ####################################################
    for ncf in train_files:
        x, sr = librosa.load(ncf, sr=SAMPLERATE, mono=True)

        # x_fft = fft(x)
        # N = len(x)
        # x_fft = 2.0 / N * np.abs(x_fft[0:N // 2])
        #
        # x_in = np.append(x, x_fft)
        # trainX = np.vstack((trainX, x_in))

        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50, scaling="spectrum", mode="magnitude")
        # x_in = Sxx.flatten()
        # trainX = np.vstack((trainX, x_in))
        trainXs.append(Sxx)

        if "no_cough" in ncf:
            truth = [1, 0]
        if "cough_added" in ncf:
            truth = [0, 1]

        trainY = np.vstack((trainY, truth))
    trainX = np.stack(trainXs)


    ################## Manually found cough ####################################################
    # real_testX = np.empty(shape=(shapex,))
    real_testXs = []
    real_testYs = []
    for mc in manual_cough_files[0:44]:
        x, sr = librosa.load(mc, sr=SAMPLERATE, mono=True)

        # x_fft = fft(x)
        # N = len(x)
        # x_fft = 2.0 / N * np.abs(x_fft[0:N // 2])
        #
        # x_in = np.append(x, x_fft)
        # real_testX = np.vstack((real_testX, x_in))

        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50, scaling="spectrum", mode="magnitude")
        # x_in = Sxx.flatten()
        # real_testX = np.vstack((real_testX, x_in))

        #real_testY = np.vstack((real_testY, [0, 1]))
        real_testYs.append([0, 1])
        real_testXs.append(Sxx)
    real_testX = np.stack(real_testXs)
    real_testY = np.stack(real_testYs)

    unbearbeitet = []
    for af in no_cough_files[0:1400]:
        x, sr = librosa.load(af, sr=SAMPLERATE, mono=True)
        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50,scaling="spectrum", mode="magnitude")
        unbearbeitet.append(Sxx)
    unbeareitet_testX  = np.stack(unbearbeitet)

    # X_and_label = np.concatenate((trainX, trainY),axis=1)
    cache_dump = {"real_testX": real_testX,"real_testY":real_testY, "trainX": trainX, "trainY": trainY, "unbeareitet_testX": unbeareitet_testX}
    pickle.dump(cache_dump, open("temp2.p", "wb"))

cache_dump = pickle.load(open("temp3.p", "rb"))
real_testX = cache_dump["real_testX"]
trainX = cache_dump["trainX"]
trainY = cache_dump["trainY"]
unbeareitet_testX = cache_dump["unbeareitet_testX"]


#testX = trainX[1700:2600]
#testY = trainY[1700:2600]
#trainX = trainX[0:1700]
#trainY = trainY[0:1700]

shapeSxx = (251, 97, 1)

#shapeSxx = (151, 60, 1)
#trainX = trainX[:,100:251,20:80]

testX = trainX[2200:2600]
testY = trainY[2200:2600]
trainX = trainX[0:2200]
trainY = trainY[0:2200]

# normalize to [0,1]
#trainX = trainX / np.max(trainX)
#trainY = trainY / np.max(trainY)
#testX = testX / np.max(testX)
#testY = testY / np.max(testY)

#standardize
trainX = (trainX - np.mean(trainX)) / np.std(trainX)
testX = (testX -np.mean(testX)) / np.std(testX)

unbeareitet_testX = (unbeareitet_testX - np.mean(unbeareitet_testX)) / np.std(unbeareitet_testX)
real_testX = (real_testX - np.mean(real_testX)) / np.std(real_testX)

# reshape to fit conv2D
trainX = trainX.reshape(trainX.shape + (1,))
testX = testX.reshape(testX.shape + (1,))
real_testX = real_testX.reshape(real_testX.shape + (1,))
unbeareitet_testX = unbeareitet_testX.reshape(unbeareitet_testX.shape + (1,))

EPOCHS=3

m1 = keras.Sequential()
m1.add(keras.layers.Conv2D(8, kernel_size=(3,3), activation='relu', input_shape=shapeSxx)) # activity_regularizer=keras.regularizers.l2(0.01)
m1.add(keras.layers.Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=shapeSxx)) # activity_regularizer=keras.regularizers.l2(0.01)
m1.add(keras.layers.Dropout(0.2))
#m1.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))

#m1.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
#m1.add(keras.layers.BatchNormalization())
#m1.add(keras.layers.Conv2D(32, (6, 6), activation='relu')),
#m1.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
m1.add(keras.layers.Flatten())
#m1.add(keras.layers.Dense(50, activation='relu'))
#m1.add(keras.layers.Dropout(0.7))
m1.add(keras.layers.Dense(2, activation='softmax'))
m1.summary()

#opt=keras.optimizers.SGD(0.01)
opt=keras.optimizers.Adam(learning_rate=0.0001)
#opt=keras.optimizers.Adagrad()
#opt=keras.optimizers.Nadam()
m1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

if flag_re_train:
#if False:
    H = m1.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=30)

    cache_dump = {"H": H,"m1": m1}
    pickle.dump(cache_dump, open( "model_2.p", "wb"))

cache_dump = pickle.load(open( "model_2.p", "rb"))
m1 = cache_dump["m1"]
H = cache_dump["H"]

#### predict real manually found cough
print("##### predict manually found cough #########")
H2 = m1.test_on_batch(real_testX, real_testY)
print(m1.metrics_names)
print(H2)
for idx, line in enumerate(m1.predict(real_testX)):
    print(idx, line)


print("##### predict unbearbeitet #########")
H3 = m1.predict(unbeareitet_testX)
print(H3)

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
plt.legend()
plt.savefig("erg.pdf")
plt.show()

