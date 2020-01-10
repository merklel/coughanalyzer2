import keras
import glob
import librosa
from scipy.fft import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Flag control ############################
flag_pp = False
flag_re_train = True
###########################################

# constants
SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds
EPOCHS=10

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
    real_testY = np.empty(shape=(2,))
    real_testXs = []
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

        real_testY = np.vstack((real_testY, [0, 1]))
        real_testXs.append(Sxx)
    real_testX = np.stack(real_testXs)

    unbearbeitet = []
    for af in no_cough_files[0:1400]:
        x, sr = librosa.load(af, sr=SAMPLERATE, mono=True)
        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50,scaling="spectrum", mode="magnitude")
        unbearbeitet.append(Sxx)
    unbeareitet_testX  = np.stack(unbearbeitet)

    # X_and_label = np.concatenate((trainX, trainY),axis=1)
    cache_dump = {"real_testX": real_testX, "trainX": trainX, "trainY": trainY, "unbeareitet_testX": unbeareitet_testX}
    pickle.dump(cache_dump, open("temp2.p", "wb"))

cache_dump = pickle.load(open("temp2.p", "rb"))
real_testX = cache_dump["real_testX"]
trainX = cache_dump["trainX"]
trainY = cache_dump["trainY"]
unbeareitet_testX = cache_dump["unbeareitet_testX"]


testX = trainX[2000:2600]
testY = trainY[2000:2600]
trainX = trainX[0:2000]
trainY = trainY[0:2000]

# normalize to [0,1]
trainX = trainX / np.max(trainX)
trainY = trainY / np.max(trainY)
testX = testX / np.max(testX)
testY = testY / np.max(testY)

unbeareitet_testX = unbeareitet_testX / np.max(unbeareitet_testX)
real_testX = real_testX / np.max(real_testX)

# reshape to fit conv2D
trainX = trainX.reshape(trainX.shape + (1,))
testX = testX.reshape(testX.shape + (1,))
real_testX = real_testX.reshape(real_testX.shape + (1,))
unbeareitet_testX = unbeareitet_testX.reshape(unbeareitet_testX.shape + (1,))

m1 = keras.Sequential()
# m1.add(keras.layers.Dense(1000, input_shape=(X_and_label.shape[1]-2,), activation="sigmoid"))
# m1.add(keras.layers.Dropout(rate=0.1))
# m1.add(keras.layers.Dense(100, activation="sigmoid"))
# m1.add(keras.layers.Dropout(rate=0.1))
# m1.add(keras.layers.Dense(2, activation="sigmoid"))

m1.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=shapeSxx))
m1.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
m1.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
m1.add(keras.layers.Dropout(0.25))
m1.add(keras.layers.Flatten())
m1.add(keras.layers.Dense(128, activation='relu'))
m1.add(keras.layers.Dropout(0.5))
m1.add(keras.layers.Dense(2, activation='softmax'))

# opt=keras.optimizers.SGD(0.01)
opt=keras.optimizers.Adam(learning_rate=0.001)
m1.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

if flag_re_train:
    H = m1.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=50)

    cache_dump = {"H": H,"m1": m1}
    pickle.dump(cache_dump, open( "model_2.p", "wb"))

cache_dump = pickle.load(open( "model_2.p", "rb"))
m1 = cache_dump["m1"]
H = cache_dump["H"]

##### predict real manually found cough
H2 = m1.predict(real_testX)
print(H2)


print("##### predict unbearbeitet")
H3 = m1.predict(unbeareitet_testX)
print(H3)

# plot the training loss and accuracy
plt.style.use("ggplot")
N = np.arange(0, EPOCHS)
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
