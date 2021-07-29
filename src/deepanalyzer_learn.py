import keras
import glob
import librosa
from scipy.fft import fft
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import pickle

flag_pp = False

# constants
SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds
EPOCHS=20

# load data
COUGH_FOLDER="data/cough_added"
NO_COUGH_FOLDER="data/no_cough"
MANUAL_COUGH="data/manual_cough"

cough_files = glob.glob(COUGH_FOLDER+"/*")
no_cough_files = glob.glob(NO_COUGH_FOLDER+"/*")
manual_cough_files = glob.glob(NO_COUGH_FOLDER+"/*")

shapex = 66000
# shapex = 24347

if flag_pp:
    trainX = np.empty(shape=(shapex,))
    trainY = np.empty(shape=(2,))
    for cf in cough_files[0:1400]:
        x, sr = librosa.load(cf, sr=SAMPLERATE, mono=True)

        x_fft = fft(x)
        N = len(x)
        x_fft = 2.0 / N * np.abs(x_fft[0:N // 2])

        x_in = np.append(x, x_fft)
        trainX = np.vstack((trainX, x_in))

        # fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50, scaling="spectrum", mode="magnitude")
        # x_in = Sxx.flatten()
        # trainX = np.vstack((trainX, x_in))

        trainY = np.vstack((trainY, [0, 1]))

    for ncf in no_cough_files[0:1400]:
        x, sr = librosa.load(ncf, sr=SAMPLERATE, mono=True)

        x_fft = fft(x)
        N = len(x)
        x_fft = 2.0 / N * np.abs(x_fft[0:N // 2])

        x_in = np.append(x, x_fft)
        trainX = np.vstack((trainX, x_in))

        # fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50, scaling="spectrum", mode="magnitude")
        # x_in = Sxx.flatten()
        # trainX = np.vstack((trainX, x_in))

        trainY = np.vstack((trainY, [1, 0]))

    real_testX = np.empty(shape=(shapex,))
    real_testY = np.empty(shape=(2,))
    for mc in manual_cough_files[0:44]:
        x, sr = librosa.load(mc, sr=SAMPLERATE, mono=True)

        x_fft = fft(x)
        N = len(x)
        x_fft = 2.0 / N * np.abs(x_fft[0:N // 2])

        x_in = np.append(x, x_fft)
        real_testX = np.vstack((real_testX, x_in))

        # fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50, scaling="spectrum", mode="magnitude")
        # x_in = Sxx.flatten()
        # real_testX = np.vstack((real_testX, x_in))

        real_testY = np.vstack((real_testY, [0, 1]))

    X_and_label = np.concatenate((trainX, trainY),axis=1)
    cache_dump = {"real_testX": real_testX, "X_and_label": X_and_label}
    pickle.dump(cache_dump, open( "temp.p", "wb"))

cache_dump = pickle.load(open( "temp.p", "rb"))
real_testX = cache_dump["real_testX"]
X_and_label = cache_dump["X_and_label"]
X_and_label = X_and_label[1:,:]
# shuffle input data and labels
np.random.shuffle(X_and_label)

trainX = X_and_label[:,:-2]
trainY = X_and_label[:,-2:]


testX = trainX[1000:1400]
testY = trainY[1000:1400]
trainX = trainX[0:1000]
trainY = trainY[0:1000]

# normalize to [0,1]
trainX = trainX / np.max(trainX)
trainY = trainY / np.max(trainY)
testX = testX / np.max(testX)
testY = testY / np.max(testY)

m1 = keras.Sequential()
m1.add(keras.layers.Dense(1000, input_shape=(X_and_label.shape[1]-2,), activation="sigmoid"))
m1.add(keras.layers.Dropout(rate=0.1))
m1.add(keras.layers.Dense(100, activation="sigmoid"))
m1.add(keras.layers.Dropout(rate=0.1))
m1.add(keras.layers.Dense(2, activation="sigmoid"))

# m1.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(X_and_label.shape[1]-2,)))
# m1.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
# m1.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# m1.add(keras.layers.Dropout(0.25))
# m1.add(keras.layers.Flatten())
# m1.add(keras.layers.Dense(128, activation='relu'))
# m1.add(keras.layers.Dropout(0.5))
# m1.add(keras.layers.Dense(2, activation='softmax'))

# opt=keras.optimizers.SGD(0.01)
opt=keras.optimizers.Adam(learning_rate=0.001)
m1.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

H = m1.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=20)

H2 = m1.predict(real_testX)
print(H2)

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
