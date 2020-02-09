import keras
import glob
import librosa
from scipy.fft import fft
from scipy import signal
import imageio
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.colors as cl
import matplotlib
from skimage.color import rgb2gray


# Flag control ############################
flag_pp = False
flag_re_train = True
###########################################

# caches
#traindata = "temp3_goodfit.p"
#traindata = "temp3.p"
traindata = "temp6_0.5.p"
#traindata = "only_cough.p"
traindata = "temp6_0.3.p"
traindata = "temp7_0.3.p"
traindata = "temp8_0.3.p"
traindata = "temp9_0.6.p"
# shapeSxx = (251, 88, 1)
traindata = "temp10_0.3.p"
traindata="cough_vs_music.p"
traindata = "temp11_0.3.p"
traindata = "temp12_short_0.3.p"

shapeSxx = (1001, 107, 1)
traindata = "temp13_16khz_short_0.3.p"

shapeSxx = (151, 321, 1)
shapeSxx=(251, 64, 1)

# constants
SAMPLERATE = 16000
CHUNKSIZE = 2 # seconds

# load data
COUGH_FOLDER="data/cough_added"
NO_COUGH_FOLDER="data/no_cough"
MANUAL_COUGH="data/manual_cough"
UNTOUCHED="data/untouched"

IMAGES_TRAIN_FOLDER="data/cough_learn_histo/train"
IMAGES_CROSSVALID_FOLDER="data/cough_learn_histo/crossvalid"

# files to crossvalidate. these have not been used for training
# CROSSVALID = "data/crossvalidation/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009 (152kbit_Opus)"
CROSSVALID = "data/crossvalidation/Beethoven _ Concerto pour piano n°3 "
#CROSSVALID = "data/crossvalidation/Rachmaninoff_ Piano Concerto no"

cough_files = glob.glob(COUGH_FOLDER+"/*.wav")
no_cough_files = glob.glob(NO_COUGH_FOLDER+"/*.wav")
manual_cough_files = glob.glob(NO_COUGH_FOLDER+"/*.wav")
untouched_files = glob.glob(UNTOUCHED+"/*.wav")
crossvalid_files = glob.glob(CROSSVALID+"/*.wav")

images_train = glob.glob(IMAGES_TRAIN_FOLDER+"/*.png")
images_crossvalid = glob.glob(IMAGES_CROSSVALID_FOLDER+"/*.png")

train_files = cough_files + no_cough_files
np.random.shuffle(train_files)

# shapex = 66000
# shapex = 24347
# shapeSxx = (251, 97, 1)

if flag_pp:
    trainYs = []
    trainXs = []

    ################## Syntetic cough training ####################################################
    for idx, ncf in enumerate(train_files):
        x, sr = librosa.load(ncf, sr=SAMPLERATE, mono=True)

        # x_fft = fft(x)
        # N = len(x)
        # x_fft = 2.0 / N * np.abs(x_fft[0:N // 2])
        #
        # x_in = np.append(x, x_fft)
        # trainX = np.vstack((trainX, x_in))

        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0, scaling="spectrum", mode="magnitude")
        f, axs = plt.subplots(1,1)
        #axs.pcolormesh(txx, fxx, Sxx, norm=cl.LogNorm())

        # x_in = Sxx.flatten()
        # trainX = np.vstack((trainX, x_in))
        trainXs.append(Sxx)

        if "no_cough" in ncf:
            truth = [1, 0]
            # f.savefig("data/cough_learn_histo/no_cough_{}.png".format(idx))
            #matplotlib.image.imsave("data/cough_learn_histo/no_cough_{}.png".format(idx), Sxx)
        if "cough_added" in ncf:
            truth = [0, 1]
            # f.savefig("data/cough_learn_histo/cough_{}.png".format(idx))
            #matplotlib.image.imsave("data/cough_learn_histo/cough_{}.png".format(idx), Sxx)
        # plt.close(f)
        trainYs.append(truth)

    trainX = np.stack(trainXs)
    trainY = np.stack(trainYs)


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

        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0, scaling="spectrum", mode="magnitude")

        # x_in = Sxx.flatten()
        # real_testX = np.vstack((real_testX, x_in))

        #real_testY = np.vstack((real_testY, [0, 1]))
        real_testYs.append([0, 1])
        real_testXs.append(Sxx)
    real_testX = np.stack(real_testXs)
    real_testY = np.stack(real_testYs)

    untouched = []
    for af in untouched_files[0:1465]:
        x, sr = librosa.load(af, sr=SAMPLERATE, mono=True)
        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0,scaling="spectrum", mode="magnitude")
        untouched.append(Sxx)
    untouched_testX  = np.stack(untouched)

    crossvalids = []
    for af in crossvalid_files:
        x, sr = librosa.load(af, sr=SAMPLERATE, mono=True)
        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0,scaling="spectrum", mode="magnitude")
        crossvalids.append(Sxx)
    crossvalids_testX  = np.stack(crossvalids)

    # X_and_label = np.concatenate((trainX, trainY),axis=1)
    cache_dump = {"real_testX": real_testX,"real_testY":real_testY, "trainX": trainX, "trainY": trainY, "untouched_testX": untouched_testX,"crossvalids_testX": crossvalids_testX}
    pickle.dump(cache_dump, open(traindata, "wb"))

## Load from pickle
load_old=False
if load_old:
    cache_dump = pickle.load(open(traindata, "rb"))
    real_testX = cache_dump["real_testX"]
    real_testY = cache_dump["real_testY"]
    trainX = cache_dump["trainX"]
    trainY = cache_dump["trainY"]
    untouched_testX = cache_dump["untouched_testX"]
    crossvalids_testX = cache_dump["crossvalids_testX"]


## Load from images
load_images=True
if load_images:

    np.random.shuffle(images_train)

    trainYs = []
    trainXs = []
    for train_image in images_train:
        image = imageio.imread(train_image)
        image = rgb2gray(image)

        trainXs.append(image)

        if train_image.startswith("cough"):
            truth = [1, 0]

        if "no_cough" in train_image:
            truth = [0, 1]
        else:
            truth = [1, 0]

        trainYs.append(truth)

    trainX = np.stack(trainXs)
    trainY = np.stack(trainYs)

    crossvalid_trainXs = []
    standard_shape = rgb2gray(imageio.imread(images_crossvalid[0])).shape
    print(standard_shape)
    for im in images_crossvalid:
        image = imageio.imread(im)
        image = rgb2gray(image)

        if image.shape == standard_shape:
            crossvalid_trainXs.append(image)

    crossvalids_testX = np.stack(crossvalid_trainXs)
    untouched_testX = np.array([])
    real_testX = np.array([])


#testX = trainX[1700:2600]
#testY = trainY[1700:2600]
#trainX = trainX[0:1700]
#trainY = trainY[0:1700]

# shapeSxx = (1001, 147, 1)

# shapeSxx = (151, 88, 1)
# trainX = trainX[:,100:251,0:88]
# real_testX = real_testX[:,100:251,0:88]
# untouched_testX = untouched_testX[:, 100:251, 0:88]
# crossvalids_testX = crossvalids_testX[:, 100:251, 0:88]

testX = trainX[0:1000]
testY = trainY[0:1000]
trainX = trainX[1000:]
trainY = trainY[1000:]

#standardize
trainX = (trainX - np.mean(trainX)) / np.std(trainX)
testX = (testX -np.mean(testX)) / np.std(testX)
untouched_testX = (untouched_testX - np.mean(untouched_testX)) / np.std(untouched_testX)
real_testX = (real_testX - np.mean(real_testX)) / np.std(real_testX)
crossvalids_testX = (crossvalids_testX - np.mean(crossvalids_testX)) / np.std(crossvalids_testX)

# reshape to fit conv2D
trainX = trainX.reshape(trainX.shape + (1,))
testX = testX.reshape(testX.shape + (1,))
real_testX = real_testX.reshape(real_testX.shape + (1,))
untouched_testX = untouched_testX.reshape(untouched_testX.shape + (1,))
crossvalids_testX = crossvalids_testX.reshape(crossvalids_testX.shape + (1,))


EPOCHS=100
BATCHSIZE=20
da = 0.09
m1 = keras.Sequential()
m1.add(keras.layers.Conv2D(5, kernel_size=(3,3), strides=1, activation='relu', input_shape=shapeSxx)) # kernel_regularizer=keras.regularizers.l1_l2()
m1.add(keras.layers.Conv2D(5, kernel_size=(2,2), strides=1, activation='relu'))
m1.add(keras.layers.MaxPooling2D((20,20)))

m1.add(keras.layers.Flatten())
m1.add(keras.layers.BatchNormalization())

#m1.add(keras.layers.Dense(20, activation='relu'))
#m1.add(keras.layers.Dense(20, activation='relu'))

#m1.add(keras.layers.Dropout(0.1))
#m1.add(keras.layers.BatchNormalization())
m1.add(keras.layers.Dense(10, activation='relu'))
m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(5, activation='relu'))
m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(5, activation='relu'))
m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.Dense(5, activation='relu'))
m1.add(keras.layers.Dropout(da))
m1.add(keras.layers.BatchNormalization())
m1.add(keras.layers.Dense(2, activation='softmax'))
#m1.add(keras.layers.Dropout(0.1))
m1.summary()


#opt=keras.optimizers.SGD(learning_rate=0.0003)
#opt=keras.optimizers.Adam(lr=0.003)
opt=keras.optimizers.Adam()
#opt=keras.optimizers.Adagrad()
#opt=keras.optimizers.Nadam()
m1.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["categorical_accuracy"])

if flag_re_train:
#if False:
    H = m1.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=BATCHSIZE)
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
print("* analyzing: {}".format(CROSSVALID))
# H3 = m1.predict(untouched_testX)
# print(H3)
f=open("untouched_labels_crossvalid.txt", "w")
for idx, line in enumerate(m1.predict(crossvalids_testX)):
    print(idx, line)
    #if line[1] > 0.6:
    f.write("{}\t{}\t{}\t{}\n".format(idx*CHUNKSIZE, (idx*CHUNKSIZE)+2, line[0], line[1]))
f.close()




