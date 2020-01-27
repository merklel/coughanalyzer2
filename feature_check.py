import librosa
import librosa.display
import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys
from sklearn import svm, metrics, linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

flag_makedata = True

# constants
SAMPLERATE = 22500
CHUNKSIZE = 2 # seconds
CACHE_FILE = "cache_2.p"

# load data
COUGH_FOLDER="data/cough_added"
NO_COUGH_FOLDER="data/no_cough"
MANUAL_COUGH="data/manual_cough"
UNTOUCHED="data/untouched"
COUGHEXAMPLES_CHANGED_FOLDER = "data/coughexamples_changed"

manual_cough_files = glob.glob(MANUAL_COUGH+"/*.wav")
cough_added_files = glob.glob(COUGH_FOLDER+"/*.wav")
no_cough_files = glob.glob(NO_COUGH_FOLDER+"/*.wav")
cough_only_files  =glob.glob(COUGHEXAMPLES_CHANGED_FOLDER+"/*.wav")

def get_chromas(y):
    chroma = librosa.feature.chroma_stft(y, sr=SAMPLERATE, n_fft=200, hop_length=200)
    return np.add.reduce(chroma,1)

def spectral_contrast_plot(y, title=""):
    S = np.abs(librosa.stft(y))
    contrast = librosa.feature.spectral_contrast(S=S, sr=SAMPLERATE, n_fft=200, hop_length=200)
    return np.add.reduce(contrast,1)
    plt.figure()
    plt.subplot(2, 1, 1)
    librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max), y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.subplot(2, 1, 2)
    librosa.display.specshow(contrast, x_axis='time', sr=SAMPLERATE)
    plt.colorbar()
    plt.ylabel('Frequency bands')
    plt.title('Spectral contrast ' + title)
    plt.tight_layout()
    #plt.show()

def get_flatness(y):
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=200, hop_length=200)
    return flatness

def get_zero_crossings(y):
    return librosa.feature.zero_crossing_rate(y, frame_length=200, hop_length=200)

def get_rms(y):
    return librosa.feature.rms(y, frame_length=200, hop_length=200)

if flag_makedata:
    ##### SVM feature vectors
    feature_X = []
    feature_Y = []

    feature_X_real = []
    feature_Y_real = []



    print("* Manual labeled cough")
    flattness_ges_mcf = []
    zc = []
    f, axs = plt.subplots(1,1)
    f.suptitle("Manual labeled cough")
    rmss = []
    for mcf in manual_cough_files:
        y, sr = librosa.load(mcf, sr=SAMPLERATE)

        rms = get_rms(y)
        rmss.append(rms)
        yy = get_flatness(y)
        flattness_ges_mcf.append(np.mean(yy))
        zerocrossings = get_zero_crossings(y)
        zc.append(get_zero_crossings(y))
        axs.plot(yy[0])
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = spectral_contrast_plot(y)
        chroma = get_chromas(y)
        features = [np.std(yy), np.mean(yy), np.median(yy), np.nanstd(spec_bw), np.nanmean(spec_bw), np.nanstd(zerocrossings), np.nanmean(zerocrossings), np.nanstd(rms), np.mean(rms)]
        features.extend(chroma)
        features.extend(contrast)
        feature_X_real.append(features)
        #feature_X_real.append([np.std(yy), np.mean(yy)])
        feature_Y_real.append(1)



    print("mean rms: ",np.mean(rmss))
    print("mean zc: ", np.mean(zc))
    print("mean spectral bandwidth: ", np.nanmean(spec_bw))
    print("mean: ", np.mean(flattness_ges_mcf))
    print("median: ", np.median(flattness_ges_mcf))
    print("std: ", np.std(flattness_ges_mcf))
    plt.figure()
    plt.title("manual labeled c")
    plt.hist(flattness_ges_mcf, bins=30, range=(0, 1))
    # plt.show()
    #
    #
    # print("* cough only")
    # flattness_ges_mcf = []
    # f, axs = plt.subplots(1,1)
    # f.suptitle("Cough only")
    # zc = []
    # for cof in cough_only_files:
    #     y, sr = librosa.load(cof, sr=SAMPLERATE)
    #
    #     # spectral_contrast_plot(y, title="no cough")
    #     yy = get_flatness(y)
    #     flattness_ges_mcf.append(np.mean(yy))
    #     axs.plot(yy[0])
    #     zerocrossings = get_zero_crossings(y)
    #     zc.append(zerocrossings)
    #     spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    #
    #
    # print("mean zc: ", np.mean(zc))
    # print("mean spectral bandwidth: ", np.nanmean(spec_bw))
    # print("mean: ", np.mean(flattness_ges_mcf))
    # print("median: ", np.median(flattness_ges_mcf))
    # print("std: ", np.std(flattness_ges_mcf))
    # plt.figure()
    # plt.title("cough only")
    # plt.hist(flattness_ges_mcf, bins=30, range=(0, 1))
    # # plt.show()


    print("* Cough added")
    flattness_ges = []
    f, axs = plt.subplots(1,1)
    rmss=[]
    f.suptitle("Cough added")
    zc=[]
    for idx, mcf in enumerate(cough_added_files[5200:6000]):
        y, sr = librosa.load(mcf, sr=SAMPLERATE)

        #spectral_contrast_plot(y, title="cough added")

        rms = get_rms(y)
        rmss.append(rms)

        # flattness
        yy = get_flatness(y)
        flattness_ges.append(np.mean(yy))
        #axs.plot(yy[0])
        zerocrossings = get_zero_crossings(y)
        zc.append(zerocrossings)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = spectral_contrast_plot(y)
        chroma = get_chromas(y)

        features = [np.std(yy), np.mean(yy), np.median(yy), np.nanstd(spec_bw), np.nanmean(spec_bw), np.nanstd(zerocrossings), np.nanmean(zerocrossings), np.nanstd(rms), np.mean(rms)]
        features.extend(chroma)
        features.extend(contrast)
        feature_X.append(features)
        #feature_X.append([np.std(yy), np.mean(yy)])
        feature_Y.append(1)

        print(idx, "/", len(cough_added_files), flush=True)
        sys.stdout.flush()

    print("")
    print("mean rms: ",np.mean(rmss))
    print("mean zc: ", np.mean(zc))
    print("mean spectral bandwidth: ", np.nanmean(spec_bw))
    print("mean: ", np.nanmean(flattness_ges))
    print("median: ", np.nanmedian(flattness_ges))
    print("std: ", np.nanstd(flattness_ges))
    plt.figure()
    plt.title("Cough added")
    plt.hist(flattness_ges, bins=30, range=(0,1))



    print("* No Cough")
    flattness_ges_nc = []
    f, axs = plt.subplots(1,1)
    f.suptitle("No Cough")
    zc=[]
    rmss=[]
    for idx, mcf in enumerate(no_cough_files[5200:6000]):
        y, sr = librosa.load(mcf, sr=SAMPLERATE)

        rms = get_rms(y)
        rmss.append(rms)

        # spectral_contrast_plot(y, title="no cough")
        flattness_ges_nc.append(np.mean(get_flatness(y)))
        yy = get_flatness(y)
        flattness_ges_nc.append(np.mean(yy))
        #axs.plot(yy[0])
        zerocrossings = get_zero_crossings(y)
        zc.append(zerocrossings)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        contrast = spectral_contrast_plot(y)
        chroma = get_chromas(y)

        features = [np.std(yy), np.mean(yy), np.median(yy), np.nanstd(spec_bw), np.nanmean(spec_bw), np.nanstd(zerocrossings), np.nanmean(zerocrossings), np.nanstd(rms), np.mean(rms)]
        features.extend(chroma)
        features.extend(contrast)
        feature_X.append(features)
        #feature_X.append([np.nanmean(spec_bw),np.nanmax(spec_bw), np.nanmean(zerocrossings), np.nanmax(zerocrossings)])
        #feature_X.append([np.std(yy), np.mean(yy)])
        feature_Y.append(0)

        print(idx,"/",len(no_cough_files), flush=True)
        sys.stdout.flush()

    print("")
    print("mean rms: ",np.mean(rmss))
    print("mean zc: ", np.mean(zc))
    print("mean spectral bandwidth: ", np.nanmean(spec_bw))
    print("mean: ", np.mean(flattness_ges_nc))
    print("median: ", np.median(flattness_ges_nc))
    print("std: ", np.std(flattness_ges_nc))
    plt.figure()
    plt.title("No Cough")
    plt.hist(flattness_ges_nc, bins=30, range=(0,1))

    # Normalize
    feature_X = preprocessing.normalize(feature_X)
    #feature_Y = preprocessing.normalize(feature_Y)
    feature_X_real = preprocessing.normalize(feature_X_real)
    #feature_Y_real = preprocessing.normalize(feature_Y_real)

    # Split dataset into training set and test set
    feature_X_train, feature_X_test, feature_Y_train, feature_Y_test = train_test_split(feature_X, feature_Y, test_size=0.3, random_state=109) # 70% training and 30% test
    # feature_X_train = feature_X[500:]
    # feature_Y_train = feature_Y[500:]
    # feature_X_test = feature_X[0:500]
    # feature_Y_test = feature_Y[0:500]

    save = {"feature_X_train": feature_X_train, "feature_X_test": feature_X_test, "feature_Y_train": feature_Y_train,"feature_Y_test": feature_Y_test, "feature_X_real": feature_X_real, "feature_Y_real": feature_Y_real}
    pickle.dump(save, open(CACHE_FILE, "wb"))



save = pickle.load(open(CACHE_FILE, "rb"))
feature_X_train = save["feature_X_train"]
feature_X_test = save["feature_X_test"]
feature_Y_train = save["feature_Y_train"]
feature_Y_test = save["feature_Y_test"]
feature_X_real = save["feature_X_real"]
feature_Y_real = save["feature_Y_real"]

###############################################################
# Train PCA
##############################################################
print("* Do PCA")
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(feature_X_train)

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
# plt.figure()
# plt.scatter(principalComponents[:,0], principalComponents[:,1])
# plt.show()

###############################################################
# Train SVM
##############################################################
print("* Train SVM")
clf = svm.SVC()
#clf = svm.LinearSVC(penalty="l2", loss="hinge")
#clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3)
clf.fit(feature_X_train, feature_Y_train)

print("* Validate on Testdataset")
y_pred = clf.predict(feature_X_test)
print("SVM Accuracy:",metrics.accuracy_score(feature_Y_test, y_pred))
# for i in y_pred:
#     print(y_pred[i], feature_Y_test[i])

print("* Validate on Validationset")
y_pred_real = clf.predict(feature_X_real)
print("SVM Accuracy:",metrics.accuracy_score(feature_Y_real, y_pred_real))
# for i in y_pred_real:
#     print(y_pred_real[i], feature_Y_real[i])


###############################################################
# Train Decision Tree
##############################################################
print("----------------------------")
print("* Train Decision Tree")
clf = DecisionTreeClassifier(random_state=0)
clf.fit(feature_X_train, feature_Y_train)

print("* Validate on Testdataset")
y_pred = clf.predict(feature_X_test)
print("Decision TreeAccuracy:",metrics.accuracy_score(feature_Y_test, y_pred))

print("* Validate on Validationset")
y_pred = clf.predict(feature_X_real)
print("Decision TreeAccuracy:",metrics.accuracy_score(feature_Y_real, y_pred_real))

#plt.show()