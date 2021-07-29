import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from scipy.fft import fft
from scipy import signal
import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import os
import random
import math
import glob
import pickle

# flags
# 1. create cough and no cough data
flag_create_cough_data = True
# 2. create manual labeled snippets
flag_create_manual_labeled = True
# 3. create crossvalid files
flag_create_crossvalid_files=True
# 4. Preprocessing Flag
flag_pp = True



# Preprocessing save name
traindata = "temp10_0.3.p"
traindata = "cough_vs_music.p"
traindata = "temp11_0.3.p"
traindata = "temp12_short_0.3.p"
traindata = "temp13_16khz_short_0.3.p"

# settings
SAMPLERATE = 16000
CHUNKSIZE = 2 # seconds

factor_volume_music = 1

prefactor_volume = 0.25
factor_volume = [0.2 * prefactor_volume, 0.25*prefactor_volume, 0.3*prefactor_volume]
#factor_volume=[1]

stretch_factors = [0.9, 0.95, 0.97, 1, 1.03, 1.05, 1.1]
#stretch_factors = [1]
pitch_steps = [-4, -2, -1, 0, 1, 2, 4]
#pitch_steps = [0]

FOLDER_COUGHS = "data/coughexamples/"
FOLDER_CROSSVALID = "data/crossvalidation"

# These files go into the learning and valid
database = [
    {"audio_file": "data/raw_youtube/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/Rachmaninoff_ Piano Concerto No. 3 - Anna Fedorova - Live concert HD (152kbit_Opus).ogg.wav",
     "man_label_file": "data/raw_youtube/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/manual_labeled_hustenandsounds3.txt",
     "cut_file": "data/raw_youtube/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/cutfile.txt"},
    {"audio_file": "data/raw_youtube/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).wav",
     "man_label_file": "data/raw_youtube/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/manual_labeled.txt",
     "cut_file": "data/raw_youtube/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/cut_file.txt"},
    {"audio_file": "data/raw_studio/shostakovich CD08 - Symphony 10/shostakovich_10_studio.wav",
     "man_label_file": None,
     "cut_file": None},
    # {"audio_file": "data/raw_studio/Daniel Barenboim - Beethoven Concerto for Violin and Orchestra in D Major, Op. 61 (2017)/beethoven_violin_studio.wav",
    #  "man_label_file": None,
    #  "cut_file": None},
    # {
    #     "audio_file":"data/raw_studio/Haydn cd2 n.41, 58/haydn_41_58.wav",
    #     "man_label_file": None,
    #     "cut_file": None
    # }
]

# The files in crossvalid are not used for training/valid purposes. Its a second round of validation with real examples
database_crossvalid = [
    {"audio_file": "data/raw_youtube/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009 (152kbit_Opus).wav",
     "man_label_file": None,
     "cut_file": None},
    {"audio_file": "data/raw_youtube/Beethoven _ Concerto pour piano n°3 ( Alice Sara Ott _ Orchestre philharmonique de Radio France)/Beethoven _ Concerto pour piano n°3 .wav",
    "man_label_file": None,
    "cut_file": None},
    {
        "audio_file": "data/raw_youtube/chopin_piano1_eminor_olgascheps/Frédéric Chopin_ Piano Concerto No. 1 e-minor (Olga Scheps.wav",
        "man_label_file": None,
        "cut_file": None
    },
    {
        "audio_file": "data/raw_youtube/Mozart_ Piano Concerto No  21 - Netherlands Philharmonic Orchestra, Ronald Brautigam - Live HD/Mozart_ Piano Concerto No. 21 - Netherlands Philharmonic Orchestra, Ronald Brautigam.wav",
        "man_label_file": None,
        "cut_file": None
    },
    {
        "audio_file": "data/raw_youtube/Rachmaninoff_ Piano Concerto no 2 op 18 - Anna Fedorova - Complete Live Concert - HD/Rachmaninoff_ Piano Concerto no.2 op.18 - Anna Fedorova - Complete Live Concert .wav",
        "man_label_file": None,
        "cut_file": None
    }
]


##################################################################################################################
# DATACREATION
##################################################################################################################
print("* Doing Datacreation")

def foreground_Separation(y, sr):

    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y, n_fft=2000, hop_length=300))

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(0.5, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 3

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    #S_background = mask_i * S_full

    return S_foreground


def check_if_within_manual_labels(input_i, df_man_label):
    idx = 0
    isInside = False
    for row in df_man_label.iterrows():
        if row[1][0] != "\\":
            start = float(row[1][0])*SAMPLERATE
            end = start+CHUNKSIZE*SAMPLERATE

            if input_i > start and input_i < end:
                isInside = True
            idx+=1

    return isInside

if flag_create_cough_data or flag_create_manual_labeled:
    ys = []
    df_man_labels = pd.DataFrame()
    print("Read and concat files...")
    for db in database:
        # Load audio file(s)
        y, sr = librosa.load(db["audio_file"], sr=SAMPLERATE, mono=True)
        # librosa.output.write_wav(AUDIO_FILE+".wav", y, sr=SAMPLERATE)

        # load labels
        if db["man_label_file"] != None:
            df_man_label = pd.read_csv(db["man_label_file"], sep="\t", header=None)
            df_man_labels = df_man_labels.append(df_man_label)

        # load cutfile
        if db["cut_file"] != None:
            df_cutfile = pd.read_csv(db["cut_file"], sep="\t", header=None)

            # set cutfile part to 0
            for row in df_cutfile.iterrows():
                if row[1][2] == "cut":
                    y[int(row[1][0]*SAMPLERATE):int(row[1][1]*SAMPLERATE)] = 0

        ys.extend(y)

    y = ys
    N_audio = len(y)

############################### create files with syntetic added husten and clean files ##############################
if flag_create_cough_data:
    files_cough_examples = glob.glob(FOLDER_COUGHS+"/*")

    # open and cut the cough files to 2s
    y_ces = []
    counter = 0
    print("Creating cough samples...")
    for idx_cough, ce in enumerate(files_cough_examples):
        y_ce, sr = librosa.load(ce, sr=SAMPLERATE, mono=True)

        # stretch and ditch the files
        for sf in stretch_factors:
            y_ce_stretched = librosa.effects.time_stretch(y_ce, sf)

            # pitch the files
            for ps in pitch_steps:

                    y_ce_stretched_pitched = librosa.effects.pitch_shift(y_ce_stretched, sr, n_steps=ps, bins_per_octave = 24)

                    # change volume
                    for vol in factor_volume:
                        y_ce_stretched_pitched_vol = vol * y_ce_stretched_pitched

                        # bring snippets to same length
                        if len(y_ce_stretched_pitched_vol) > CHUNKSIZE*SAMPLERATE:
                            y_ce_stretched_pitched_vol = y_ce_stretched_pitched_vol[0:CHUNKSIZE*SAMPLERATE]
                        if len(y_ce_stretched_pitched_vol) < CHUNKSIZE*SAMPLERATE:
                            y_ce_stretched_pitched_vol = np.append(y_ce_stretched_pitched_vol,  [0]*((CHUNKSIZE*SAMPLERATE) - len(y_ce_stretched_pitched_vol)))
                            # y_ce.append([0]*((CHUNKSIZE*SAMPLERATE) - len(y_ce)))

                        librosa.output.write_wav("data/coughexamples_changed/changed_{}_s{}_p{}_v{}.wav".format(idx_cough, sf, ps, vol), y_ce_stretched_pitched_vol, sr=SAMPLERATE)
                        counter+=1
                        y_ces.append(y_ce_stretched_pitched_vol)
                        print(ce, "length", len(y_ce_stretched_pitched_vol)/SAMPLERATE)




    # loop the main audio to create 2s husten samples
    print("* saving snippets...")
    status=""
    f = open("data/untouched/untouched_files.txt", "w")
    idx=0
    y_ges_piece_with_cough = []

    print("looping through ", N_audio, "datapoints...")
    for i in range(0, N_audio-(CHUNKSIZE*SAMPLERATE), CHUNKSIZE*SAMPLERATE):

        # choose random cough
        rand_idx_cough = random.randint(0,len(y_ces)-1)
        rand_idx_vol = random.randint(0,len(factor_volume)-1)
        pre_gap = [0] * random.randint(0, 0.5*SAMPLERATE) # up to half a second pre gap possible

        # add cough and save the snippet. skip last one if lengths are not equal
        #if len(y[i:i+CHUNKSIZE*SAMPLERATE]) == CHUNKSIZE*SAMPLERATE:
        # add pregap and slice to correct length again
        y_cough = np.append(pre_gap, y_ces[rand_idx_cough])
        y_cough = y_cough[0:CHUNKSIZE*SAMPLERATE]

        # add the two audios and save
        y_incl = factor_volume_music * np.array((y[i:i+CHUNKSIZE*SAMPLERATE])) + (y_cough)
        # y_ges_piece_with_cough.extend(y_incl)
        librosa.output.write_wav("data/cough_added/cough_added_{}.wav".format(i), np.array(y_incl), sr=SAMPLERATE)

        # save no cough version
        if not check_if_within_manual_labels(i, df_man_labels):
            librosa.output.write_wav("data/no_cough/no_cough_{}.wav".format(i), np.array(y[i:i+CHUNKSIZE*SAMPLERATE]), sr=SAMPLERATE)

        # save untouched version
        librosa.output.write_wav("data/untouched/untouched_{}.wav".format(i), np.array(y[i:i + CHUNKSIZE * SAMPLERATE]),sr=SAMPLERATE)
        f.write("{nr} {start} {end} {min}:{sec}\n".format(nr=idx, start=i, end=i+CHUNKSIZE*SAMPLERATE,
                                                          min=math.floor(i / SAMPLERATE / 60),
                                                          sec=round((i / SAMPLERATE) % 60, 2)))
        idx+=1
        status ="Doing chunk {} - {}\r".format(i, i+CHUNKSIZE*SAMPLERATE)
        print(status, end="")
    f.close()
    # librosa.output.write_wav(AUDIO_FILE+"_coughs_added.wav", np.array(y_ges_piece_with_cough),sr=SAMPLERATE)
#######################################################################################################################


############ get the manually labeled snipets as advanced test data ###################################################
if flag_create_manual_labeled:
    print("create manually labeled snippets...")
    f = open("data/manual_cough/manual_cough_files.txt", "w")
    idx = 0
    for row in df_man_labels.iterrows():
        if row[1][0] != "\\" and row[1][2] == "husten":
            print(row[1][0])
            start = float(row[1][0])*SAMPLERATE
            end = start+CHUNKSIZE*SAMPLERATE

            librosa.output.write_wav("data/manual_cough/manual_cough_{}.wav".format(idx), np.array(y[int(start):int(end)]), sr=SAMPLERATE)
            f.write("{nr} {start} {end} {min}:{sec}\n".format(nr=idx, start=start,end=end, min=math.floor(start/SAMPLERATE/60), sec=round((start/SAMPLERATE) % 60, 2)))
            idx+=1
    f.close()
#######################################################################################################################



############## Create crossvalid data #################################################################################
print("* saving crossvalidation files")
if flag_create_crossvalid_files:
    for db_cv in database_crossvalid:
        print("doing: ", db_cv["audio_file"])
        y, sr = librosa.load(db_cv["audio_file"], sr=SAMPLERATE, mono=True)

        filename = os.path.split(db_cv["audio_file"])[1].split(".")[0]

        #create dirs if not there
        if not os.path.exists(FOLDER_CROSSVALID + "/{}".format(filename)):
            os.makedirs(FOLDER_CROSSVALID + "/{}".format(filename))

        N_audio = len(y)
        for i in range(0, N_audio-(CHUNKSIZE*SAMPLERATE), CHUNKSIZE*SAMPLERATE):
            librosa.output.write_wav(FOLDER_CROSSVALID + "/{}/snippet_cv_{}.wav".format(filename, i), np.array(y[i:i+SAMPLERATE*CHUNKSIZE]), sr=SAMPLERATE)

        del y
#######################################################################################################################




#######################################################################################################################
#######################################################################################################################
# PREPROCESSING
#######################################################################################################################
#######################################################################################################################
if flag_pp:
    print("* Doing Preprocessing")

    # load data
    COUGH_FOLDER="data/cough_added"
    NO_COUGH_FOLDER="data/no_cough"
    MANUAL_COUGH="data/manual_cough"
    UNTOUCHED="data/untouched"

    # files to crossvalidate. these have not been used for training
    # CROSSVALID = "data/crossvalidation/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009 (152kbit_Opus)"
    CROSSVALID = "data/crossvalidation/Beethoven _ Concerto pour piano n°3 "

    cough_files = glob.glob(COUGH_FOLDER+"/*.wav")
    no_cough_files = glob.glob(NO_COUGH_FOLDER+"/*.wav")
    manual_cough_files = glob.glob(NO_COUGH_FOLDER+"/*.wav")
    untouched_files = glob.glob(UNTOUCHED+"/*.wav")
    crossvalid_files = glob.glob(CROSSVALID+"/*.wav")


    train_files = cough_files + no_cough_files
    np.random.shuffle(train_files)




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
        Sxx_foreground = foreground_Separation(x, SAMPLERATE)
        # f, axs = plt.subplots(1,1)
        # axs.pcolormesh(txx, fxx, Sxx, norm=cl.LogNorm())

        # x_in = Sxx.flatten()
        # trainX = np.vstack((trainX, x_in))
        trainXs.append(Sxx_foreground)

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
        Sxx_foreground = foreground_Separation(x, SAMPLERATE)
        # x_in = Sxx.flatten()
        # real_testX = np.vstack((real_testX, x_in))

        #real_testY = np.vstack((real_testY, [0, 1]))
        real_testYs.append([0, 1])
        real_testXs.append(Sxx_foreground)
    real_testX = np.stack(real_testXs)
    real_testY = np.stack(real_testYs)

    untouched = []
    for af in untouched_files[0:1465]:
        x, sr = librosa.load(af, sr=SAMPLERATE, mono=True)
        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0,scaling="spectrum", mode="magnitude")
        Sxx_foreground = foreground_Separation(x, SAMPLERATE)
        untouched.append(Sxx_foreground)
    untouched_testX  = np.stack(untouched)

    crossvalids = []
    for af in crossvalid_files:
        x, sr = librosa.load(af, sr=SAMPLERATE, mono=True)
        fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0,scaling="spectrum", mode="magnitude")
        Sxx_foreground = foreground_Separation(x, SAMPLERATE)
        crossvalids.append(Sxx_foreground)
    crossvalids_testX  = np.stack(crossvalids)

    # X_and_label = np.concatenate((trainX, trainY),axis=1)
    cache_dump = {"real_testX": real_testX,"real_testY":real_testY, "trainX": trainX, "trainY": trainY, "untouched_testX": untouched_testX,"crossvalids_testX": crossvalids_testX}
    pickle.dump(cache_dump, open(traindata, "wb"), protocol=4)

print("Finished!")
