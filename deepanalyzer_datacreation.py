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
import random
import glob

# flags
# 1. create cough and no cough data
flag_create_cough_data = True
# 2. create manual labeled snippets
flag_create_manual_labeled = True

# settings
FOLDER_COUGHS = "data/coughexamples/"
AUDIO_FILE="/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/Rachmaninoff_ Piano Concerto No. 3 - Anna Fedorova - Live concert HD (152kbit_Opus).ogg.wav"
MAN_LABEL_FILE = "/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/manual_labeled_hustenandsounds.txt"

SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds
factor_volume = [0.01, 0.05, 0.07, 0.09, 0.11]


# Load audio file
y, sr = librosa.load(AUDIO_FILE, sr=SAMPLERATE, mono=True)
N_audio = len(y)
# librosa.output.write_wav(AUDIO_FILE+".wav", y, sr=SAMPLERATE)

# load labels
df_man_label = pd.read_csv(MAN_LABEL_FILE, sep="\t", header=None)

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



############################### create files with syntetic added husten and clean files ##############################
if flag_create_cough_data:
    files_cough_examples = glob.glob(FOLDER_COUGHS+"/*")

    y_ces = []
    for ce in files_cough_examples:
        y_ce, sr = librosa.load(ce, sr=SAMPLERATE, mono=True)


        # bring snippets to same length
        if len(y_ce) > CHUNKSIZE*SAMPLERATE:
            y_ce = y_ce[0:CHUNKSIZE*SAMPLERATE]
        if len(y_ce) < CHUNKSIZE*SAMPLERATE:
            y_ce = np.append(y_ce,  [1]*((CHUNKSIZE*SAMPLERATE) - len(y_ce)))
            # y_ce.append([0]*((CHUNKSIZE*SAMPLERATE) - len(y_ce)))

        y_ces.append(y_ce)
        print(ce, "length", len(y_ce)/SAMPLERATE)


    # loop the main audio to create 2s husten samples
    print("saving snippets", end="")
    for i in range(0, N_audio, CHUNKSIZE*SAMPLERATE):

        # choose random cough
        rand_idx_cough = random.randint(0,len(y_ces)-1)
        rand_idx_vol = random.randint(0,len(factor_volume)-1)
        pre_gap = [0] * random.randint(0, 0.5*SAMPLERATE) # up to half a second pre gap possible

        # add cough and save the snippet. skip last one if lengths are not equal
        if len(y[i:i+CHUNKSIZE*SAMPLERATE]) == CHUNKSIZE*SAMPLERATE:
            # add pregap and slice to correct length again
            y_cough = np.append(pre_gap, y_ces[rand_idx_cough])
            y_cough = y_cough[0:CHUNKSIZE*SAMPLERATE]

            # add the two audios and save
            y_incl = y[i:i+CHUNKSIZE*SAMPLERATE] + (factor_volume[rand_idx_vol] * y_cough)
            librosa.output.write_wav("data/cough_added/cough_added_{}.wav".format(i), y_incl, sr=SAMPLERATE)

            if not check_if_within_manual_labels(i, df_man_label):
                librosa.output.write_wav("data/no_cough/no_cough_{}.wav".format(i), y[i:i+CHUNKSIZE*SAMPLERATE], sr=SAMPLERATE)

        print(".", end="")
#######################################################################################################################


############ get the manually labeled snipets as advanced test data ###################################################
if flag_create_manual_labeled:

    idx = 0
    for row in df_man_label.iterrows():
        if row[1][0] != "\\" and row[1][2] == "husten":
            print(row[1][0])
            start = float(row[1][0])*SAMPLERATE
            end = start+CHUNKSIZE*SAMPLERATE

            librosa.output.write_wav("data/manual_cough/manual_cough{}.wav".format(idx), y[int(start):int(end)], sr=SAMPLERATE)
            idx+=1
print(0)
#######################################################################################################################