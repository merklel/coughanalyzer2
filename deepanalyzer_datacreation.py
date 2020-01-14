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
import math
import glob

# flags
# 1. create cough and no cough data
flag_create_cough_data = True
# 2. create manual labeled snippets
flag_create_manual_labeled = True

# settings
FOLDER_COUGHS = "data/coughexamples/"
AUDIO_FILE="/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/Rachmaninoff_ Piano Concerto No. 3 - Anna Fedorova - Live concert HD (152kbit_Opus).ogg.wav"
MAN_LABEL_FILE = "/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/manual_labeled_hustenandsounds2.txt"
CUT_FILE="data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/cutfile.txt"

SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds

factor_volume_music = 1

f=0.3
factor_volume = [0.2*f, 0.3*f, 0.5*f]
#factor_volume=[1]

stretch_factors = [0.9, 1, 1.1]
#stretch_factors = [1]
pitch_steps = [-5, 0, 5]
#pitch_steps = [0]

# Load audio file(s)
y, sr = librosa.load(AUDIO_FILE, sr=SAMPLERATE, mono=True)
N_audio = len(y)
# librosa.output.write_wav(AUDIO_FILE+".wav", y, sr=SAMPLERATE)

# load labels
df_man_label = pd.read_csv(MAN_LABEL_FILE, sep="\t", header=None)

# load cutfile
df_cutfile = pd.read_csv(CUT_FILE, sep="\t", header=None)

# set cutfile part to 0
for row in df_cutfile.iterrows():
    if row[1][2] == "cut":
        y[int(row[1][0]*SAMPLERATE):int(row[1][1]*SAMPLERATE)] = 0

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

    # open and cut the cough files to 2s
    y_ces = []
    counter = 0
    for ce in files_cough_examples:
        y_ce, sr = librosa.load(ce, sr=SAMPLERATE, mono=True)

        # stretch and ditch the files
        for sf in stretch_factors:
            y_ce_stretched = librosa.effects.time_stretch(y_ce, sf)

            # pitch the files
            for ps in pitch_steps:

                y_ce_stretched_pitched = librosa.effects.pitch_shift(y_ce_stretched, sr, n_steps=ps, bins_per_octave = 24)

                # bring snippets to same length
                if len(y_ce_stretched_pitched) > CHUNKSIZE*SAMPLERATE:
                    y_ce_stretched_pitched = y_ce_stretched_pitched[0:CHUNKSIZE*SAMPLERATE]
                if len(y_ce_stretched_pitched) < CHUNKSIZE*SAMPLERATE:
                    y_ce_stretched_pitched = np.append(y_ce_stretched_pitched,  [1]*((CHUNKSIZE*SAMPLERATE) - len(y_ce_stretched_pitched)))
                    # y_ce.append([0]*((CHUNKSIZE*SAMPLERATE) - len(y_ce)))

                librosa.output.write_wav("data/coughexamples_changed/changed_{}.wav".format(counter), y_ce_stretched_pitched, sr=SAMPLERATE)
                counter+=1
                y_ces.append(y_ce_stretched_pitched)
                print(ce, "length", len(y_ce_stretched_pitched)/SAMPLERATE)





    # loop the main audio to create 2s husten samples
    print("saving snippets", end="")
    status=""
    f = open("data/untouched/untouched_files.txt", "w")
    idx=0
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
            y_incl = factor_volume_music*(y[i:i+CHUNKSIZE*SAMPLERATE]) + (factor_volume[rand_idx_vol] * y_cough)
            librosa.output.write_wav("data/cough_added/cough_added_{}.wav".format(i), y_incl, sr=SAMPLERATE)

            # save no cough version
            if not check_if_within_manual_labels(i, df_man_label):
                librosa.output.write_wav("data/no_cough/no_cough_{}.wav".format(i), y[i:i+CHUNKSIZE*SAMPLERATE], sr=SAMPLERATE)

            # save untouched version
            librosa.output.write_wav("data/untouched/untouched_{}.wav".format(i), y[i:i + CHUNKSIZE * SAMPLERATE],sr=SAMPLERATE)
            f.write("{nr} {start} {end} {min}:{sec}\n".format(nr=idx, start=i, end=i+CHUNKSIZE*SAMPLERATE,
                                                              min=math.floor(i / SAMPLERATE / 60),
                                                              sec=round((i / SAMPLERATE) % 60, 2)))
        idx+=1
        status +="."
        print(status)
    f.close()
#######################################################################################################################


############ get the manually labeled snipets as advanced test data ###################################################
if flag_create_manual_labeled:
    f = open("data/manual_cough/manual_cough_files.txt", "w")
    idx = 0
    for row in df_man_label.iterrows():
        if row[1][0] != "\\" and row[1][2] == "husten":
            print(row[1][0])
            start = float(row[1][0])*SAMPLERATE
            end = start+CHUNKSIZE*SAMPLERATE

            librosa.output.write_wav("data/manual_cough/manual_cough_{}.wav".format(idx), y[int(start):int(end)], sr=SAMPLERATE)
            f.write("{nr} {start} {end} {min}:{sec}\n".format(nr=idx, start=start,end=end, min=math.floor(start/SAMPLERATE/60), sec=round((start/SAMPLERATE) % 60, 2)))
            idx+=1
print(0)
f.close()
#######################################################################################################################