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
MAN_LABEL_FILE = "/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/manual_labeled_hustenandsounds3.txt"
CUT_FILE="data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/cutfile.txt"

database = [
    {"audio_file": AUDIO_FILE, "man_label_file": MAN_LABEL_FILE, "cut_file": CUT_FILE},
    {"audio_file": "data/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).wav",
     "man_label_file": "data/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/manual_labeled.txt",
     "cut_file": "data/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/cut_file.txt"}
]

SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds

factor_volume_music = 1

f=0.3
factor_volume = [0.2*f, 0.25*f, 0.3*f]
#factor_volume=[1]

stretch_factors = [0.9, 1, 1.1]
#stretch_factors = [1]
pitch_steps = [-5, 0, 5]
#pitch_steps = [0]

ys = []
df_man_labels = pd.DataFrame()
for db in database:
    # Load audio file(s)
    y, sr = librosa.load(db["audio_file"], sr=SAMPLERATE, mono=True)
    # librosa.output.write_wav(AUDIO_FILE+".wav", y, sr=SAMPLERATE)

    # load labels
    df_man_label = pd.read_csv(MAN_LABEL_FILE, sep="\t", header=None)
    df_man_labels = df_man_labels.append(df_man_label)

    # load cutfile
    df_cutfile = pd.read_csv(CUT_FILE, sep="\t", header=None)

    # set cutfile part to 0
    for row in df_cutfile.iterrows():
        if row[1][2] == "cut":
            y[int(row[1][0]*SAMPLERATE):int(row[1][1]*SAMPLERATE)] = 0

    ys.extend(y)

y = ys
N_audio = len(y)


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

                    # change volume
                    for vol in factor_volume:
                        y_ce_stretched_pitched_vol = vol * y_ce_stretched_pitched

                        # bring snippets to same length
                        if len(y_ce_stretched_pitched_vol) > CHUNKSIZE*SAMPLERATE:
                            y_ce_stretched_pitched_vol = y_ce_stretched_pitched_vol[0:CHUNKSIZE*SAMPLERATE]
                        if len(y_ce_stretched_pitched_vol) < CHUNKSIZE*SAMPLERATE:
                            y_ce_stretched_pitched_vol = np.append(y_ce_stretched_pitched_vol,  [0]*((CHUNKSIZE*SAMPLERATE) - len(y_ce_stretched_pitched_vol)))
                            # y_ce.append([0]*((CHUNKSIZE*SAMPLERATE) - len(y_ce)))

                        librosa.output.write_wav("data/coughexamples_changed/changed_{}_s{}_p{}_v{}.wav".format(counter, sf, ps, vol), y_ce_stretched_pitched_vol, sr=SAMPLERATE)
                        counter+=1
                        y_ces.append(y_ce_stretched_pitched_vol)
                        print(ce, "length", len(y_ce_stretched_pitched_vol)/SAMPLERATE)





    # loop the main audio to create 2s husten samples
    print("saving snippets", end="")
    status=""
    f = open("data/untouched/untouched_files.txt", "w")
    idx=0
    y_ges_piece_with_cough = []
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
            y_incl = factor_volume_music*(y[i:i+CHUNKSIZE*SAMPLERATE]) + (y_cough)
            y_ges_piece_with_cough.extend(y_incl)
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
        status +="."
        print(status)
    f.close()
    librosa.output.write_wav(AUDIO_FILE+"_coughs_added.wav", np.array(y_ges_piece_with_cough),sr=SAMPLERATE)
#######################################################################################################################


############ get the manually labeled snipets as advanced test data ###################################################
if flag_create_manual_labeled:
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
print(0)
f.close()
#######################################################################################################################