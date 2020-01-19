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

# flags
# 1. create cough and no cough data
flag_create_cough_data = False
# 2. create manual labeled snippets
flag_create_manual_labeled = False
# 3. create crossvalid files
flag_create_crossvalid_files=True

# settings
FOLDER_COUGHS = "data/coughexamples/"
FOLDER_CROSSVALID = "data/crossvalidation"
AUDIO_FILE="/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/Rachmaninoff_ Piano Concerto No. 3 - Anna Fedorova - Live concert HD (152kbit_Opus).ogg.wav"
MAN_LABEL_FILE = "/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/manual_labeled_hustenandsounds3.txt"
CUT_FILE="data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/cutfile.txt"

database = [
    {"audio_file": AUDIO_FILE, "man_label_file": MAN_LABEL_FILE, "cut_file": CUT_FILE},
    {"audio_file": "data/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).wav",
     "man_label_file": "data/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/manual_labeled.txt",
     "cut_file": "data/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/cut_file.txt"},
    {"audio_file": "data/shostakovich CD08 - Symphony 10/shostakovich_10_studio.wav",
     "man_label_file": None,
     "cut_file": None},
    {"audio_file": "data/Daniel Barenboim - Beethoven Concerto for Violin and Orchestra in D Major, Op. 61 (2017)/beethoven_violin_studio.wav",
     "man_label_file": None,
     "cut_file": None}
]

database_crossvalid = [
    {"audio_file": "/home/ga36raf/Documents/coughanalyzer/data/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009 (152kbit_Opus).wav",
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

SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds

factor_volume_music = 1

f=0.3
factor_volume = [0.25*f, 0.3*f]
#factor_volume=[1]

stretch_factors = [0.95, 1, 1.1]
#stretch_factors = [1]
pitch_steps = [-1, 0, 5]
#pitch_steps = [0]


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
    print("saving snippets...")
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
        y_incl = factor_volume_music*(y[i:i+CHUNKSIZE*SAMPLERATE]) + (y_cough)
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
print("saving crossvalidation files")
if flag_create_crossvalid_files:
    for db_cv in database_crossvalid:
        y, sr = librosa.load(db_cv["audio_file"], sr=SAMPLERATE, mono=True)

        filename = os.path.split(db_cv["audio_file"])[1].split(".")[0]

        #create dirs if not there
        if not os.path.exists(FOLDER_CROSSVALID + "/{}".format(filename)):
            os.makedirs(FOLDER_CROSSVALID + "/{}".format(filename))

        N_audio = len(y)
        for i in range(0, N_audio-(CHUNKSIZE*SAMPLERATE), CHUNKSIZE*SAMPLERATE):
            librosa.output.write_wav(FOLDER_CROSSVALID + "/{}/snippet_cv_{}.wav".format(filename, i), np.array(y[i:i+SAMPLERATE*CHUNKSIZE]), sr=SAMPLERATE)
#######################################################################################################################


print("Finished!")
