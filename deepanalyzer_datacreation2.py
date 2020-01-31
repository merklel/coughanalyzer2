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
import matplotlib

# settings
SAMPLERATE = 16000
CHUNKSIZE = 2 # seconds

# Folders
FOLDER_COUGHS = "data/coughexamples/"
coughexamples_files = glob.glob(FOLDER_COUGHS+"*.wav")

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
    {"audio_file": "data/raw_studio/Daniel Barenboim - Beethoven Concerto for Violin and Orchestra in D Major, Op. 61 (2017)/beethoven_violin_studio.wav",
     "man_label_file": None,
     "cut_file": None},
    {
        "audio_file":"data/raw_studio/Haydn cd2 n.41, 58/haydn_41_58.wav",
        "man_label_file": None,
        "cut_file": None
    }
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


def check_if_within_cutfile(input_i, df_man_label):
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

def foreground_Separation(y, sr):

    # And compute the spectrogram magnitude and phase
    S_full, phase = librosa.magphase(librosa.stft(y, n_fft=300, hop_length=100))

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

#########################################################################################################################
# Create Train images
#########################################################################################################################

counter = 0
for db in database:

    sr_orig = librosa.core.get_samplerate(db["audio_file"])
    
    stream = librosa.stream(db["audio_file"], block_length=1, frame_length=sr_orig*CHUNKSIZE, hop_length=sr_orig*CHUNKSIZE)

    if db["cut_file"] != None:
        df_cutfile = pd.read_csv(db["cut_file"], sep="\t", header=None)

    for idx, block_y in enumerate(stream):

        if db["cut_file"] != None:
            skipp = check_if_within_cutfile(idx*CHUNKSIZE, df_cutfile)
        else:
            skipp = False

        if not skipp:
            # get random cough sample
            rand_idx_cough = random.randint(0,len(coughexamples_files)-1)
            random_cough_file = coughexamples_files[rand_idx_cough]

            # open the random cough file
            y_cough, sr_cough = librosa.load(random_cough_file, sr=SAMPLERATE, mono=True)

            y = librosa.resample(block_y, sr_orig, SAMPLERATE)

            pre_gap = [0] * random.randint(0, 0.5*SAMPLERATE) # up to half a second pre gap possible
            y_cough = np.append(pre_gap, y_cough)
            y_cough = y_cough[0:CHUNKSIZE*SAMPLERATE]
            if len(y_cough) < CHUNKSIZE * SAMPLERATE:
                y_cough = np.append(y_cough,[0] * ((CHUNKSIZE * SAMPLERATE) - len(y_cough)))

            # print(y_cough)
            print(len(y), len(y_cough))
            # add the two audios and save

            if len(y) == len(y_cough):
                y_incl = 1 * np.array(y) + y_cough

                Sxx_cough = foreground_Separation(y_incl, sr=SAMPLERATE)
                Sxx_no_cough = foreground_Separation(y, sr=SAMPLERATE)
                matplotlib.image.imsave("data/cough_learn_histo/train/no_cough_{}.png".format(counter), Sxx_no_cough, cmap="gray")
                matplotlib.image.imsave("data/cough_learn_histo/train/cough_{}.png".format(counter), Sxx_cough, cmap="gray")

                counter+=1


#########################################################################################################################
# Create Train images
#########################################################################################################################
counter = 0
for db in database_crossvalid:

    db["audio_file"]
    sr_orig = librosa.core.get_samplerate(db["audio_file"])
    
    stream = librosa.stream(db["audio_file"], block_length=1, frame_length=sr_orig*CHUNKSIZE, hop_length=sr_orig*CHUNKSIZE)

    for block_y in stream:

        y = librosa.resample(block_y, sr_orig, SAMPLERATE)
        
        Sxx_no_cough = foreground_Separation(y, sr=SAMPLERATE)
        matplotlib.image.imsave("data/cough_learn_histo/crossvalid/crossvalid_{}.png".format(counter), cmap="gray")

        counter+=1
