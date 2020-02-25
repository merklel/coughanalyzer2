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
import glob


FOLDER_COUGHS = "data/coughexamples/"

# settings
SAMPLERATE = 16000
CHUNKSIZE = 2 # seconds

prefactor_volume = 0.25
factor_volume = [0.2 * prefactor_volume, 0.25*prefactor_volume, 0.3*prefactor_volume]

factor_volume = [0.25]

stretch_factors = [0.9, 0.97, 1, 1.03, 1.1]
#stretch_factors = [1]
pitch_steps = [-4, -1, 0, 1, 4]
#pitch_steps = [0]

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