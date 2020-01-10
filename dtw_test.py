import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from scipy.fft import fft
from scipy import signal
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

AUDIO_FILE_1 = "/home/ga36raf/Documents/coughanalyzer/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).ogg"
AUDIO_FILE_2 = "/home/ga36raf/Documents/coughanalyzer/Khatia Buniatishvili Joseph Haydn Piano Concerto No 11 in D major, Hob XVIII 11/Khatia Buniatishvili Joseph Haydn Piano Concerto No 11 in D major, Hob XVIII 11 (152kbit_Opus).ogg"

SAMPLERATE = 48000

# Load audio file
y1, sr1 = librosa.load(AUDIO_FILE_1, sr=SAMPLERATE, mono=True)
y2, sr2 = librosa.load(AUDIO_FILE_2, sr=SAMPLERATE, mono=True)

y1 = y1[6*SAMPLERATE:SAMPLERATE*14]
y2 = y2[10*SAMPLERATE:SAMPLERATE*18]

distance, path = fastdtw(y1, y2, dist=euclidean)
print(distance)

f, axs = plt.subplots(3,1)
f.suptitle("dtw test")
axs[0].plot(y1)
axs[1].plot(y2)
axs[2].plot(path)
plt.show()