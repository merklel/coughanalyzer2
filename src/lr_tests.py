import librosa
import librosa.display as dp
import matplotlib.pyplot as plt
import matplotlib.colors as cl
from scipy.fft import fft
from scipy import signal
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

AUDIO_FILE_1 = "/home/ga36raf/Documents/coughanalyzer/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).ogg"
SAMPLERATE = 48000

y1, sr1 = librosa.load(AUDIO_FILE_1, sr=SAMPLERATE, mono=True)

# y1 = y1[0:100*SAMPLERATE]


S = np.abs(librosa.stft(y1))
chroma = librosa.feature.chroma_stft(S=S, sr=SAMPLERATE)

plt.figure(figsize=(10, 4))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()