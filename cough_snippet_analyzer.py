import glob
import matplotlib
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image
import os
from scipy.fft import fft
import numpy as np

SAMPLERATE=22000
CHUNKSIZE=2

FOLDER_COUGH_EXAMPLES = "data/coughexamples/"
FOLDER_COUGH_CHANGED = "data/coughexamples_changed/"

FOLDER_OUTPUT = "data/cough_learn_histo/onlycough"

cough_examples_files = glob.glob(FOLDER_COUGH_EXAMPLES+"/*.wav")
cough_changed_files = glob.glob(FOLDER_COUGH_CHANGED+"/*.wav")

def calculate_fft(y, sr=16000):

    N=len(y)


    y_fft = fft(y, n=CHUNKSIZE*sr)

    y_fft = abs(y_fft[0:int(N/2)]) ** 2


    #ny_fft = abs((y_fft - np.mean(y_fft)) / np.std(y_fft))
    ny_fft=y_fft
    ny_fft = ny_fft / np.max(ny_fft)

    return ny_fft, np.mean(ny_fft)


plt.figure()

for cf in cough_changed_files[0:5]:
	x, sr = librosa.load(cf, sr=SAMPLERATE, mono=True)
	ftt, m = calculate_fft(x, sr=16000)

	plt.plot(list(ftt))
plt.show()
