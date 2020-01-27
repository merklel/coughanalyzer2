import glob
import matplotlib
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.image
import os

SAMPLERATE=22000

FOLDER_COUGH_EXAMPLES = "data/coughexamples/"
FOLDER_COUGH_CHANGED = "data/coughexamples_changed/"

FOLDER_OUTPUT = "data/cough_learn_histo/onlycough"

cough_examples_files = glob.glob(FOLDER_COUGH_EXAMPLES+"/*.wav")
cough_changed_files = glob.glob(FOLDER_COUGH_CHANGED+"/*.wav")

for idx, cough in enumerate(cough_examples_files):
    cough_fn = os.path.split(cough)[1]
    x, sr = librosa.load(cough, sr=SAMPLERATE, mono=True)

    fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0, scaling="spectrum", mode="magnitude")

    matplotlib.image.imsave(FOLDER_OUTPUT+"/a_" + cough_fn + ".png".format(idx), Sxx)

for idx, cough in enumerate(cough_changed_files):
    x, sr = librosa.load(cough, sr=SAMPLERATE, mono=True)
    cough_fn = os.path.split(cough)[1]
    fxx, txx, Sxx = signal.spectrogram(x, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=0, scaling="spectrum", mode="magnitude")

    matplotlib.image.imsave(FOLDER_OUTPUT+"/" + cough_fn + ".png".format(idx), Sxx)
