import librosa
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import signal
import numpy as np
import pandas as pd
import math

def plotter(y_husten, y_husten_fft, N, SAMPLERATE, start, end, flag_husten):
    # spectrogram
    fxx, txx, Sxx = signal.spectrogram(y_husten, SAMPLERATE, window=('tukey', 0.25), nperseg=1000, noverlap=500)

    # plot
    f, axs = plt.subplots(3, 1)
    f.suptitle("Timecode: {}:{} to {}:{}. Husten: {}".format(math.floor(start/SAMPLERATE/60),(start/SAMPLERATE)%60, math.floor(end/SAMPLERATE/60),(end/SAMPLERATE)%60, flag_husten))
    axs[0].plot(y_husten)

    xfft = np.linspace(0.0, N // 2, N // 2)
    axs[1].plot(xfft, 2.0 / N * np.abs(y_husten_fft[0:N // 2]))
    axs[1].set_xlim(0, 1000)
    axs[1].set_ylim(0, 0.01)
    axs[1].grid()

    axs[2].pcolormesh(txx, fxx, Sxx)
    axs[2].set_ylabel('Frequency [Hz]')
    axs[2].set_xlabel('Time [sec]')
    axs[2].set_ylim(0, 1500)

AUDIO_FILE = "/home/ga36raf/Documents/coughanalyzer/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).ogg"

SAMPLERATE = 48000
# Load audio file
y, sr = librosa.load(AUDIO_FILE, sr=SAMPLERATE, mono=True)

# cut applaus
y = y[0:22*60*SAMPLERATE+15*SAMPLERATE]

CHUNKSIZE = 2 # seconds
N_audio = len(y)
# N_audio=2000000
# loop through audio
for i in range(0,N_audio-SAMPLERATE*CHUNKSIZE, SAMPLERATE*CHUNKSIZE):

    # slice
    start = i
    end = start + SAMPLERATE*CHUNKSIZE
    if end>N_audio: end = N_audio
    y_chunk = y[start:end]
    N_chunk = len(y_chunk)
    print(start, end)

    #fft
    y_chunk_fft = fft(y_chunk)
    y_chunk_fft_proc = 2.0 / N_chunk * np.abs(y_chunk_fft[0:N_chunk // 2])
    xfft = np.linspace(0.0, N_chunk // 2, N_chunk // 2)
    df_fft = pd.DataFrame({"frequency": xfft, "amplitude": y_chunk_fft_proc})

    # analyze
    df_fft_200 = df_fft[df_fft["frequency"] < 200]
    df_fft_200_thresh = df_fft_200[df_fft_200["amplitude"] > 0.001]

    n_thresh  = int(df_fft_200_thresh["amplitude"].count())

    print(n_thresh)

    if n_thresh >50:
        plotter(y_chunk, y_chunk_fft, N_chunk, SAMPLERATE, start, end, n_thresh)

    # plotter(y_chunk, y_chunk_fft, N_chunk, SAMPLERATE, start, end, res)
    #
    # if i > 15*SAMPLERATE:
    #     break


