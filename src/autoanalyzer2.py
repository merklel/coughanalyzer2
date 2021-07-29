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
import scipy.stats


flag_plot=False

def onset_test(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    # Or compute pulse with an alternate prior, like log-normal
    prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1)
    pulse_lognorm = librosa.beat.plp(onset_envelope=onset_env, sr=sr,
                                     prior=prior)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr)
    ax = plt.subplot(3, 1, 1)
    librosa.display.specshow(librosa.power_to_db(melspec,
                                                 ref=np.max),
                             x_axis='time', y_axis='mel')
    plt.title('Mel spectrogram')
    plt.subplot(3, 1, 2, sharex=ax)
    plt.plot(librosa.times_like(onset_env),
             librosa.util.normalize(onset_env),
             label='Onset strength')
    plt.plot(librosa.times_like(pulse),
             librosa.util.normalize(pulse),
             label='Predominant local pulse (PLP)')
    plt.title('Uniform tempo prior [30, 300]')
    plt.subplot(3, 1, 3, sharex=ax)
    plt.plot(librosa.times_like(onset_env),
             librosa.util.normalize(onset_env),
             label='Onset strength')
    plt.plot(librosa.times_like(pulse_lognorm),
             librosa.util.normalize(pulse_lognorm),
             label='Predominant local pulse (PLP)')
    plt.title('Log-normal tempo prior, mean=120')
    plt.legend()
    plt.xlim([30, 35])
    plt.tight_layout()

    # PLP local maxima can be used as estimates of beat positions.

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))
    ax = plt.subplot(2, 1, 1)
    times = librosa.times_like(onset_env, sr=sr)
    plt.plot(times, librosa.util.normalize(onset_env),
             label='Onset strength')
    plt.vlines(times[beats], 0, 1, alpha=0.5, color='r',
               linestyle='--', label='Beats')
    plt.legend(frameon=True, framealpha=0.75)
    plt.title('librosa.beat.beat_track')
    # Limit the plot to a 15-second window
    plt.subplot(2, 1, 2, sharex=ax)
    times = librosa.times_like(pulse, sr=sr)
    plt.plot(times, librosa.util.normalize(pulse),
             label='PLP')
    plt.vlines(times[beats_plp], 0, 1, alpha=0.5, color='r',
               linestyle='--', label='PLP Beats')
    plt.legend(frameon=True, framealpha=0.75)
    plt.title('librosa.beat.plp')
    plt.xlim(30, 35)
    ax.xaxis.set_major_formatter(librosa.display.TimeFormatter())
    plt.tight_layout()
    #plt.show()

def detect_energy_rise(df_spectrogram):
    df_spectrogram = df_spectrogram[df_spectrogram.index>2000]
    sumdiff = df_spectrogram.sum(axis=0).diff()
    stdabw = df_spectrogram.std(axis=0)
    # sumdiff = sumdiff.rolling(2).sum()
    return sumdiff, stdabw

# AUDIO_FILE = "/home/ga36raf/Documents/coughanalyzer/Joseph Haydn - Piano Concerto No  11 in D major, Hob  XVIII_11 - Mikhail Pletnev/Joseph Haydn - Piano Concerto No. 11 in D major, Hob. XVIII_11 - Mikhail Pletnev (152kbit_Opus).ogg"
# AUDIO_FILE = "/home/ga36raf/Documents/coughanalyzer/Khatia Buniatishvili Joseph Haydn Piano Concerto No 11 in D major, Hob XVIII 11/Khatia Buniatishvili Joseph Haydn Piano Concerto No 11 in D major, Hob XVIII 11 (152kbit_Opus).ogg"
AUDIO_FILE="/home/ga36raf/Documents/coughanalyzer/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009/Martha Argerich_ Ravel - Piano Concerto in G Major _ Nobel Prize Concert 2009 (152kbit_Opus).ogg"
AUDIO_FILE="/home/ga36raf/Documents/coughanalyzer/data/Rachmaninoff_ Piano Concerto No  3 - Anna Fedorova - Live concert HD/Rachmaninoff_ Piano Concerto No. 3 - Anna Fedorova - Live concert HD (152kbit_Opus).ogg"
SAMPLERATE = 22000
CHUNKSIZE = 2 # seconds

# Load audio file
y, sr = librosa.load(AUDIO_FILE, sr=SAMPLERATE, mono=True)
N_audio = len(y)

# overall beat estimation
hop_length = 512
win_length=1000
onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate = np.mean)
overall_tempo, overall_beats = librosa.beat.beat_track(onset_envelope=onset_env, sr = sr, hop_length=hop_length)

overall_pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=1000)
overall_beats_plp = np.flatnonzero(librosa.util.localmax(overall_pulse))
overall_times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
df_overall_beats = pd.DataFrame({"onset_env": onset_env, "times":overall_times, "overall_pulse":overall_pulse})
df_overall_beats.index=df_overall_beats["times"]

df_raw = pd.DataFrame({"time": librosa.times_like(y, sr=sr, hop_length=1), "y":y})
df_raw.index=df_raw["time"]
# iterate slices ############################################
# slice to husten
#   m   s
#   7   25
#   12  59
#   13  20
# flag_husten   = [1, 0,  1,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1,  1,  1,  1,  0]
# husten_minute = [7, 7,  7,  12, 12, 12, 13, 13, 13, 14, 14, 15, 17, 20, 8,  8,  21]
# husten_second = [29,33, 45, 58, 12, 19, 3,  19, 23, 6,  9,  32, 32, 18, 20, 32, 20]
# duration      = [1, 2,  2,  2,  3,  2,  2,  2,  2,  2,  2,  2,  2,  1,  9,  5,  2]

flag_husten = [-1]*len(range(0,N_audio-SAMPLERATE*CHUNKSIZE, SAMPLERATE*CHUNKSIZE))

DURATION = 5
distances=[]
energys_min_max = []
h_mins=[]
h_secs=[]
# for i in range(0,len(husten_minute)):
for i in range(0,N_audio-SAMPLERATE*CHUNKSIZE, SAMPLERATE*CHUNKSIZE):

    # h_min = husten_minute[i]
    # h_sec = husten_second[i]

    # h_start = h_min*60*SAMPLERATE + h_sec*SAMPLERATE
    # h_ende = h_start + duration[i]*SAMPLERATE

    h_start = i
    h_ende = i+SAMPLERATE*CHUNKSIZE

    h_min = np.floor(h_start/SAMPLERATE/60)
    h_sec = (h_start/SAMPLERATE)%60
    h_mins.append(h_min)
    h_secs.append(h_sec)

    # slice audio
    y_husten = y[h_start:h_ende]

    df_y_husten=df_raw[(df_raw["time"]>=h_start/SAMPLERATE) & (df_raw["time"]<h_ende/SAMPLERATE)]["y"]

    y_husten = df_y_husten.values

    # fft
    y_husten_fft = fft(y_husten)
    N = len(y_husten)
    y_husten_fft_proc = 2.0 / N * np.abs(y_husten_fft[0:N // 2])
    xfft = np.linspace(0.0, N // 2, N // 2)

    # spectrogram
    fxx,txx,Sxx = signal.spectrogram(y_husten, SAMPLERATE, window=('tukey', 0.1), nperseg=500, noverlap=50, scaling="spectrum", mode="magnitude")
    df_spectrogram = pd.DataFrame(Sxx, index=fxx, columns=txx)
    # detect rise in spectogram-energy
    sumdiff, stdabw = detect_energy_rise(df_spectrogram)
    energys_min_max.append( (np.max(sumdiff)+abs(np.min(sumdiff))) / np.std(sumdiff))

    ######## Beat estiamtion ####################################################################
    # hop_length = 2000
    onset_env = librosa.onset.onset_strength(y_husten, sr=sr, aggregate = np.mean)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr = sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)


    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr, hop_length=hop_length, win_length=150)
    beats_plp = np.flatnonzero(librosa.util.localmax(pulse))

    t=overall_times[overall_beats_plp][(overall_times[overall_beats_plp]>=h_start/SAMPLERATE) & (overall_times[overall_beats_plp]<h_ende/SAMPLERATE)]
    t2 = df_overall_beats[(df_overall_beats.index >= h_start/SAMPLERATE) & (df_overall_beats.index < h_ende/SAMPLERATE)]["times"]

    #dtw beats
    distance, path = fastdtw(t-int(t[0]), times[beats_plp], dist=euclidean)
    local_beats = times[beats_plp]
    global_beats = t-int(t[0])

    idx = librosa.util.match_events(local_beats, global_beats)
    zipped = zip(local_beats, global_beats[idx])
    ds = []
    for z in zipped:
        ds.append(abs(z[0]-z[1]))

    distance = np.max(ds)
    distances.append(distance)

    # analyze
    df_fft = pd.DataFrame({"frequency": xfft, "amplitude": y_husten_fft_proc})
    amp_4_8 = df_fft[(df_fft["frequency"] > 4000) & (df_fft["frequency"] <8000)].sum()["amplitude"]

    #chromagramm
    S = np.abs(librosa.stft(y_husten))
    chroma = librosa.feature.chroma_stft(S=S, sr=SAMPLERATE)

    # plot
    if flag_plot:
        f, axs = plt.subplots(5,1)
        axs[0].plot(y_husten)

        axs[1].plot(xfft, y_husten_fft_proc)
        axs[1].set_xlim(0,4000)
        axs[1].set_ylim(0,0.01)
        axs[1].grid()

        axs[2].pcolormesh(txx, fxx, Sxx, norm=cl.LogNorm())
        axs[2].set_ylabel('Frequency [Hz]')
        axs[2].set_xlabel('Time [sec]')
        axs[2].set_ylim(0,8000)

        # axs[3]: chromagram
        # librosa.display.specshow(chroma, y_axis='chroma', ax=axs[3], x_axis='time')
        axs[3].plot(sumdiff, label="sumdiff")
        axs[3].plot(stdabw, label="std")
        axs[3].legend()

        # axs[4]
        # librosa.display.specshow(tempogram, sr=sr, ax=axs[4], hop_length=hop_length, x_axis = 'time', y_axis = 'tempo')
        # axs[4].axhline(tempo, color='w', linestyle='--', alpha=1, label = 'Estimated tempo={:g}'.format(tempo))
        # axs[4].legend(frameon=True, framealpha=0.75)

        axs[4].plot(times, librosa.util.normalize(pulse), color="b", label = 'local onset')
        axs[4].vlines(times[beats_plp], 0, 1, alpha=0.5, color='b', linestyle = '--', label = 'local Beats')
        axs[4].vlines(t-int(t[0]), 0, 1, alpha=0.5, color='r', linestyle='--', label='overall Beats')
        # axs[4].plot(t2-int(t2.values[0]), df_overall_beats[(df_overall_beats.index >= h_start/SAMPLERATE) & (df_overall_beats.index < h_ende/SAMPLERATE)]["onset_env"], label="overall onset", color="r")
        axs[4].plot(t2-int(t2.values[0]), df_overall_beats[(df_overall_beats.index >= h_start/SAMPLERATE) & (df_overall_beats.index < h_ende/SAMPLERATE)]["overall_pulse"], label="overall pulse", color="r")
        axs[4].legend(frameon=True, framealpha=0.75)

        f.suptitle("Timecode: {}:{}. Husten: {}. Thresh: {}".format(h_min, h_sec, -1, distance))


print("go")
file_gen_label = "gern_label.txt"
f=open(file_gen_label,"w")
for idx,fg in enumerate(flag_husten):
    if round(distances[idx],3) > 0.1 and round(energys_min_max[idx],3) > 10:
        high = "*"
        print("{}: H: {}. Distance: {}{}. EnergyMinMax: {}. {}:{}".format(idx, fg, high, round(distances[idx], 3),
                                                                          round(energys_min_max[idx], 3),
                                                                          int(h_mins[idx]), int(h_secs[idx])))

        f.write("{}\t{}\tD{}_E{}".format(h_mins[idx] * 60 + h_secs[idx], h_mins[idx] * 60 + h_secs[idx] + 2,
                                         round(distances[idx], 3), round(energys_min_max[idx], 3)))

    else:
        high = ""
f.close()
#plt.show()
print(0)










