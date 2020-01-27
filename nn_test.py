# Code source: Brian McFee
# License: ISC

##################
# Standard imports
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

cough_file = "/home/ga36raf/Documents/coughanalyzer/data/cough_added/cough_added_880000.wav"
cough_file2 = "/home/ga36raf/Documents/coughanalyzer/data/cough_added/cough_added_924000.wav"
no_cough_file = "/home/ga36raf/Documents/coughanalyzer/data/untouched/untouched_880000.wav"
no_cough_file2 = "/home/ga36raf/Documents/coughanalyzer/data/untouched/untouched_924000.wav"
no_cough_file3 = "/home/ga36raf/Documents/coughanalyzer/data/untouched/untouched_968000.wav"
no_cough_file4 = "/home/ga36raf/Documents/coughanalyzer/data/untouched/untouched_1012000.wav"
no_cough_file5 = "/home/ga36raf/Documents/coughanalyzer/data/untouched/untouched_1056000.wav"
no_cough_file6 = "/home/ga36raf/Documents/coughanalyzer/data/untouched/untouched_1100000.wav"
cough_file_manual = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_27.wav"
cough_file_manual2 = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_28.wav"
cough_file_manual3 = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_38.wav"
cough_file_manual4 = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_39.wav"
cough_file_manual5 = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_40.wav"
cough_file_manual6 = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_41.wav"
cough_file_manual7 = "/home/ga36raf/Documents/coughanalyzer/data/manual_cough/manual_cough_42.wav"

y, sr = librosa.load(cough_file_manual3)


# And compute the spectrogram magnitude and phase
S_full, phase = librosa.magphase(librosa.stft(y, n_fft=2000, hop_length=300))

idx = slice(*librosa.time_to_frames([0, 2], sr=sr))

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
S_background = mask_i * S_full

# sphinx_gallery_thumbnail_number = 2

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                         y_axis='linear', sr=sr)
plt.title('Full spectrum')
plt.colorbar()

plt.subplot(3, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                         y_axis='linear', sr=sr)
plt.title('Background')
plt.colorbar()
plt.subplot(3, 1, 3)
librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                         y_axis='linear', x_axis='time', sr=sr)
plt.title('Foreground')
plt.colorbar()
plt.tight_layout()
plt.show()

