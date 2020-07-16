import configparser
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import mne
from mne import concatenate_epochs
from mne.io import read_raw_edf
from mne.datasets import sample
from mne.stats import bonferroni_correction, fdr_correction

from pathfile import PATHfile
from epoch_raw import Epoch_raw, Setting_file

print(__doc__)

tmin, tmax =-1., 4.
event_id = [1] #1-left 2-right 3-another
day, name, trial, task_num, path, C, gamma, n_components, time = Setting_file().set_file()

if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
        (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]

#   Setup for reading the raw data
channel = 'C3'
include = [channel]

epochs = []

for path, event in path_b:
    raw = read_raw_edf(path, stim_channel=False, preload=True)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, include=include, exclude='bads')
    epochs.append(Epoch_raw.Epochs_raw(raw, event, event_id, tmin = tmin, tmax = tmax, picks=picks))    

epochs = concatenate_epochs(epochs)

X = epochs.get_data()  # as 3D matrix
X = X[:, 0, :]  # take only one channel to get a 2D array

T, pval = stats.ttest_1samp(X, 0)
alpha = 0.05

n_samples, n_tests = X.shape
threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)

reject_bonferroni, pval_bonferroni = bonferroni_correction(pval, alpha=alpha)
threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)

reject_fdr, pval_fdr = fdr_correction(pval, alpha=alpha, method='indep')
threshold_fdr = np.min(np.abs(T)[reject_fdr])

times = 1e3 * epochs.times

plt.close('all')
plt.plot(times, abs(T), 'k', label='T-stat')
xmin, xmax = plt.xlim()
plt.hlines(threshold_uncorrected, xmin, xmax, linestyle='--', colors='k',
            label='p=0.05 (uncorrected)', linewidth=2)
plt.hlines(threshold_bonferroni, xmin, xmax, linestyle='--', colors='r',
            label='p=0.05 (Bonferroni)', linewidth=2)
plt.hlines(threshold_fdr, xmin, xmax, linestyle='--', colors='b',
            label='p=0.05 (FDR)', linewidth=2)
plt.legend()
plt.xlabel("Time (ms)")
plt.ylabel("T-stat")
plt.show()