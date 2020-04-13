import configparser
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

import mne
from mne import Epochs, concatenate_epochs
from mne.io import read_raw_edf
from mne.datasets import sample
from mne.stats import bonferroni_correction, fdr_correction

from pathfile import PATHfile

print(__doc__)

def epoch_raw(path, event):
    raw = read_raw_edf(path, stim_channel=False, preload=True)
    event = pd.read_csv(event, header=None)
    events = event.values
    picks = mne.pick_types(raw.info,  include=include, eeg=True, exclude='bads')
    epochs = Epochs(raw, events, event_id, tmin - 0.5, tmax + 0.5, proj=True, picks=picks, 
                    baseline=None, preload=True, event_repeated='drop')
    del raw

    return epochs

tmin, tmax =-1., 4.
event_id = [1] #1-left 2-right 3-another
inifile = configparser.ConfigParser()
inifile.read('./parameter.ini', 'UTF-8')

day = inifile.get('setting', 'day')
name = inifile.get('setting', 'name')
trial = inifile.get('setting', 'trial')

path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1"))]
#        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
#       (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]

#   Setup for reading the raw data
channel = 'C3'
include = [channel]

epochs = []
for path, event in path_b:
    epochs.append(epoch_raw(path, event))    

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