import configparser
import matplotlib.pyplot as plt
import pandas as pd
import sys

import mne
from mne import concatenate_epochs
from mne.io import read_raw_edf
from mne.stats import permutation_cluster_test
from mne.datasets import sample

from pathfile import PATHfile
from epoch_raw import Epoch_raw, Setting_file

if __name__ == "__main__":
    # set epoching parameters
    tmin, tmax =-1., 4.
    fmin, fmax = 4, 45
    event_id = [1, 2]
    day, name, trial, task_num, path, C, gamma, n_components, time = Setting_file().set_file()

    if path == "day":
        path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
                (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
            (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
    elif path == "trial":
        path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]


    #   Setup for reading the raw data
    channel = 'C5'
    include = [channel]
    epochs1 = []
    event_id = 1
    for path, event in path_b:
        event = pd.read_csv(event, header=None)
        events = event.values
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        picks = mne.pick_channels(raw.info["ch_names"], include)
        epochs1.append(mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,baseline=(None, 0)))
    epochs1 = concatenate_epochs(epochs1)
    condition1 = epochs1.get_data()  # as 3D matrix
    print(condition1.shape)
    condition1 = condition1[:, 0, :]  # take only one channel to get a 2D array
    print(condition1.shape)
    mean1 = condition1.mean(axis=0)
    print(mean1.shape)

    epochs2 = []
    event_id = 2
    for path, event in path_b:
        event = pd.read_csv(event, header=None)
        events = event.values
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        picks = mne.pick_channels(raw.info["ch_names"], include)
        epochs2.append(mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,baseline=(None, 0)))
    epochs2 = concatenate_epochs(epochs2)
    condition2 = epochs2.get_data()  # as 3D matrix
    condition2 = condition2[:, 0, :]  # take only one channel to get a 2D array
    mean2 = condition2.mean(axis=0)

    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([condition1, condition2], n_permutations=1000,
                                threshold=threshold, tail=1, n_jobs=1)

    times = epochs1.times


    plt.close('all')
    """
    plt.subplot(211)
    plt.title('Channel : ' + channel)
    plt.plot(times, mean1 - mean2,
            label="ERP Contrast (Event 1 - Event 2)")
    plt.ylabel("Hz")
    plt.legend()
    plt.subplot(212)
    """
    h = []
    count = 0
    for i_c, c in enumerate(clusters):
        c = c[0]
        if cluster_p_values[i_c] <= 0.05:
            h = plt.axvspan(times[c.start], times[c.stop - 1],
                            color='r', alpha=0.3)
            count+=1
        else:
            plt.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3),
                        alpha=0.3)
    hf = plt.plot(times, T_obs, 'g')
    plt.legend((h, ), ('cluster p-value < 0.05', ))
    plt.xlabel("time (ms)")
    plt.ylabel("f-values")
    if path == 'day':
        pltpath = 'figure/' + day + '-' + name + '-' + channel + '.png'
    else:
        pltpath = 'figure/' + day + '-' + name + '-' + trial + '-' + channel + '.png'
    plt.savefig(pltpath)
    plt.show()