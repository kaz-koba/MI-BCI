import configparser
import matplotlib.pyplot as plt
import pandas as pd
import sys

import mne
from mne import concatenate_epochs
from mne.stats import permutation_cluster_test
from mne.datasets import sample

from pathfile import PATHfile
from epoch_raw import Epoch_raw

if __name__ == "__main__":
    # set epoching parameters
    tmin, tmax =-1., 4.
    fmin, fmax = 1, 45
    event_id = [1, 2, 3]
    inifile = configparser.ConfigParser()
    inifile.read('./parameter.ini', 'UTF-8')

    day = inifile.get('setting', 'day')
    name = inifile.get('setting', 'name')
    trial = inifile.get('setting', 'trial')

    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]


    #   Setup for reading the raw data
    channel = 'FC3'
    include = [channel]

    epochs1 = []
    event_id = 1
    for path, event in path_b:
        epochs1.append(Epoch_raw.Epochs_raw(path, event, event_id, fmin, fmax, tmin, tmax, include))    
    epochs1 = concatenate_epochs(epochs1)
    condition1 = epochs1.get_data()  # as 3D matrix
    print(condition1.shape)
    condition1 = condition1[:, 0, :]  # take only one channel to get a 2D array
    mean1 = condition1.mean(axis=0)

    epochs2 = []
    event_id = 2
    for path, event in path_b:
        epochs2.append(Epoch_raw.Epochs_raw(path, event, event_id, fmin, fmax, tmin, tmax, include))    
    epochs2 = concatenate_epochs(epochs2)
    condition2 = epochs2.get_data()  # as 3D matrix
    condition2 = condition2[:, 0, :]  # take only one channel to get a 2D array
    mean2 = condition2.mean(axis=0)
    """
    epochs3 = []
    event_id = 3
    for path, event in path_b:
        epochs3.append(epoch_raw(path, event, event_id))    
    epochs3 = concatenate_epochs(epochs3)
    condition3 = epochs3.get_data()  # as 3D matrix
    condition3 = condition3[:, 0, :]  # take only one channel to get a 2D array
    mean3 = condition3.mean(axis=0)
    """

    threshold = 6.0
    T_obs, clusters, cluster_p_values, H0 = \
        permutation_cluster_test([condition1, condition2], n_permutations=1000,
                                threshold=threshold, tail=1, n_jobs=1)

    times = epochs1.times


    plt.close('all')
    plt.subplot(211)
    plt.ylim(-30, 30)
    plt.title('Channel : ' + channel)
    plt.plot(times, mean1 - mean2,
            label="ERF Contrast (Event 1 - Event 2)")
    plt.ylabel("MEG (T / m)")
    plt.legend()
    plt.subplot(212)
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
    plt.show()
    pltpath = 'figure/' + day + '-' + name + '.png'
    plt.savefig(pltpath)

