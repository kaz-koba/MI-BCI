import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import sys

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold

import mne
from mne import concatenate_epochs

from mne.decoding import CSP , Vectorizer
from mne.io import read_raw_edf
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from pathfile import PATHfile
from epoch_raw import Epoch_raw, Setting_file
import pickle_make
import time_map


def fix_labels(i, task_num):
    id = 1
    if i % task_num == 0:
        return task_num 
    while True:
        if i % task_num == id:
            return id
        id += 1

# set epoching parameters
tmin, tmax =-1., 5.
day, name, trial, task_num, path, C, gamma, n_components, time = Setting_file().set_file()


iter_freqs = [
    ('A', 4, 8, 1.0),
    ('B', 8, 12, 1.0),
    ('C',12, 16, 1.0),
    ('D',16, 20, 1.0),
    ('E',20, 24, 1.0),
    ('F',24, 28, 1.0),
    ('G',28, 32, 1.0)
]


if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2"))]
        #(PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]


time_map = time_map.mapping_time(time, task_num)

if task_num == "2":
    event_id = dict(Left=1, Right=2)  # map event IDs to tasks
    target_names = ['left', 'right']
    data40 = np.empty((len(path_b)*40,0))
elif task_num == "3":
    event_id = dict(Left=1, Right=2, Another=3)
    target_names = ['left', 'right', 'Another']
    data40 = np.empty((len(path_b)*60,0))
else:
    print('Error: Please task_num is 2 or 3', file=sys.stderr)
    sys.exit(1)  

epochs_time = []
for tmin, tmax, time_id in time_map:
    epochs = []
    for path, event in path_b:
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        epochs.append(Epoch_raw.Epochs_raw(raw, event, event_id=dict(Left=1), fmin=4, fmax = 32, tmin=tmin, tmax=tmax))
    epochs = concatenate_epochs(epochs)

    epochs_time.append(epochs.get_data().mean(axis=0))

for tmin, tmax, time_id in time_map:
    epochs = []
    for path, event in path_b:
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        epochs.append(Epoch_raw.Epochs_raw(raw, event, event_id=dict(Left=2), fmin=4, fmax = 32, tmin=tmin, tmax=tmax))
    epochs = concatenate_epochs(epochs)

    epochs_time.append(epochs.get_data().mean(axis=0))

print(len(epochs_time))

path_b = [(PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
epochs = []
for path, event in path_b:
    raw = read_raw_edf(path, stim_channel=False, preload=True)
    epochs.append(Epoch_raw.Epochs_raw(raw, event, event_id=event_id, fmin=4, fmax = 32, tmin=0., tmax=1.))
epochs = concatenate_epochs(epochs)
epochs = epochs.get_data().tolist()

for i in range(len(epochs)):
    best_dist
    for simi in epochs_time:
        


