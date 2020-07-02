import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold

import mne
from mne import Epochs, pick_types, events_from_annotations, concatenate_epochs
from mne.io import read_raw_edf, read_raw_gdf
from mne.channels import make_standard_montage

from mne.decoding import CSP , Vectorizer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from pathfile import PATHfile
from epoch_raw import Epoch_raw, Setting_file
import pickle_make
import time_map

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)

    svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)

    scores = cross_val_score(svm, data40, l_labels, cv=cv, n_jobs=1)

    print("Classification accuracy: {}" .format(np.mean(scores)))

    return np.mean(scores)

tmin, tmax = -1., 4.
day, name, trial, task_num, path, C, gamma, n_components, time = Setting_file().set_file()


if task_num == 2:
    event_id = dict(Left=1, Right=2) # map event IDs to tasks
    target_names = ['left','right']

elif task_num == 3:
    event_id = dict(Left=1, Right=2, Another=3)
    target_names = ['left', 'right', 'Another']

if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
        (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]

epochs = []
for path, event in path_b:
    raw = read_raw_edf(path, stim_channel=False, preload=True)
    montage = make_standard_montage('standard_1020')
    raw.set_montage(montage)
    epochs.append(Epoch_raw.Epochs_raw(raw, event, event_id, 7., 30., tmin, tmax))
epochs = concatenate_epochs(epochs)
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
labels = epochs.events[:, -1]

scores = []
epochs_data = epochs.get_data()
epochs_data_train = epochs_train.get_data()

cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)
scaler = preprocessing.StandardScaler()
vectorizer = Vectorizer()
csp = CSP(n_components = n_components, reg=None, log=True, norm_trace=False, transform_into='average_power')
svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)

epochs_data_train = csp.fit_transform(epochs_data_train, labels)
epochs_data_train = vectorizer.fit_transform(epochs_data_train, labels)
epochs_data_train = scaler.fit_transform(epochs_data_train, labels)
scores = cross_val_score(svm, epochs_data_train, labels, cv=cv, n_jobs=1)

class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                          class_balance))

# plot CSP patterns estimated on full data for visualization
csp.fit_transform(epochs_data, labels)

csp.plot_patterns(epochs.info, ch_type='eeg', size=1.5)                                                         
