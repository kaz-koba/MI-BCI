import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold

import mne
from mne import concatenate_epochs

from mne.decoding import CSP , Vectorizer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from pathfile import PATHfile
from epoch_raw import Epoch_raw
import pickle_make

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)

    svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)

    scores = cross_val_score(svm, data40, labels, cv=cv, n_jobs=1)

    print("Classification accuracy: {}" .format(np.mean(scores)))

    return np.mean(scores)

# set epoching parameters
tmin, tmax =-1., 4.
inifile = configparser.ConfigParser()
inifile.read('./parameter.ini', 'UTF-8')

day = inifile.get('setting', 'day')
name = inifile.get('setting', 'name')
trial = inifile.get('setting', 'trial')
task_num = inifile.get('setting', 'task_num')
path = inifile.get('setting', 'path')


if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
        (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]

if task_num == "2":
    event_id = dict(Left=1, Right=2)  # map event IDs to tasks
    target_names = ['left', 'right']
    data40 = np.empty((len(path_b)*40,0))
elif task_num == "3":
    event_id = dict(Left=1, Right=2, Another=3)
    target_names = ['left', 'right', 'Another']
    data40 = np.empty((len(path_b)*60,0))


#frequency bands
iter_freqs = [
    ('Theta', 4, 7, 1.0),
    ('Alpha', 8, 12, 1.0),
    ('Beta', 13, 25, 1.0),
    ('Gamma', 30, 45, 1.0)
]

#parameters
svm = SVC(C=float(inifile.get('setting', 'C')), gamma = float(inifile.get('setting', 'gamma')), kernel='rbf', 
        cache_size=100)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)

# set epoching parameters
tmin, tmax =-1., 4.

csp_map = list()
for band, fmin, fmax, mag in iter_freqs:
    epochs = []
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    csp = CSP(n_components = int(inifile.get('setting', 'n_components')), reg=None, log=True, 
            norm_trace=False, transform_into='average_power')
    # (re)load the data to save memory
    for path, event in path_b:
        epochs.append(Epoch_raw.Epochs_raw(path, event, event_id, fmin, fmax, tmin, tmax))
    epochs = concatenate_epochs(epochs)
    labels = epochs.events[:, -1]
    # remove evoked response
    #epochs.subtract_evoked()
    epochs_train = epochs.copy().crop(tmin=0.25, tmax=1.25)
    epochs_data_train = epochs_train.get_data()
    x = csp.fit_transform(epochs_data_train, labels)
    csp_map.append(x)
    del epochs

print(csp_map)