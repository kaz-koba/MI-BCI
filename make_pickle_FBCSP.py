import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold

import mne
from mne import Epochs, pick_types, events_from_annotations, concatenate_epochs
from mne.io import read_raw_edf, read_raw_gdf

from mne.decoding import CSP , Vectorizer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from pathfile import PATHfile

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)

    svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)

    scores = cross_val_score(svm, data40, labels, cv=cv, n_jobs=1)

    print("Classification accuracy: {}" .format(np.mean(scores)))

    return np.mean(scores)

def epoch_raw(path, event):
    raw = read_raw_edf(path, stim_channel=False, preload=True)
    event = pd.read_csv(event, header=None)
    events = event.values
    raw.filter(fmin, fmax, n_jobs=1,  
            l_trans_bandwidth=1,  
            h_trans_bandwidth=1)
    epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True, event_repeated='drop')
    del raw

    return epochs

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
"""
iter_freqs = [
    ('Alpha', 8, 12),
    ('Beta', 13, 25),
    ('ALL', 1, 30),
    ('Alpha+Beta', 8, 25)
]
"""
iter_freqs = [
    ('Theta', 4, 7, 1.0),
    ('Alpha', 8, 12, 1.0),
    ('Beta', 13, 25, 1.0),
    ('Gamma', 30, 45, 1.0)
]

#parameters
scaler = preprocessing.StandardScaler()
vectorizer = Vectorizer()
csp = CSP(n_components = int(inifile.get('setting', 'n_components')), reg=None, log=True, norm_trace=False, transform_into='average_power')
svm = SVC(C=float(inifile.get('setting', 'C')), gamma = float(inifile.get('setting', 'gamma')), kernel='rbf', cache_size=100)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)

# set epoching parameters
tmin, tmax =-1., 4.

acc_map = list()
csp_map = list()
vec_map = list()
sca_map = list()

for band, fmin, fmax, mag in iter_freqs:
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    csp = CSP(n_components = int(inifile.get('setting', 'n_components')), reg=None, log=True, norm_trace=False, transform_into='average_power')
    epochs = []
    # (re)load the data to save memory
    for path, event in path_b:
        epochs.append(epoch_raw(path, event))
    epochs = concatenate_epochs(epochs)
    labels = epochs.events[:, -1]
    # remove evoked response
    #epochs.subtract_evoked()
    epochs_train = epochs.copy().crop(tmin=0.25, tmax=1.25)
    epochs_data_train = epochs_train.get_data()
    print(epochs_data_train.shape)
    x = csp.fit_transform(epochs_data_train, labels)
    csp_map.append(csp)
    x = vectorizer.fit_transform(x, labels)
    vec_map.append(vectorizer)
    x = scaler.fit_transform(x, labels)
    sca_map.append(scaler)
    x *= mag
    data40 = np.hstack((data40, x))
    scores = cross_val_score(svm, x, labels, cv=cv, n_jobs=1)
    acc_map.append((band, fmin, fmax, scores))
    del epochs

for freq_name, fmin, fmax, acc in acc_map:
    print("{}({}~{}) Classification accuracy: {}" .format(freq_name, fmin, fmax, np.mean(acc)))

print(data40)
#svm.fit(data40, labels)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
svm = SVC(C=study.best_trial.params['C'], gamma = study.best_trial.params['gamma'], kernel='rbf', cache_size=100)
svm.fit(data40, labels)


print("Classification accuracy: {}" .format(study.best_value))
for key, value in study.best_trial.params.items():
    print('    {}: {}'.format(key, value))

with open('csp_map.pickle', 'wb') as f:
    pickle.dump(csp_map, f)

with open('svm.pickle', 'wb') as f:
    pickle.dump(svm, f)

with open('freqs.pickle', 'wb') as f:
    pickle.dump(iter_freqs, f)

with open('vec.pickle', 'wb') as f:
    pickle.dump(vec_map, f)

with open('sca.pickle', 'wb') as f:
    pickle.dump(sca_map, f)
