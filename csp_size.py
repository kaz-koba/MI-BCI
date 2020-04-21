import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import math

from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit, cross_val_score, StratifiedKFold

import mne
from mne import concatenate_epochs

from mne.decoding import Vectorizer
from mne.io import read_raw_edf
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from pathfile import PATHfile
from epoch_raw import Epoch_raw
import pickle_make
from CSPmodel import CSP

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)
    n_csp = trial.suggest_int('n_csp', 1, 40)

    svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)
    X = csplist.T[:n_csp].T
    scores = cross_val_score(svm, X, labels, cv=cv, n_jobs=1)

    print("Classification accuracy: {}" .format(np.mean(scores)))

    return np.mean(scores)

def seiki(x):
    return (math.exp(-x**2/2))/math.sqrt(2*math.pi)

def sigmoid(x, p):
    return 2.0 / (1.0 + np.exp(x/p))

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
csplist = np.empty((120,0))
evlist = np.empty((0))
for band, fmin, fmax, mag in iter_freqs:
    epochs = []
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    csp = CSP(n_components = int(inifile.get('setting', 'n_components')), reg=None, log=True, 
                norm_trace=False, transform_into='average_power')
    # (re)load the data to save memory
    for path, event in path_b:
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        epochs.append(Epoch_raw.Epochs_raw(raw, event, event_id, fmin, fmax, tmin, tmax))
    epochs = concatenate_epochs(epochs)
    labels = epochs.events[:, -1]
    # remove evoked response
    #epochs.subtract_evoked()
    epochs_train = epochs.copy().crop(tmin=0.25, tmax=1.25)
    epochs_data_train = epochs_train.get_data()
    x, ev = csp.fit(epochs_data_train, labels).transform(epochs_data_train)
    x = vectorizer.fit_transform(x, labels)
    x = scaler.fit_transform(x, labels)
    x *= mag
    print(x[0])
    csplist = np.hstack((csplist, x))
    evlist = np.hstack((evlist, ev))
    print(evlist)
    csp_map.append(x)
    del epochs

print(evlist)
ix = np.argsort(evlist)[::-1]
print(ix)
csplist = np.array(csplist[:, ix])

"""
for i in range(len(csplist.T)):
    csplist.T[i] *= sigmoid(i, 5)

for i in range(len(csplist.T)):
    csplist.T[i] *= seiki(i)

for i in range(len(csplist.T)):
    csplist.T[i] *= np.exp(-((i)**2) / (2*(5**2)))
print(csplist[0])

for i in range(len(csplist.T)):
    csplist.T[i] *= 1/(i+1)
"""
csplist = csplist.T[:20].T
print(csplist.shape)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
svm = SVC(C=study.best_trial.params['C'], gamma = study.best_trial.params['gamma'], kernel='rbf', cache_size=100)
csplist = csplist.T[:study.best_params['n_csp']].T

print("Classification accuracy: {}" .format(study.best_value))
for key, value in study.best_trial.params.items():
    print('    {}: {}'.format(key, value))

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

preds = np.empty(len(labels))
for train, test in cv.split(csplist, labels):
    svm.fit(csplist[train], labels[train])
    preds[test] = svm.predict(csplist[test])

# Classification report
report = classification_report(labels, preds, target_names=target_names)
print(report)

cm = confusion_matrix(preds, labels)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = pd.DataFrame(data=cm_normalized, index=target_names, columns=target_names)
sns.heatmap(cm_normalized, annot=True, cmap='Blues', square=True)
if path == "day":
    trial_name = day
else:
    trial_name = trial
plt.savefig('figure/confusion_matrix_{}_{}_{}.png' .format(name, day, trial_name))
plt.show()