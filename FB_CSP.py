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

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)

    svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)

    scores = cross_val_score(svm, data40, labels, cv=cv, n_jobs=1)

    print("Classification accuracy: {}" .format(np.mean(scores)))

    return np.mean(scores)

# set epoching parameters
tmin, tmax =-1., 4.
day, name, trial, task_num, path, C, gamma, n_components, time = Setting_file(time_t=False).set_file()

if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
        (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]

if task_num == 2:
    event_id = dict(Left=1, Right=2)  # map event IDs to tasks
    target_names = ['left', 'right']
    data40 = np.empty((len(path_b)*40,0))
elif task_num == 3:
    event_id = dict(Left=1, Right=2, Another=3)
    target_names = ['left', 'right', 'Another']
    data40 = np.empty((len(path_b)*60,0))
else:
    print('Error: Please task_num is 2 or 3', file=sys.stderr)
    sys.exit(1)  


#frequency bands
iter_freqs = [
    ('A', 4, 8, 1.0),
    ('B', 8, 12, 1.0),
    ('C',12, 16, 1.0),
    ('D',16, 20, 1.0),
    ('E',20, 24, 1.0),
    ('F',24, 28, 1.0),
    ('G',28, 32, 1.0)
]

#parameters
svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)

# set epoching parameters
tmin, tmax =-1., 4.

acc_map = list()
csp_map = list()
vec_map = list()
sca_map = list()
for band, fmin, fmax, mag in iter_freqs:
    epochs = []
    scaler = preprocessing.StandardScaler()
    vectorizer = Vectorizer()
    csp = CSP(n_components = n_components, reg=None, log=True, 
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

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")
selector.fit(data40, labels)
data40 = selector.transform(data40)
print(data40.shape)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
svm = SVC(C=study.best_trial.params['C'], gamma = study.best_trial.params['gamma'], kernel='rbf', cache_size=100)


print("Classification accuracy: {}" .format(study.best_value))
for key, value in study.best_trial.params.items():
    print('    {}: {}'.format(key, value))

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

preds = np.empty(len(labels))
for train, test in cv.split(data40, labels):
    svm.fit(data40[train], labels[train])
    preds[test] = svm.predict(data40[test])

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

svm.fit(data40, labels)
pickle_map = [csp_map, svm, vec_map, sca_map, iter_freqs, selector]
pickle_make.maker("csp_map.pickle", pickle_map)
