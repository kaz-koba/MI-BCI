import configparser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

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
from epoch_raw import Epoch_raw
import pickle_make

def objective(trial):
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    gamma = trial.suggest_loguniform('gamma', 1e-4, 1e4)

    svm = SVC(C=C, gamma = gamma, kernel='rbf', cache_size=100)

    scores = cross_val_score(svm, data40, l_labels, cv=cv, n_jobs=1)

    print("Classification accuracy: {}" .format(np.mean(scores)))

    return np.mean(scores)

def fix_labels(i, task_num):
    id = task_num
    while True:
        if i % task_num == 0:
            return id
        id -= 1
        if id == 1:
            return 1

# set epoching parameters
tmin, tmax =-1., 4.
inifile = configparser.ConfigParser()
inifile.read('./parameter.ini', 'UTF-8')

day = inifile.get('setting', 'day')
name = inifile.get('setting', 'name')
trial = inifile.get('setting', 'trial')
task_num = inifile.get('setting', 'task_num')
path = inifile.get('setting', 'path')

iter_freqs = [
    ('Theta', 4, 7, 1.0),
    ('Alpha', 8, 12, 1.0),
    ('Beta', 13, 25, 1.0),
    ('Gamma', 30, 45, 1.0)
]

if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2"))]
        #(PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]

time_map = [
    (0., 1., task_num*0),
    (0.5, 1.5, task_num*0),
    (1., 2., task_num*1),
    (1.5, 2.5, task_num*1),
    (2., 3., task_num*2),
    (2.5, 3.5, task_num*2)
    ]

if task_num == "2":
    event_id = dict(Left=1, Right=2) # map event IDs to tasks
    target_names = ['left','right']
    data40 = np.empty((len(path_b)*40*len(time_map),0))

elif task_num == "3":
    event_id = dict(Left=1, Right=2, Another=3)
    target_names = ['left', 'right', 'Another']
    data40 = np.empty((len(path_b)*60*len(time_map),0))


#parameters
svm = SVC(C=float(inifile.get('setting', 'C')), gamma = float(inifile.get('setting', 'gamma')), kernel='rbf', cache_size=100)
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
    csp = CSP(n_components = int(inifile.get('setting', 'n_components')), reg=None, log=True, norm_trace=False, transform_into='average_power')
    # (re)load the data to save memory
    for path, event in path_b:
        epochs.append(Epoch_raw.Epochs_raw(path, event, event_id, fmin, fmax, tmin, tmax))
    epochs = concatenate_epochs(epochs)
    # remove evoked response
    #epochs.subtract_evoked()
    large_x = np.empty((0, 10, 513))
    l_labels = np.empty((0))
    for lmin, mmin, time_id in time_map:
        labels = epochs.events[:, -1] + time_id
        epochs_train = epochs.copy().crop(tmin=lmin, tmax=mmin)
        epochs_data_train = epochs_train.get_data()
        print(epochs_data_train.shape)
        large_x = np.vstack((large_x, epochs_data_train))
        l_labels = np.concatenate([l_labels, labels], 0)
    large_x = csp.fit_transform(large_x, l_labels)
    csp_map.append(csp)
    large_x = vectorizer.fit_transform(large_x, l_labels)
    vec_map.append(vectorizer)
    large_x = scaler.fit_transform(large_x, l_labels)
    sca_map.append(scaler)
    large_x *= mag
    data40 = np.hstack((data40, large_x))
    scores = cross_val_score(svm, large_x, l_labels, cv=cv, n_jobs=1)
    acc_map.append((band, fmin, fmax, scores))
    del epochs

for freq_name, fmin, fmax, acc in acc_map:
    print("{}({}~{}) Classification accuracy: {}" .format(freq_name, fmin, fmax, np.mean(acc)))

print(data40)
#svm.fit(data40, labels)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
svm = SVC(C=study.best_trial.params['C'], gamma = study.best_trial.params['gamma'], kernel='rbf', cache_size=100)


print("Classification accuracy: {}" .format(study.best_value))
for key, value in study.best_trial.params.items():
    print('    {}: {}'.format(key, value))

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

preds = np.empty(len(l_labels))
for train, test in cv.split(data40, l_labels):
    svm.fit(data40[train], l_labels[train])
    preds[test] = svm.predict(data40[test])

print(svm.predict(data40))
print(preds)
preds = [fix_labels(i, task_num) for i in preds]
l_labels = [fix_labels(i, task_num) for i in l_labels]


# Classification report
report = classification_report(l_labels, preds, target_names=target_names)
print(report)

cm = confusion_matrix(preds, l_labels)
cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
cm_normalized = pd.DataFrame(data=cm_normalized, index=target_names, columns=target_names)
sns.heatmap(cm_normalized, annot=True, cmap='Blues', square=True)
if path == "day":
    trial_name = day
else:
    trial_name = trial
plt.savefig('figure/confusion_matrix_multi_{}_{}_{}.png' .format(name, day, trial_name))
plt.show()

svm.fit(data40, l_labels)
pickle_map = [csp_map, svm, vec_map, sca_map, iter_freqs]
pickle_make.maker("csp_map.pickle", pickle_map)