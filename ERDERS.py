import configparser

import numpy as np
import matplotlib.pyplot as plt

import mne
from mne import concatenate_epochs

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from pathfile import PATHfile
from epoch_raw import Epoch_raw

if __name__ == "__main__":
    # set epoching parameters
    tmin, tmax =-2., 5.
    fmin, fmax = 2., 40.
    event_map = {0:"Left", 1:'Right', 2:'Another'}
    sample_freq = 512
    reg_sec = 1
    inifile = configparser.ConfigParser()
    inifile.read('./parameter.ini', 'UTF-8')
    exec_time = datetime.now().strftime('%Y%m%d_%H:%M:%S')
    log_file = "file.log"

    day = inifile.get('setting', 'day')
    name = inifile.get('setting', 'name')
    trial = inifile.get('setting', 'trial')
    task_num = inifile.get('setting', 'task_num')
    path = inifile.get('setting', 'path')
    ch_list = ["Cz", "C3", "C4"]

    if task_num == "2":
        event_id = [1, 2]
    elif task_num == "3":
        event_id = [1, 2, 3]

    if path == "day":
        path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
            (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
            (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
    elif path == "trial":
        path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]

    epochs = []
    # (re)load the data to save memory
    for path, event in path_b:
        epochs.append(Epoch_raw.Epochs_raw(path, event, event_id, tmin=tmin, tmax=tmax))
    epochs = concatenate_epochs(epochs)
    ch_names = epochs.ch_names
    print(ch_names)
    labels = epochs.events[:, -1]

    index = [np.where(labels == id)[0] for id in event_id]
    data = [epochs[i] for i in index]
    data = [X.get_data() for X in data]  #(trial, ch, t)

    data_jth_ave = [] #(ch, t)
    for X in data:
        data_jth_ave.append(np.array([X[:,:,j].mean(axis=0) for j in range(0, X.shape[2])]).T)

    #compute each averages of all trials
    average = [np.zeros(X[0].shape) for X in data]

    for i in range(len(average)):
        for x in data[i]:
            average[i] += (x - data_jth_ave[i]) ** 2
        average[i] /= len(data[i])

    # compute reference power
    ref = [np.atleast_2d(ave[:, :reg_sec * sample_freq].mean(axis=1)).T for ave in average]

    #compute ERD
    erd = [] #(ch, t)
    for i in range(len(average)):
        erd.append((average[i] - ref[i]) / ref[i] * 100)

    time = np.arange(len(average[0][0])) / sample_freq
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    color_num = 0
    cmap = plt.get_cmap("tab10")

    for i in range(len(erd)):
        for x, ch in zip(erd[i], ch_names):
            if ch in ch_list:
                ax[i].plot(time, x, label=event_map[i] + ch, color=cmap(color_num))
                color_num += 1

    for sub in ax:
        sub.set_xlim(0, 7)
        sub.set_ylim(-100, 150)
        sub.set_xlabel("time [sec]")
        sub.set_ylabel("ERD [%]")
        sub.vlines(3, -200, 200, colors='red', label="Cue")
        sub.hlines(0, 0, 7, colors='gray', linestyles='dotted')
        sub.legend()

    ax[0].set_title("Left hand trials")
    ax[1].set_title("Right hand trials")
    ax[2].set_title("Another trials")

    plt.suptitle(f"{name}'s ERD ({fmin}-{fmax} Hz)")
    plt.savefig(f"figure/_[]_{fmin}-{fmax}.png")

    plt.show()