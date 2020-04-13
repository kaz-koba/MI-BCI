import configparser

import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import concatenate_epochs
from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap

from pathfile import PATHfile
from epoch_raw import Epoch_raw

# load and preprocess data ####################################################
inifile = configparser.ConfigParser()
inifile.read('./parameter.ini', 'UTF-8')

day = inifile.get('setting', 'day')
name = inifile.get('setting', 'name')
trial = inifile.get('setting', 'trial')
path = inifile.get('setting', 'path')
task_num = inifile.get('setting', 'task_num')

if path == "day":
    path_b = [(PATHfile.edfpath(name, day, "1"), PATHfile.eventpath(name, day, "1")),
        (PATHfile.edfpath(name, day, "2"), PATHfile.eventpath(name, day, "2")),
        (PATHfile.edfpath(name, day, "3"), PATHfile.eventpath(name, day, "3"))]
elif path == "trial":
    path_b = [(PATHfile.edfpath(name, day, trial), PATHfile.eventpath(name, day, trial))]
    

tmin, tmax = -2, 4  # define epochs around events (in s)

if task_num == "2":
    event_id = dict(Left=1, Right=2)  # map event IDs to tasks
elif task_num == "3":
    event_id = dict(Left=1, Right=2, Another=3)

epochs = []
for path, event in path_b:
    epochs.append(Epoch_raw.Epochs_raw(path, event, event_id = event_id, tmin=tmin, tmax=tmax, 
                    channel_names=["Cz", "C3", "C4"]))    

epochs = concatenate_epochs(epochs)

# compute ERDS maps ###########################################################
freqs = np.arange(5, 35, 1)  # frequencies from 2-35Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -0.5, 1.0  # set min and max ERDS values in plot
baseline = [-2, -1]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
                buffer_size=None)  # for cluster test

# Run TF decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                    use_fft=True, return_itc=False, average=False,
                    decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
for event in event_id:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                            gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                            axes=ax, colorbar=False, show=False, mask=mask,
                            mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()
    if path == "day":
        trial_name = day
    else:
        trial_name = trial
    fig.savefig('figure/ERDsmap({}-{}-{}-{}).png' .format(name, day, trial_name, event))