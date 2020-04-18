import mne
from mne import Epochs
from mne.io import read_raw_edf
import pandas as pd

class Epoch_raw:
    def __init__(self, path, event, event_id, fmin=2, fmax=45, tmin=-1, tmax=4,
                channel_names = ["FCz", "FC1", "FC2", "FC3", "FC4", "Cz", "C1", "C2", "C3", "C4"],
                exclude_names = None):
        #mustparameter
        self.path = path
        self.event = event
        self.event_id = event_id
        #if you use band-pass-filter
        self.fmin = fmin
        self.fmax = fmax
        #epochtime
        self.tmin = tmin
        self.tmax = tmax
        self.channel_names = channel_names
        self.exclude_names = exclude_names
    
    def Epochs_raw(path, event, event_id, fmin = 2, fmax = 45, tmin = -1, tmax = 4, 
                    channel_names = ["FCz", "FC1", "FC2", "FC3", "FC4", "Cz", "C1", "C2", "C3", "C4"]):
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        event = pd.read_csv(event, header=None)
        events = event.values
        picks = mne.pick_types(raw.info["ch_names"], eeg=True, stim=True, include=channel_names, exclude=None)
        raw.filter(fmin, fmax, n_jobs=1,  
                l_trans_bandwidth=1,  
                h_trans_bandwidth=1)
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True, 
                        picks = picks, event_repeated='drop')
        del raw

        return epochs