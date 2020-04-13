from mne import Epochs
from mne.io import read_raw_edf
import pandas as pd

class Epoch_raw:
    def __init__(self, path, event, fmin=4, fmax=35, event_id, tmin=-1, tmax=4):
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
    
    def Epochs_raw(path, event, fmin, fmax, event_id, tmin, tmax):
        raw = read_raw_edf(path, stim_channel=False, preload=True)
        event = pd.read_csv(event, header=None)
        events = event.values
        raw.filter(fmin, fmax, n_jobs=1,  
                l_trans_bandwidth=1,  
                h_trans_bandwidth=1)
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True, 
                        event_repeated='drop')
        del raw

        return epochs