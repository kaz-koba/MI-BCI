import mne
from mne import Epochs
import pandas as pd

class Epoch_raw:
    def __init__(self, raw, event, event_id, fmin=2, fmax=45, tmin=-1, tmax=4, picks=None):
        #mustparameter
        self.raw = raw
        self.event = event
        self.event_id = event_id
        #if you use band-pass-filter
        self.fmin = fmin
        self.fmax = fmax
        #epochtime
        self.tmin = tmin
        self.tmax = tmax
        self.picks = picks
    
    def Epochs_raw(raw, event, event_id, fmin = 2, fmax = 45, tmin = -1, tmax = 4, picks = None):
        event = pd.read_csv(event, header=None)
        events = event.values
        raw.filter(fmin, fmax, fir_design='firwin')
        epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, baseline=None, preload=True, 
                        picks = picks, event_repeated='drop')
        del raw

        return epochs