import mne
from mne import Epochs
import pandas as pd
import configparser

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

class Setting_file:
    def __init__(self, day_t=True, name_t=True, trial_t=True, task_num_t=True, path_t=True, C_t=True, gamma_t=True, n_components_t=True, time_t=True):
        self.day_t = day_t
        self.name_t = name_t
        self.trial_t = trial_t
        self.task_num_t = task_num_t
        self.path_t = path_t
        self.C_t = C_t
        self.gamma_t = gamma_t
        self.n_components_t = n_components_t
        self.time_t = time_t
    
    def set_file(self):
        inifile = configparser.ConfigParser()
        inifile.read('./parameter.ini', 'UTF-8')
        day = inifile.get('setting', 'day') if self.day_t else None
        name = inifile.get('setting', 'name') if self.name_t else None
        trial = inifile.get('setting', 'trial') if self.trial_t else None
        task_num = int(inifile.get('setting', 'task_num')) if self.task_num_t else None
        path = inifile.get('setting', 'path') if self.path_t else None
        C = float(inifile.get('setting', 'C')) if self.C_t else None
        gamma = float(inifile.get('setting', 'gamma')) if self.gamma_t else None
        n_components = int(inifile.get('setting', 'n_components')) if self.n_components_t else None
        time = float(inifile.get('setting', 'time')) if self.time_t else None

        return day, name, trial, task_num, path, C, gamma, n_components, time
