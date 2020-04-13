from pylsl import StreamInlet, resolve_streams
from collections import deque
from mne import filter

import numpy as np
import threading

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import pickle
with open('csp_map.pickle', mode='rb') as fp:
    csp_map = pickle.load(fp)
with open('svm.pickle', mode='rb') as fp:
    svm = pickle.load(fp)
with open('vec.pickle', mode='rb') as fp:
    vec_map = pickle.load(fp)
with open('sca.pickle', mode='rb') as fp:
    sca_map = pickle.load(fp)
with open('freqs.pickle', mode='rb') as fp:
    iter_freqs = pickle.load(fp)

stim = 0

def inlet_specific_stream(stream_name):
    streams = resolve_streams(wait_time=3.)
    stream_names = []
    for stream in streams:
        inlet = StreamInlet(stream)
        stream_names.append(inlet.info().name())
    idx = np.where(np.array(stream_names)==stream_name)[0][0]
    inlet = StreamInlet(streams[idx])
    return inlet 

def signal_print():
    global stim
    count = 0
    Truecount = 0
    while True:
        d, _ = inlet1.pull_chunk(timeout=1.05 ,max_samples=513)
        data40 = np.empty((1,0))
        i=0
        d = np.array(d).T
        for band, fmin, fmax, mag in iter_freqs:
            x = d
            csp = csp_map[i]
            vectorizer = vec_map[i]
            scaler = sca_map[i]
            print(x.shape)
            x = filter.filter_data(x, l_freq=fmin, h_freq=fmax, n_jobs=1, sfreq=512,
                                    l_trans_bandwidth=1, h_trans_bandwidth=1)
            x = csp.transform(x[np.newaxis,:,:])
            x = vectorizer.transform(x)
            x = scaler.transform(x)
            x *= mag
            data40 = np.hstack((data40, x))
            i += 1

        output = svm.predict(data40)
        print(output)
        #print(output)
        if stim==1 or stim==2:
            count += 1
        if output[0] == stim:
            Truecount += 1
        if count != 0:
            print("acc: {}%" .format(Truecount/count*100))
        
def check_stim():
    global stim
    while True:
        print(stim)
        d, _ = inlet2.pull_sample()
        print(d)
        if d[0] == 769:
            stim = 1
        elif d[0] == 770:
            stim = 2
        elif d[0] == 800:
            stim = 0

if __name__ == "__main__":
    stream_signal = 'openvibeSignal'
    stream_marker = 'openvibeMarkers'
    inlet2 = inlet_specific_stream(stream_marker)
    inlet1 = inlet_specific_stream(stream_signal)
    thread_1 = threading.Thread(target=signal_print)
    thread_2 = threading.Thread(target=check_stim)
    thread_1.start()
    thread_2.start()