from pylsl import StreamInlet, resolve_streams
from pythonosc import udp_client
from collections import deque
from mne import filter

import numpy as np
import threading
import configparser
import argparse

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import pickle
with open('csp_map.pickle', mode='rb') as fp:
    pickle_map = pickle.load(fp)
csp_map = pickle_map[0]
svm = pickle_map[1]
vec_map = pickle_map[2]
sca_map = pickle_map[3]
iter_freqs = pickle_map[4]

inifile = configparser.ConfigParser()
inifile.read('./parameter.ini', 'UTF-8')
task_num = int(inifile.get('setting', 'task_num'))

def inlet_specific_stream(stream_name):
    streams = resolve_streams(wait_time=3.)
    stream_names = []
    for stream in streams:
        inlet = StreamInlet(stream)
        stream_names.append(inlet.info().name())
    idx = np.where(np.array(stream_names)==stream_name)[0][0]
    inlet = StreamInlet(streams[idx])
    return inlet

def fix_labels(i, task_num):
    id = 1
    if i % task_num == 0:
        return task_num 
    while True:
        if i % task_num == id:
            return id
        id += 1

def signal_print():
    count = [0]*3
    Truecount = [0]*3
    while True:
        d, _ = inlet1.pull_chunk(timeout=1. ,max_samples=512)
        data = np.empty((1,0))
        i=0
        d = np.array(d).T
        for band, fmin, fmax, mag in iter_freqs:
            x = d
            csp = csp_map[i]
            vectorizer = vec_map[i]
            scaler = sca_map[i]
            print(x.shape)
            x = filter.filter_data(x, sfreq=513, l_freq=fmin, h_freq=fmax, fir_design='firwin')
            x = csp.transform(x[np.newaxis,:,:])
            print(x)
            x = vectorizer.transform(x)
            x = scaler.transform(x)
            x *= mag
            data = np.hstack((data, x))
            i += 1

        output = svm.predict(data)
        output = fix_labels(output[0], task_num)
        print(output)
        client.send_message("/output", output)
            

if __name__ == "__main__":
    #LSLsetting
    stream_signal = 'openvibeSignal'
    stream_marker = 'openvibeMarkers'
    inlet2 = inlet_specific_stream(stream_marker)
    inlet1 = inlet_specific_stream(stream_signal)
    thread_1 = threading.Thread(target=signal_print)

    #OSCsetting
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1",
    help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=8000,
    help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)

    thread_1.start()