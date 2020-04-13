import pickle

def maker(name, data):
    with open(name, 'wb') as f:
        pickle.dump(data, f)