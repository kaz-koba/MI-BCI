import pyedflib
#import csv
import numpy as np
import pandas as pd
import sys
import os.path

def main():
    args = sys.argv
    edfpath = os.path.join("EDFfile", args[1] + ".edf")
    np.set_printoptions(threshold=np.inf)
    edf = pyedflib.EdfReader(edfpath)
    annot = pd.DataFrame(edf.read_annotation(),
                        columns=['time', 'Zero', 'Stimulations'])
    path = os.path.join("CSVfile", args[1] + ".csv")
    annot.to_csv(path, index=False)

if __name__ == '__main__':
    main()