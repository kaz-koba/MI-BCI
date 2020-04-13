import pandas as pd
import numpy as np
import sys
import os.path

def main():
    args = sys.argv
    path = os.path.join("CSVfile", args[1] + ".csv")
    df = pd.read_csv(path)
    df.time = round((df.time / 10000000) / 0.001953125)
    df.Zero = 0
    df =  df[(df['Stimulations'] == "b'OVTK_GDF_Left'" ) | (df['Stimulations'] == "b'OVTK_GDF_Right'") | (df['Stimulations'] == "b'OVTK_GDF_Down'")]
    df.loc[df['Stimulations'] == "b'OVTK_GDF_Left'", 'Stimulations'] = 1
    df.loc[df['Stimulations'] == "b'OVTK_GDF_Right'", 'Stimulations'] = 2
    df.loc[df['Stimulations'] == "b'OVTK_GDF_Down'", 'Stimulations'] = 3

    df.time = df['time'].astype(np.int64)
    eventpath = os.path.join("eventdata","event-" + args[1] + ".csv")
    df.to_csv(eventpath, index=False, header=False)

if __name__ == '__main__':
    main()
    

