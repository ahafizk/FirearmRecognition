import numpy as np
import pandas as pd
import os
from sigtool import *
from os import listdir
from os.path import isfile, join
from feature_extractor import *
from filter import *
from scipy import signal

def process_data(filtered=True):

    sampling_frequency = 64 # 64 Hz
    overlapping_window = 32 #50% overlap

    cutoff_freq = 30 # not more than 50% of your sampling frquency
    #index of the x, y and z axis data, if the first column index is zero
    x_index = 0
    y_index = 1
    z_index = 2


    data_file = 'data/Clean 20 shots slow phone in pocket.csv'
    save_dir = 'feature/'
    save_file = 'clean_features.csv'

    df = pd.read_csv(data_file, delimiter=',', header=None)

    #now getting the accelerometer sensor data
    # df = df.ix[:, [0, 1, 2]]

    x = df[x_index]
    y = df[y_index]
    z = df[z_index]
    x = pd.to_numeric( np.array(x[1:]))
    y = pd.to_numeric( np.array(y[1:]))
    z = pd.to_numeric( np.array(z[1:]))
    print y
    print z

    #now creating frames from the x, y and z
    if filtered:
        x_frames = generate_frames_1D(filter_data(x, sampling_frequency, cutoff_freq), sampling_frequency, overlapping_window)
        y_frames = generate_frames_1D(filter_data(y, sampling_frequency, cutoff_freq), sampling_frequency, overlapping_window)
        z_frames = generate_frames_1D(filter_data(z, sampling_frequency, cutoff_freq), sampling_frequency, overlapping_window)
    else:
        # do processing without filter the raw data.
        x_frames = generate_frames_1D(x, sampling_frequency,
                                      overlapping_window)
        y_frames = generate_frames_1D(y,sampling_frequency,
                                      overlapping_window)
        z_frames = generate_frames_1D(z, sampling_frequency,
                                      overlapping_window)
    #assuming all x, y, z have same length
    row, col = x_frames.shape
    features = []
    for i in range(row):
        fet = get_features(x_frames[i], y_frames[i], z_frames[i])
        features.append(fet)

    df_fet = pd.DataFrame(features)  # create dataframes

    d = pd.Series(['gunshot'] * row) #labels or class -- it can be numeric also pd.Series([1] * row)

    df_fet[col] = d
    print df_fet.shape

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    savefile = join(save_dir, save_file)
    df_fet.to_csv(savefile, header=False, index=False)


if __name__=='__main__':

    process_data(False)