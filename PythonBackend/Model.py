# Imports
import os
import glob
import fnmatch
import pandas as pd
import numpy as np
import librosa #To deal with Audio files
import math
import tensorflow as tf
import scipy.io as sio
import run_audio_model
import torch

from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
path = 'ekg_model2.h5'
# Load the model
model = load_model(path)


encoding = {0: 'normal', 1: 'abnormal'}

def preprocessing(file):
    num_rows = 2160
    n = 20
    m = 108
    t = 1
    c = 1
    ecg_data = np.loadtxt(file, delimiter=',', dtype=np.float32)
    ecg_data = ecg_data.reshape(1, t, n, m, c)
    return ecg_data

def get_prediction(file, model_choice="EKG"):
    if (model_choice == "EKG"):
        x_test = preprocessing(file)
        y_pred = model.predict(x_test, batch_size=32)
        return encoding[round(y_pred[0][0])]
    else:
        return run_audio_model.run_audio_model(file, 0.01)


