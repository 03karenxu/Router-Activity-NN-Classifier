# PREPROCESS.PY -----------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.interpolate import UnivariateSpline
from sklearn.model_selection import train_test_split


# main preprocess function
def preprocess(x,s):
    x = min_max_scale(x)
    # x = fit_spline(x,s)
    return x


# plots a random sample
def plot_random_sample(x, y, i=-1):
    if i == -1:
        i = np.random.choice(x.shape[0])
    row = x[i]
    label = y[i]
    plot_sample(row,label)


# plots a specified sample
def plot_sample(x,y):
    plt.plot(x)
    plt.xlabel("Time Step")
    plt.ylabel("Router Activity")
    plt.title(f"Label: {y}")
    plt.show()
    plt.close()


# plots all samples overlayed
def plot_all(label, x, y):
    indices_label = np.where(y == label)[0]
    plt.figure(figsize=(10, 6))
    for i in indices_label:
        plt.plot(x[i], alpha=0.1, color="blue")
    plt.ylabel("Activity")
    plt.title(f"All Samples with Label {label}")
    plt.show()
    plt.close()


# loads data from csv and splits into train/test
def load_data(file_path,test_ratio):
    df = pd.read_csv(file_path, header=None)
    x,y = split_features(df)
    return train_test_split(
        x, y, test_size=test_ratio, stratify=y
    )


# splits features from labels
def split_features(df):
    x = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return x,y


# scales sample-wise to values between 0 and 1
def min_max_scale(x):
    # x_min = x.min()
    # x_max = x.max()
    x_min = x.min(axis=1,keepdims=True)
    x_max = x.max(axis=1,keepdims=True)
    return (x - x_min) / (x_max - x_min + 1e-9)


# applies smote to minority class
def apply_smote(x, y):
    smote = SMOTE(sampling_strategy="minority")
    x_res, y_res = smote.fit_resample(x, y)
    return x_res, y_res


# fits a univariate spline to all samples
def fit_spline(x,s):
    x_smoothed = []
    time_steps = np.arange(x.shape[1])
    for sample in x:
        spl = UnivariateSpline(time_steps,sample,s=s) 
        smoothed_sample = spl(time_steps) 
        x_smoothed.append(smoothed_sample) 
    return np.array(x_smoothed)


# prints processed data to csv
def save_processed_data(x, y, out_file):
    processed_df = pd.DataFrame(x)
    processed_df["label"] = y.astype(int)
    processed_df.to_csv(out_file, index=False, header=False)


# ------------------------------------------------------