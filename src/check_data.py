import numpy as np
import pandas as pd
import warnings
import os

def load_features():
    # Load Data
    train_file = "train_featuresHiLo_thresh_4.5_db4.csv"
    test_file = "test_featuresHiLo_thresh_4.5_db4.csv"
    if not (os.path.isfile('../input/' + train_file) \
            & os.path.isfile('../input/' + test_file)):
        print("No features files..")
        raise AssertionError()

    features = ["entropy", "n5", "n25", "n75", "n95", "median", "mean", "std", "var", "rms", "no_zero_crossings", "no_mean_crossings", "min_height", "max_height", "mean_height", "min_width", "max_width", "mean_width", "num_detect_peak", "num_true_peaks", "low_high_ratio", "hi_true", "lo_true", "low_high_ratio_true"]
    target = ["fault"]

    data_file = "../input/" + train_file
    df_train = pd.read_csv(data_file)
    train = df_train[features + target]

    data_file = "../input/" + test_file
    df_test = pd.read_csv(data_file)
    test = df_test[features]

    return train, test

train, test = load_features()

# print(train.describe()[['num_true_peaks', 'hi_true', 'num_detect_peak', 'lo_true', 'mean', 'std']])

train_feature = train['num_true_peaks'].values

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.distplot(train_feature)

plt.show()


print(train_feature.mean())

