import pandas as pd
import numpy as np


def simple_split(
        train_data: pd.DataFrame,
        target_data: pd.DataFrame,
        fraction: float
    ) -> tuple[tuple[pd.DataFrame, pd.DataFrame],
               tuple[pd.DataFrame, pd.DataFrame]]:
    if len(train_data) != len(target_data):
        raise ValueError('Length of train is not equal to length of target')
    train_size = int(len(train_data) * fraction)
    X_train, X_test = train_data[:train_size].copy().reset_index(drop=True), train_data[train_size:].copy().reset_index(drop=True)
    Y_train, Y_test = target_data[:train_size].copy().reset_index(drop=True), target_data[train_size:].copy().reset_index(drop=True)
    return (X_train, Y_train), (X_test, Y_test)


def calculate_class_weights(Y_train):
    n_samples = len(Y_train)
    n_classes = len(np.unique(Y_train))
    class_counts = np.bincount(Y_train['1_sec'])
    class_weights = n_samples / (n_classes * class_counts)
    class_weights_dict = dict(zip(np.unique(Y_train), class_weights))
    return class_weights_dict
