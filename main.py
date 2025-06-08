import numpy as np
import csv

def load_data(filename):
    data = np.loadtxt(filename)
    y = data[:, 0]
    X = data[:, 1:]
    return X, y

def normalize_features(X):
    stds = np.std(X, axis=0)
    means = np.mean(X, axis=0)
    valid = stds != 0
    X = X[:, valid]
    means = means[valid]
    stds = stds[valid]
    return (X - means) / stds


def leave_one_out_accuracy(X, y, feature_indices):
    if not feature_indices:
        return 0.0
    correct = 0
    for i in range(len(X)):
        test_sample = X[i, feature_indices]
        test_label = y[i]
        train_X = np.delete(X, i, axis=0)[:, feature_indices]
        train_y = np.delete(y, i)
        dists = np.linalg.norm(train_X - test_sample, axis=1)
        nearest = np.argmin(dists)
        if train_y[nearest] == test_label:
            correct += 1
    return correct / len(X) * 100
