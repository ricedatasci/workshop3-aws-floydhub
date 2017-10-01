import os
from collections import defaultdict
import pandas as pd
import numpy as np

DATA_PATH = "../data/"


def load_features():
    feature_names = []
    with open(os.path.join(DATA_PATH, "features.txt"), 'r') as features:
        for line in features:
            str_ind, feature = line.rstrip('\n').split(' ')
            feature_names.append(feature)
            assert feature_names[int(str_ind) - 1] == feature
    return feature_names


FEATURES = load_features()


def load_train_measurements():
    return pd.read_csv(os.path.join(DATA_PATH, "training_data.csv"), names=FEATURES)


def load_train_labels():
    return pd.read_csv(os.path.join(DATA_PATH, "training_labels.csv"), names=["labels"]) - 1


def load_train():
    return load_train_measurements().values, load_train_labels().values


def load_train_subjects():
    subjects = pd.read_csv(os.path.join(
        DATA_PATH, "training_subjects.csv"), names=["subject"])
    return subjects.groupby(["subject"])


def load_train_val(split=0.2, shuffle=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Load the data and subject info
    x, y = load_train()
    subjects = load_train_subjects()
    # create an index on the subjects
    if shuffle:
        inds = np.random.permutation(len(subjects.groups))
    else:
        inds = np.arange(len(subjects.groups))
    # Split the data
    split_ind = int(-split * len(inds))
    train_subject_samples = np.concatenate(
        [subjects.groups[i] for i in inds[:-split_ind]])
    val_subject_samples = np.concatenate(
        [subjects.groups[i] for i in inds[-split_ind:]])
    return (x[train_subject_samples], y[train_subject_samples]), (x[val_subject_samples], y[val_subject_samples])


def load_test_measurements():
    return pd.read_csv(os.path.join(DATA_PATH, "test_data.csv"), names=FEATURES)


def load_test():
    return load_test_measurements().values


def create_submission(prob_preds, submission_fname="submission.csv"):
    # Add 1 to move from 0..5, to 1..6
    preds = np.argmax(prob_preds, axis=1) + 1
    # Concatenate the IDs
    data = np.stack([np.arange(1, len(preds) + 1), preds]).T
    # Get the submission filepath
    fp = os.path.join("../submissions", submission_fname)
    pd.DataFrame(data=data, columns=['Id', 'Prediction']).to_csv(
        fp, index=False)
