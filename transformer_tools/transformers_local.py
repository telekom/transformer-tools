# Copyright (c) 2021 Philip May, Deutsche Telekom AG
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Transformers tools."""
import sklearn
import torch


class LabeledDataset(torch.utils.data.Dataset):
    """Dataset with labes."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class KFoldLabeledDataset:
    """Utility to do k-fold cross-validation on ``LabeledDataset``."""

    def __init__(self, n_splits=7, n_repeats=1, random_state=None):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, labeled_dataset, stratification_labels=None):
        """Generates data splits of training and test set."""
        idxs = list(range(len(labeled_dataset)))
        if stratification_labels is None:  # no stratification wanted
            k_fold = sklearn.model_selection.RepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            k_fold_split = k_fold.split(idxs)
        else:  # stratification wanted
            k_fold = sklearn.model_selection.RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
            )
            k_fold_split = k_fold.split(idxs, stratification_labels)

        for train_idxs, test_idxs in k_fold_split:
            train = torch.utils.data.Subset(labeled_dataset, train_idxs)
            test = torch.utils.data.Subset(labeled_dataset, test_idxs)
            yield train, test
