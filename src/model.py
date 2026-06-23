"""Random Forest modulation classifier: training and the SNR sweep."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def split(X, y, snr, test_size=0.2, random_state=42):
    """Stratified train/test split that keeps the per-sample SNR aligned."""
    return train_test_split(
        X, y, snr, test_size=test_size, random_state=random_state, stratify=y
    )


def train_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """Fit a Random Forest on the training fold."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def accuracy_vs_snr(model, X_test, y_test, snr_test):
    """Compute test accuracy at each SNR level.

    Returns (snr_levels, accuracies) as sorted parallel arrays. This is
    the headline experiment: classical features carry almost no signal at
    very low SNR and accuracy should climb toward an asymptote as SNR rises.
    """
    snr_levels = np.unique(snr_test)
    accuracies = []
    for level in snr_levels:
        mask = snr_test == level
        accuracies.append(model.score(X_test[mask], y_test[mask]))
    return snr_levels, np.array(accuracies)
