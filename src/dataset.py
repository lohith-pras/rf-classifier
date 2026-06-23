"""Load RML2016.10a and turn it into a cached feature matrix.

The raw dataset is a dict keyed by (modulation, snr) -> array of shape
(n_samples, 2, 128). Feature extraction over all 220k samples takes
~15 min, so the result is cached to a .npz and reused on later runs.
"""

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

from features import extract_features

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "RML2016.10a_dict.dat"
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "features.npz"


def load_raw(data_path=DEFAULT_DATA_PATH):
    """Load the raw pickled RML2016.10a dict."""
    with open(data_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_feature_matrix(data_path=DEFAULT_DATA_PATH, cache_path=DEFAULT_CACHE_PATH, use_cache=True):
    """Extract features for every sample.

    Returns (X, y, snr):
        X   : (N, 12) float32 feature matrix
        y   : (N,) modulation labels (str)
        snr : (N,) signal-to-noise ratio in dB (int)

    The per-sample SNR is preserved so we can evaluate accuracy as a
    function of SNR later. Results are cached to ``cache_path``.
    """
    cache_path = Path(cache_path)
    if use_cache and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        return cached["X"], cached["y"], cached["snr"]

    data = load_raw(data_path)

    features, labels, snrs = [], [], []
    for (modulation, snr), samples in tqdm(data.items(), desc="Extracting features"):
        for sample in samples:
            features.append(extract_features(sample))
            labels.append(modulation)
            snrs.append(snr)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    snr = np.asarray(snrs, dtype=np.int16)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, snr=snr)
    return X, y, snr
