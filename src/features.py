"""Hand-crafted feature extraction for I/Q radio samples.

Each RML2016.10a sample is a (2, 128) array: row 0 = in-phase (I),
row 1 = quadrature (Q). We summarise each 128-sample window with 12
statistical features drawn from the amplitude, phase, instantaneous
frequency, and frequency-domain (FFT power) representations.
"""

import numpy as np
import scipy.stats as stats

FEATURE_NAMES = [
    "amp_mean",
    "amp_std",
    "amp_skew",
    "amp_kurt",
    "phase_mean",
    "phase_std",
    "inst_freq_mean",
    "inst_freq_std",
    "fft_power_mean",
    "fft_power_std",
    "fft_power_skew",
    "fft_power_kurt",
]


def extract_features(sample):
    """Return a length-12 float32 feature vector for one (2, 128) I/Q sample."""
    i_row = sample[0, :]
    q_row = sample[1, :]

    # Amplitude (instantaneous magnitude) statistics.
    amplitude = np.sqrt(i_row**2 + q_row**2)
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_skew = stats.skew(amplitude)
    amp_kurt = stats.kurtosis(amplitude)

    # Phase statistics on the unwrapped instantaneous phase.
    unwrapped_phase = np.unwrap(np.arctan2(q_row, i_row))
    phase_mean = np.mean(unwrapped_phase)
    phase_std = np.std(unwrapped_phase)

    # Instantaneous frequency = derivative of unwrapped phase.
    inst_freq = np.diff(unwrapped_phase)
    inst_freq_mean = np.mean(inst_freq)
    inst_freq_std = np.std(inst_freq)

    # Frequency-domain power spectrum statistics.
    complex_signal = i_row + 1j * q_row
    fft_power = np.abs(np.fft.fft(complex_signal)) ** 2
    fft_mean = np.mean(fft_power)
    fft_std = np.std(fft_power)
    fft_skew = stats.skew(fft_power)
    fft_kurt = stats.kurtosis(fft_power)

    return np.array(
        [
            amp_mean,
            amp_std,
            amp_skew,
            amp_kurt,
            phase_mean,
            phase_std,
            inst_freq_mean,
            inst_freq_std,
            fft_mean,
            fft_std,
            fft_skew,
            fft_kurt,
        ],
        dtype=np.float32,
    )
