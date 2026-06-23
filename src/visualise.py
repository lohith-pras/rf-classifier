"""Plotting helpers: raw I/Q, confusion matrix, accuracy-vs-SNR curve."""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def plot_iq(sample, show=True):
    """Plot the I and Q time series of a single (2, 128) sample."""
    i_row = sample[0, :]
    q_row = sample[1, :]
    x = range(len(i_row))
    fig, axs = plt.subplots(2)
    plt.suptitle("I and Q components")
    axs[0].plot(x, i_row)
    axs[0].set_xlabel("Sample Index")
    axs[0].set_ylabel("Amplitude")
    axs[1].plot(x, q_row)
    axs[1].set_xlabel("Sample Index")
    axs[1].set_ylabel("Amplitude")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_confusion_matrix(y_test, y_pred, labels, save_path=None, show=True):
    """Row-normalised confusion matrix over the modulation classes."""
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(11, 11))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Confusion Matrix — RF Modulation Classifier")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_accuracy_vs_snr(snr_levels, accuracies, save_path=None, show=True):
    """Test accuracy as a function of SNR (the SNR sweep result)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr_levels, accuracies, marker="o")
    ax.axhline(1 / 11, color="gray", linestyle="--", label="Random guess (1/11)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy vs SNR — RF Modulation Classifier")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig
