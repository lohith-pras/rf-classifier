"""End-to-end pipeline: load data -> features -> train -> evaluate -> plots.

Run from the repo root:

    uv run python src/pipeline.py

Outputs the confusion matrix and accuracy-vs-SNR curve to notebooks/.
"""

from pathlib import Path

from sklearn.metrics import classification_report

from dataset import build_feature_matrix
from model import accuracy_vs_snr, split, train_classifier
from visualise import plot_accuracy_vs_snr, plot_confusion_matrix

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building feature matrix (cached after first run)...")
    X, y, snr = build_feature_matrix()
    print(f"  X={X.shape}  y={y.shape}  classes={sorted(set(y))}")

    X_train, X_test, y_train, y_test, _, snr_test = split(X, y, snr)

    print("Training Random Forest...")
    model = train_classifier(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = model.score(X_test, y_test)
    print(f"\nOverall test accuracy: {acc:.4f}  (random guess = {1/11:.4f})\n")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(
        y_test, y_pred, labels=model.classes_,
        save_path=OUT_DIR / "confusion_matrix.png", show=False,
    )

    snr_levels, accuracies = accuracy_vs_snr(model, X_test, y_test, snr_test)
    print("\nAccuracy vs SNR:")
    for level, a in zip(snr_levels, accuracies):
        print(f"  {level:>4} dB : {a:.3f}")

    plot_accuracy_vs_snr(
        snr_levels, accuracies,
        save_path=OUT_DIR / "accuracy_vs_snr.png", show=False,
    )
    print(f"\nPlots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
