import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str] | None = None,
    figsize: tuple[int, int] = (6, 6),
    text_size: int = 15,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm_norm, cmap="Blues")
    fig.colorbar(cax, format=PercentFormatter(xmax=1))

    labels = np.arange(cm.shape[0]) if classes is None else classes

    ax.set(
        title=title,
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=labels,
        yticklabels=labels,
    )

    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    threshold = (cm_norm.max() + cm_norm.min()) / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            x=j,
            y=i,
            s=f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm_norm[i, j] > threshold else "black",
            size=text_size,
        )
