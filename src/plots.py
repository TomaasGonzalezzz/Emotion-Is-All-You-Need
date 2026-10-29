import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=None, ax=None, title=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure
    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    if title:
        ax.set_title(title)
    return fig, ax
