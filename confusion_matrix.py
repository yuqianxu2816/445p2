"""
EECS 445 - Introduction to Machine Learning
Fall 2024  - Project 2
Generate confusion matrix graphs.
"""

import torch
import numpy as np
from dataset import get_train_val_test_loaders
from model.source import Source
from train_common import *
from utils import config
import utils
import os
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def gen_labels(loader, model):
    """Return true and predicted values."""
    y_true, y_pred = [], []
    for X, y in loader:
        with torch.no_grad():
            output = model(X)
            predicted = predictions(output.data)
            y_true = np.append(y_true, y.numpy())
            y_pred = np.append(y_pred, predicted.numpy())
    return y_true, y_pred


def plot_conf(loader, model, sem_labels, png_name):
    """Draw confusion matrix."""
    y_true, y_pred = gen_labels(loader, model)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues, interpolation="nearest")
    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Frequency", rotation=270, labelpad=10)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha="center", va="center")
    plt.gcf().text(0.02, 0.4, sem_labels, fontsize=9)
    plt.subplots_adjust(left=0.3)
    ax.set_xlabel("Predictions")
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("True Labels")
    plt.savefig(png_name)


def main():
    """Create confusion matrix and save to file."""
    tr_loader, va_loader, te_loader, semantic_labels = get_train_val_test_loaders(
        task="source", batch_size=config("source.batch_size")
    )

    model = Source()
    print("Loading source...")
    model, epoch, stats = restore_checkpoint(model, config("source.checkpoint"))

    sem_labels = "0 - Samoyed\n1 - Miniature Poodle\n2 - Saint Bernard\n3 - Great Dane\n4 - Dalmatian\n5 - Chihuahua\n6 - Siberian Husky\n7 - Yorkshire Terrier"

    # Evaluate model
    plot_conf(va_loader, model, sem_labels, "conf_matrix.png")


if __name__ == "__main__":
    main()
