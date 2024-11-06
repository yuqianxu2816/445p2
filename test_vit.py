"""
EECS 445 - Introduction to Machine Learning
Fall 2024  - Project 2
Test ViT
    Test our trained CNN from train_cnn.py on the heldout test data.
    Load the trained CNN model from a saved checkpoint and evaulates using
    accuracy and AUROC metrics.
    Usage: python test_Vit.py
"""

import torch
import numpy as np
import random
from dataset import get_train_val_test_loaders
from model.vit import ViT
from train_common import *
from utils import config
import utils

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    """Print performance metrics for model at specified epoch."""
    # Data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task="target",
        batch_size=config("vit.batch_size"),
    )

    # Model
    model = ViT(num_blocks=2,
                   num_heads=2,
                   num_hidden=16,
                   num_patches=16)

    # define loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Attempts to restore the latest checkpoint if exists
    print("Loading Vit...")
    model, start_epoch, stats = restore_checkpoint(model, config("vit.checkpoint"))

    axes = utils.make_training_plot()

    # Evaluate the model
    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        include_test=True,
        update_plot=False,
    )


if __name__ == "__main__":
    main()
