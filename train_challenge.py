"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

from dataset_challenge import get_train_val_test_loaders
from model.challenge import Challenge
from train_common import *
from utils import config

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def main():
    transform = A.Compose([
        # A.HorizontalFlip(p=0.1),
        # A.VerticalFlip(p=0.1),
        # A.RandomRotate90(p=0.2),
        # A.RandomResizedCrop(size=(64, 64), p=0.5),
        # A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
        A.Normalize(mean=[0.48614902, 0.46230196, 0.3684], std=[0.24718824, 0.23398824, 0.24293725]),
        ToTensorV2(),
    ])
    # transform = None
    # Data loaders
    if check_for_augmented_data("./data"):
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
            transform=transform,
            augment=True
        )
    else:
        tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
            task="target",
            batch_size=config("challenge.batch_size"),
            transform=transform,
        )
    # Model
    model = Challenge(dropout_prob=0.6)

    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    #

    # Attempts to restore the latest checkpoint if exists
    print("Loading challenge...")
    model, start_epoch, stats = restore_checkpoint(model, config("challenge.checkpoint"))

    axes = utils.make_training_plot()

    # Evaluate the randomly initialized model
    evaluate_epoch(
        axes, tr_loader, va_loader, te_loader, model, criterion, start_epoch, stats, include_test=True
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: define patience for early stopping
    patience = 20
    curr_patience = 0
    #

    # Loop over the entire dataset multiple times
    epoch = start_epoch
    while curr_patience < patience:
        # Train model
        train_epoch(tr_loader, model, criterion, optimizer)

        # Evaluate model
        evaluate_epoch(
            axes, tr_loader, va_loader, te_loader, model, criterion, epoch + 1, stats, include_test=True
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, config("challenge.checkpoint"), stats)

        # Updates early stopping parameters
        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        print(prev_val_loss, curr_patience)
        #
        epoch += 1
    print("Finished Training")
    # Save figure and keep plot open
    utils.save_challenge_training_plot()
    utils.hold_training_plot()


if __name__ == "__main__":
    main()
