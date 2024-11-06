"""
EECS 445 - Introduction to Machine Learning
Fall 2024 - Project 2
Train Challenge
    Train a convolutional neural network to classify the heldout images
    Periodically output training information, and saves model checkpoints
    Usage: python train_challenge.py
"""
import random

from dataset_challenge import get_train_val_test_loaders
from model.challenge import Challenge
from model.vit import ViT
from train_common import *
from utils import config
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def train(tr_loader, va_loader, te_loader, model, model_name, task, multiclass=False):
    """Train transfer learning model."""
    # TODO: define loss function, and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)
    #

    model, start_epoch, stats = restore_checkpoint(model, model_name)

    axes = utils.make_training_plot("Challenge Training")

    evaluate_epoch(
        axes,
        tr_loader,
        va_loader,
        te_loader,
        model,
        criterion,
        start_epoch,
        stats,
        multiclass=multiclass,
        include_test=True,
    )

    # initial val loss for early stopping
    prev_val_loss = stats[0][1]

    # TODO: patience for early stopping
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
            axes,
            tr_loader,
            va_loader,
            te_loader,
            model,
            criterion,
            epoch + 1,
            stats,
            multiclass=multiclass,
            include_test=True,
        )

        # Save model parameters
        save_checkpoint(model, epoch + 1, model_name, stats)

        curr_patience, prev_val_loss = early_stopping(
            stats, curr_patience, prev_val_loss
        )
        epoch += 1

    print("Finished Training")

    # Keep plot open
    utils.save_challenge_training_plot(task)
    utils.hold_training_plot()


task_2_num_classes = {
    "source": 10,
    "target": 2,
}


def train_task(task):
    """Train transfer learning model and display training plots.

    Train four different models with {0, 1, 2, 3} layers frozen.
    """
    # data loaders
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task=task,
        batch_size=config("challenge.batch_size"),
    )

    model = ViT(
        num_patches=16, num_heads=2, num_hidden=16, num_blocks=2, num_classes=task_2_num_classes[task]
    )
    print(f"Loading {task} model...")
    model_name = config(f"challenge.{task}_checkpoint")
    model, start_epoch, stats = restore_checkpoint(model, model_name)

    train(tr_loader, va_loader, te_loader, model, f'./checkpoints/challenge_{task}', task)


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
    # train source data
    task = 'source'
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task=task,
        batch_size=config("challenge.batch_size"),
        transform=transform,
    )
    model = Challenge(num_classes=8, dropout_prob=0.4)
    model_name = config(f"challenge.{task}_checkpoint")
    model, _, _ = restore_checkpoint(
        model, model_name, force=False, pretrain=False
    )
    train(tr_loader, va_loader, te_loader, model, model_name, task, multiclass=True)

    # train transfer model
    task = 'target'
    tr_loader, va_loader, te_loader, _ = get_train_val_test_loaders(
        task=task,
        batch_size=config("challenge.batch_size"),
        transform=transform,
    )
    model = Challenge(num_classes=2, dropout_prob=0.4)
    model, _, _ = restore_checkpoint(
        model, config('challenge.source_checkpoint'), force=True, pretrain=True,
    )  # type: Challenge
    model_name = config(f"challenge.{task}_checkpoint")
    train(tr_loader, va_loader, te_loader, model, model_name, task)


if __name__ == "__main__":
    main()
