"""
EECS 445 - Introduction to Machine Learning
Fall 2024  - Project 2

Helper file for common training functions.
"""
import time

from torch.distributed.tensor.parallel import loss_parallel

from utils import config
import numpy as np
import itertools
import os
import torch
from torch.nn.functional import softmax
from sklearn import metrics
import utils
import pdb


def count_parameters(model):
    """Count number of learnable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, epoch, checkpoint_dir, stats):
    """
    Save the 'model' parameters, the cumulative stats, and current epoch number as a checkpoint file (.pth.tar) in 'checkpoint_dir'. 
    Args:
        model: The model to be saved. This is one of the model classes in the 'model' folder.
        epoch (int): The current epoch number.
        checkpoint_dir (str): Directory where the checkpoint file will be saved.
        stats (list): A cumulative list consisted of all the model accuracy, loss, and AUC for every epoch up to the current epoch. 
             Note: we will almost always use the last element of stats -- stats[-1] -- which represents the most recent stats. 

    Description:
        This function saves the current state of the model, including its parameters, epoch number, and
        training statistics to a checkpoint file. The checkpoint file is named according to the current
        epoch, and if the specified directory does not exist, it will be created.
    """
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "stats": stats,
    }

    filename = os.path.join(checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(epoch))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir,exist_ok=True)
    torch.save(state, filename)


def check_for_augmented_data(data_dir):
    """Ask to use augmented data if `augmented_dogs.csv` exists in the data directory."""
    if "augmented_dogs.csv" in os.listdir(data_dir):
        print("Augmented data found, would you like to use it? y/n")
        print(">> ", end="")
        rep = str(input())
        return rep == "y"
    return False


def restore_checkpoint(model, checkpoint_dir, cuda=False, force=False, pretrain=False):
    """
    Restore model from checkpoint if it exists.

    Args:
        model (torch.nn.Module): The model to be restored.
        checkpoint_dir (str): Directory where checkpoint files are stored.
        cuda (bool, optional): Whether to load the model on GPU if available. Defaults to False.
        force (bool, optional): If True, force the user to choose an epoch. Defaults to False.
        pretrain (bool, optional): If True, allows partial loading of the model state (used for pretraining). Defaults to False.

    Returns:
        tuple: The restored model, the starting epoch, and the list of statistics.

    Description:
        This function attempts to restore a saved model from the specified `checkpoint_dir`.
        If no checkpoint is found, the function either raises an exception (if `force` is True) or returns
        the original model and starts from epoch 0. If a checkpoint is available, the user can choose which
        epoch to load from. The model's parameters, epoch number, and training statistics are restored.
    """
    try:
        cp_files = [
            file_
            for file_ in os.listdir(checkpoint_dir)
            if file_.startswith("epoch=") and file_.endswith(".checkpoint.pth.tar")
        ]
    except FileNotFoundError:
        cp_files = None
        os.makedirs(checkpoint_dir)
    print(os.listdir(checkpoint_dir), checkpoint_dir)
    if not cp_files:
        print("No saved model parameters found")
        if force:
            raise Exception("Checkpoint not found")
        else:
            return model, 0, []

    # Find latest epoch
    for i in itertools.count(1):
        if "epoch={}.checkpoint.pth.tar".format(i) in cp_files:
            epoch = i
        else:
            break

    if not force:
        print(
            "Which epoch to load from? Choose in range [0, {}].".format(epoch),
            "Enter 0 to train from scratch.",
        )
        print(">> ", end="")
        inp_epoch = int(input())
        if inp_epoch not in range(epoch + 1):
            raise Exception("Invalid epoch number")
        if inp_epoch == 0:
            print("Checkpoint not loaded")
            clear_checkpoint(checkpoint_dir)
            return model, 0, []
    else:
        print("Which epoch to load from? Choose in range [1, {}].".format(epoch))
        inp_epoch = int(input())
        if inp_epoch not in range(1, epoch + 1):
            raise Exception("Invalid epoch number")

    filename = os.path.join(
        checkpoint_dir, "epoch={}.checkpoint.pth.tar".format(inp_epoch)
    )

    print("Loading from checkpoint {}?".format(filename))

    if cuda:
        checkpoint = torch.load(filename)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)

    try:
        start_epoch = checkpoint["epoch"]
        stats = checkpoint["stats"]

        if pretrain:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"])
        print(
            "=> Successfully restored checkpoint (trained for {} epochs)".format(
                checkpoint["epoch"]
            )
        )
    except:
        print("=> Checkpoint not successfully restored")
        raise

    return model, inp_epoch, stats


def clear_checkpoint(checkpoint_dir):
    """Remove checkpoints in `checkpoint_dir`."""
    filelist = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth.tar")]
    for f in filelist:
        os.remove(os.path.join(checkpoint_dir, f))

    print("Checkpoint successfully removed")


def early_stopping(stats, curr_count_to_patience, prev_val_loss):
    """Calculate new patience and validation loss.

    args:
        stats (list): A cumulative list to store the model accuracy, loss, and AUC for every epoch.
            Usage: stats[epoch][0] = validation accuracy, stats[epoch][1] = validation loss, stats[epoch][2] = validation AUC
                    stats[epoch][3] = training accuracy, stats[epoch][4] = training loss, stats[epoch][5] = training AUC
                    stats[epoch][6] = test accuracy, stats[epoch][7] = test loss, stats[epoch][8] = test AUC (test only appears when we are finetuning our target model)

            Note: we will almost always use the last element of stats -- stats[-1] -- which represents the most recent stats. 
        
        curr_count_to_patience (int): Number of epochs since the last time the validation loss decreased.

        prev_val_loss (float): Validation loss from the previous epoch.

    Description:
        Increment curr_count_to_patience by one if new loss is not less than prev_val_loss
        Otherwise, update prev_val_loss with the current val loss, and reset curr_count_to_patience to 0

    Returns: new values of curr_count_to_patience and prev_val_loss
    """
    # TODO implement early stopping
    if  stats[-1][1] >= prev_val_loss:
        curr_count_to_patience += 1
    else:
        curr_count_to_patience = 0
        prev_val_loss = stats[-1][1]
    return curr_count_to_patience, prev_val_loss


def evaluate_epoch(
    axes,
    tr_loader,
    val_loader,
    te_loader,
    model,
    criterion,
    epoch,
    stats,
    include_test=False,
    update_plot=True,
    multiclass=False,
):
    """
    Evaluate the `model` on the train, validation, and optionally test sets on the specified 'criterion' at the given 'epoch'.

    Args:
        axes (matplotlib.axes._subplots.AxesSubplot): Axes object for plotting the training progress.

        tr_loader (DataLoader): DataLoader for the training set.

        val_loader (DataLoader): DataLoader for the validation set.

        te_loader (DataLoader): DataLoader for the test set. This is only used to compute test metrics if 'include_test' is True.

        model: The model to be evaluated. This is one of the model classes in the 'model' folder.

        criterion: The loss function used to compute the model's loss.

        epoch (int): The current epoch number. This is used for logging and plotting.

        stats (list): A cumulative list to store model accuracy, loss, and AUC for every epoch.
            Usage: stats[epoch][0] = validation accuracy, stats[epoch][1] = validation loss, stats[epoch][2] = validation AUC
                   stats[epoch][3] = training accuracy, stats[epoch][4] = training loss, stats[epoch][5] = training AUC
                   stats[epoch][6] = test accuracy, stats[epoch][7] = test loss, stats[epoch][8] = test AUC (test only appears when we are finetuning our target model)

            Note: we will almost always use the last element of stats -- stats[-1] -- which represents the most recent stats. 

        include_test (bool, optional): Whether to evaluate the model on the test set. We set this to true when we are finetuning our target model after pretraining on the source task. 

        update_plot (bool, optional): Whether to update the training plot. During training, you will see the graph change as stats are being updated if update_plot is true. 

        multiclass (bool, optional): Indicates if the task is a multiclass classification problem. Defaults to False. This is true for the source task, and false for the target task. 

    Description:
        This function sets the model to evaluation mode and evaluates it on the training, validation sets, and optionally the test set.
        If `include_test` is True, it also evaluates the model on the test set.
        The function calculates metrics such as accuracy, loss, and AUC for each dataset and appends the current statistics into 'stats'. Optionally, it also updates the training plot.

    Returns: None
    """
    model.eval()
    def _get_metrics(loader):
        '''
            Evaluates the model on the given loader (either train, val, or test) and returns the accuracy, loss, and AUC.
        '''
        y_true, y_pred, y_score = [], [], []
        correct, total = 0, 0
        running_loss = []
        for X, y in loader:
            with torch.no_grad():
                output = model(X)
                predicted = predictions(output.data)
                y_true.append(y)
                y_pred.append(predicted)
                if not multiclass:
                    y_score.append(softmax(output.data, dim=1)[:, 1])
                else:
                    y_score.append(softmax(output.data, dim=1))
                total += y.size(0)
                correct += (predicted == y).sum().item()
                running_loss.append(criterion(output, y).item())
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        y_score = torch.cat(y_score)
        loss = np.mean(running_loss)
        acc = correct / total
        if not multiclass:
            auroc = metrics.roc_auc_score(y_true, y_score)
        else:
            auroc = metrics.roc_auc_score(y_true, y_score, multi_class="ovo")
        return acc, loss, auroc

    train_acc, train_loss, train_auc = _get_metrics(tr_loader)
    val_acc, val_loss, val_auc = _get_metrics(val_loader)

    stats_at_epoch = [
        val_acc,
        val_loss,
        val_auc,
        train_acc,
        train_loss,
        train_auc,
    ]
    if include_test:
        stats_at_epoch += list(_get_metrics(te_loader))

    stats.append(stats_at_epoch)
    utils.log_training(epoch, stats)
    if update_plot:
        utils.update_training_plot(axes, epoch, stats)


def train_epoch(data_loader, model, criterion, optimizer):
    """
    Train the `model` for one epoch using data from `data_loader`.

    Args:
        data_loader: DataLoader providing batches of input data and corresponding labels.

        model: The model to be trained. This is one of the model classes in the 'model' folder. 

        criterion (torch.nn.Module): The loss function used to compute the model's loss.

        optimizer: The optimizer used to update the model parameters.

    Description:
        This function sets the model to training mode and use the data loader to iterate through the entire dataset.
        For each batch, it performs the following steps:
        1. Resets the gradient calculations in the optimizer.
        2. Performs a forward pass to get the model predictions.
        3. Computes the loss between predictions and true labels using the specified `criterion`.
        4. Performs a backward pass to calculate gradients.
        5. Updates the model weights using the `optimizer`.
    
    Returns: None
    """
    model.train()
    for i, (X, y) in enumerate(data_loader):
        # TODO implement training steps
        #Reset optimizer gradient calculations
        optimizer.zero_grad()

        #Get model predictions (forward pass)
        outputs = model(X)
        # print(outputs)

        #Calculate loss between model prediction and true labels
        loss = criterion(outputs, y)

        #Perform backward pass
        loss.backward()

        #Update model weights
        optimizer.step()
        pass


def predictions(logits):
    """Determine predicted class index given logits.

    args: 
        logits (torch.Tensor): The model's output logits. It is a 2D tensor of shape (batch_size, num_classes). 

    Returns:
        the predicted class output that has the highest probability as a PyTorch Tensor. This should be of size (batch_size,).
    """
    # TODO implement predictions
    _, output = torch.max(logits, 1)
    return output

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result
