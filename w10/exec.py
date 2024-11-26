import logging
import os
import sys
from glob import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_2d, list_data_collate, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandSpatialCropd,
)
from monai.visualize import plot_2d_or_3d_image

# This code is modified from the MONAI 2D segmentation tutorial.
# The license below

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
def _get_data(train_dir):
    images = sorted(glob(os.path.join(train_dir, "*T1w.npy")))
    segs = sorted(glob(os.path.join(train_dir, "*T1w_tissue_segmentation.npy")))
    files = [{"img": img, "seg": seg} for img, seg in zip(images, segs)]
    return files


def _get_training_transforms():
    # TODO:
    # These are the basic necessary transforms
    # for loading the training data.
    # The two first are necessary, i.e.,
    # loading and reshaping the data.
    # Add as many as you feel necessary after those two.
    # It's probably smart to keep the cropping as the last step though
    # feel free to change the size. It's used in the validation loop.
    #
    # See available transforms here:
    # https://docs.monai.io/en/stable/transforms.html#dictionary-transforms
    #
    # NOTE: remember to import whatever you use, you can see examples
    # at the beginning of this code block.
    # NOTE: Not all augmentations are suitable for *both* images and segmentations
    # if you want to only apply an augmentation on the image, you should only pass
    # ["img"] as the keys

    roi_size = [200, 200]

    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
            RandSpatialCropd(
                  keys=["img", "seg"],roi_size=roi_size),
        ]
    )


    return train_transforms, roi_size


def _get_validation_transforms():
    # NOTE: you do not need to change the validation transforms
    # only the training transforms.

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            EnsureChannelFirstd(keys=["img", "seg"]),
        ]
    )

    return val_transforms

def _get_data_loaders(train_files, train_transforms, val_files, val_transforms):

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)

    return train_loader, val_loader

def _loss_function(outputs, labels):
    # TODO:
    # I'm only using a Dice loss here.
    # You can change that (or keep it)
    # and also add different losses.
    # If you want to add losses
    # first grab the loss from the monai module
    # (like is done for Dice here)
    # and then add it to the full_loss.
    # Feel free to use different weights.
    #
    # See available segmentation losses here:
    # https://docs.monai.io/en/stable/losses.html#loss-functions
    #

    dice_loss = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    full_loss = dice_loss(outputs, labels)

    return full_loss

def _get_model(device):
    # TODO:
    # I'm using the most basic U-Net implemented in MONAI
    # here. You can either modify this architecture or
    # pick another one. The full list is here:
    # https://docs.monai.io/en/stable/networks.html#nets
    #
    # NOTE: The spatial_dims, in_channels and out_channels
    # will not change if you pick another network. Those are
    # fixed by the training data.

    model = monai.networks.nets.BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=4,
        features=(16, 32, 32, 64, 128, 16),
    ).to(device)

    return model

def _get_optimizer(model):
    # TODO:
    # I'm using vanilla stochastic gradient descent here.
    # You can change it's paremeters, e.g., momentum, or
    # change the optimizers. See available algorithms here:
    # https://pytorch.org/docs/stable/optim.html#algorithms

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0)

    return optimizer

# Training function start here
def train(train_dir, validation_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Get the training data files (images and segmentation)
    train_files = _get_data(train_dir)

    # Set up training transforms.
    train_transforms, roi_size = _get_training_transforms()

    # Get the validation data files (images and segmentations)
    val_files = _get_data(validation_dir)

    # Get the validation transforms.
    val_transforms = _get_validation_transforms()


    # Get data loaders for running the training and validation
    train_loader, val_loader = _get_data_loaders(train_files, train_transforms, val_files, val_transforms)

    # The device variable is used to automatically run the training on
    # the GPU if its available, otherwise the CPU is used
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: Change the _get_model function
    model = _get_model(device)

    # TODO: Change the _get_optimizer function
    optimizer = _get_optimizer(model)


    # Final thing before training. Setup the validation metric so we can measure
    # the validation performance during training.
    # I'm not including the background into the validation dice as we are more interest
    # in the foreground structures.
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # Define post-processing steps for the output so we can compare it to the validation segmentations
    # The loss does this automatically because we have defined
    # to_onehot_y=True, softmax=True
    # so this is not necessary for training, just for validation
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])

    # Okay start the training loop
    # Feel free to change the number of
    # epochs and the other parameters

    # Validation interval
    val_interval = 10

    # Keep track of the best metric & epoch
    best_metric = -1
    best_metric_epoch = -1

    # TODO:
    # You can use even more the 500 epochs if you want.

    epochs = 500

    # Losses and metrics
    epoch_loss_values = list()
    metric_values = list()

    # Save a log into a directory called log
    writer = SummaryWriter(log_dir='./log')

    for epoch in range(epochs):
        print("-" * 100)
        print(f"epoch {epoch + 1}/{epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1

            #Get training data for this batch
            inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)
            optimizer.zero_grad()

            # Push the input through the model
            outputs = model(inputs)

            # Get the loss
            loss = _loss_function(outputs, labels)

            # Backpropagate and call the optimizer
            loss.backward()
            optimizer.step()

            # Store the losses
            epoch_loss += loss.item()
            print(f"{step}/{len(train_loader)}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), len(train_loader) * epoch + step)

        # These statements plot examples into the tensorboard log dile
        plot_2d_or_3d_image(inputs[0], epoch + 1, writer, index=0, tag="ínput_image")
        plot_2d_or_3d_image(labels[0], epoch + 1, writer, index=0, tag="input_labeling")
        output_tmp = [post_trans(i) for i in decollate_batch(outputs)]
        plot_2d_or_3d_image(output_tmp[0], epoch + 1, writer, index=0, tag="input_prediction")

        # Save the average loss per epoch
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Run the validation, every val_interval steps
        if (epoch + 1) % val_interval == 0:
            model.eval()

            # The no_grad is just to tell torch that it doesn't need to backpropagate here
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                im_num = 0
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                    # Plot the validation images, labels and predictions into the log file
                    plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag="validation_image"+str(im_num))
                    plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag="validation_labeling"+str(im_num))
                    plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag="validation_prediction"+str(im_num))
                    im_num += 1

                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)

                # Keep track of the best metric and save the best model
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), "best_metric_model_segmentation2d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()

# This function is used for the testing the performance once the test data
# is released
def test(test_dir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    test_files = _get_data(test_dir)
    test_transforms = _get_validation_transforms()
    test_ds = monai.data.Dataset(data=test_files, transform=test_transforms)
    # sliding window inference need to input 1 image in every iteration
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(softmax=True), AsDiscrete(argmax=True)])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists('output'):
        os.mkdir('output')

    model = _get_model(device)
    model.load_state_dict(torch.load("best_metric_model_segmentation2d_dict.pth"))

    model.eval()

    sub = 0
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data["img"].to(device), test_data["seg"].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (200, 200)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(test_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(test_labels)
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

            for val_output, val_label in zip(val_outputs, val_labels):
                labels = val_output.unique().tolist()
                out_shape = list(val_label.shape)
                out_shape[0] = len(labels)
                tmp = np.zeros(tuple(out_shape))
                for i in labels:
                    tmp[int(i), :, :] = val_output.cpu() == int(i)

                plt.imsave(os.path.join('output', str(sub)+'_unet_segmentation.png'), np.moveaxis(tmp[1:,:,:], 0, -1))

                tmp = np.zeros(tuple(out_shape))
                for i in labels:
                    tmp[int(i), :, :] = val_label.cpu() == int(i)

                plt.imsave(os.path.join('output',  str(sub)+'_segmentation.png'), np.moveaxis(tmp[1:,:,:], 0, -1))
                sub+=1

        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())

        # reset the status
        dice_metric.reset()

train('./training_data_exercise/training_data/', './training_data_exercise/validation_data/')