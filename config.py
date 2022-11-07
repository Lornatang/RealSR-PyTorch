# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# Model architecture name
model_arch_name = "realsr_rcan_x4"
# Model arch config
in_channels = 3
channels = 64
reduction = 16
num_rg = 10
num_rcab = 20
# Test upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "RealSR_RCAN_x4-RealSR_V3"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/RealSR_V3/train/x{upscale_factor}/HR"
    train_lr_images_dir = f"./data/RealSR_V3/train/x{upscale_factor}/LR"

    test_gt_images_dir = f"./data/RealSR_V3/test/x{upscale_factor}/HR"
    test_lr_images_dir = f"./data/RealSR_V3/test/x{upscale_factor}/LR"

    gt_image_size = 192
    batch_size = 16
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = f""

    # Incremental training and migration training
    resume_model_weights_path = f""

    # Total num epochs
    epochs = 1000

    # Loss function weight
    loss_weights = [1.0]

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.999)
    model_eps = 1e-8
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.99998

    # Dynamically adjust the learning rate policy
    lr_scheduler_step_size = 200
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    valid_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = f"./data/RealSR_V3/test/x{upscale_factor}/LR"
    sr_dir = f"./results/test/{exp_name}"
    gt_dir = f"./data/RealSR_V3/test/x{upscale_factor}/HR"

    model_weights_path = "./results/pretrained_models/RealSR_RCAN_x4-RealSR_V3-e52b03e4.pth.tar"
