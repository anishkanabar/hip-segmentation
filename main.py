"""
# Description: Main training script for the project.
"""

import argparse
import tomli
import torch
import initialize_logging
import datetime
import os
import monai

from monai.networks.nets import FlexibleUNet
from src.lm import HipSegmenter
from src.dm import HipSegDataModule

from initialize_logging import log_init
from lightning.pytorch import Trainer

from pytorch_lightning.loggers import WandbLogger

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    ModelCheckpoint
)

# ----------------------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------------------

torch.set_float32_matmul_precision("high")

# ----------------------------------------------------------------------------------------
# train
# ----------------------------------------------------------------------------------------

import wandb
wandb.init(project="mjsw",
           name='mgss')

def train(config_dict: dict):
    """Main training function for the project."""
    
    log_init(config_dict=config_dict)

    model = monai.networks.nets.FlexibleUNet(
        in_channels=1,
        out_channels=config_dict["MODEL_OUT_LENGTH"],
        backbone='efficientnet-b6',
        dropout=0.1
    )

    dm = HipSegDataModule(
        data_dir=config_dict["DATA_DIR"],
        batch_size=config_dict["BATCH_SIZE"],
        val_frac=config_dict["VAL_FRAC"],
        test_frac=config_dict["TEST_FRAC"],
        seg_labels=config_dict["class_names"]
    )

    lm = HipSegmenter(
        model=model,
        max_epochs=config_dict["MAX_EPOCHS"],
        LR=config_dict["INITIAL_LR"],
        NUM_CLASSES=len(config_dict["class_names"]),
        CLASS_NAMES=config_dict["class_names"]
    )

    model_ckpt = ModelCheckpoint(
        dirpath=os.path.join(config_dict["EXPERIMENT_DIR"], "weights"),
        filename='{epoch}-{val_dice:.2f}',
        mode='max',
        monitor="Val/Dice"
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = WandbLogger(
    save_dir=os.path.join(config_dict["EXPERIMENT_DIR"], "wandb_runs")
    )
    
    trainer = Trainer(
        max_epochs=config_dict["MAX_EPOCHS"],
        accelerator='gpu',
        devices=config_dict["DEVICE"],
        logger=logger,
        log_every_n_steps=5,
        callbacks=[
            lr_monitor, 
            RichProgressBar(),
            model_ckpt
        ]
    )
    trainer.fit(lm, datamodule=dm)

# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------


def main(args: argparse.Namespace):
    """Main function for the project.

    Args:
        args (argparse.Namespace): Parsed arguments.

    """
    try:
        with open(args.config, "rb") as f:
            config_dict = tomli.load(f)
    except FileNotFoundError:
        print("Configuration file not found, check path.")
        return

    if args.mode == "train":
        train(config_dict)
    else:
        raise NotImplementedError(f"Mode {args.mode} not implemented.")

if __name__ == "__main__":

    # Parser
    parser = argparse.ArgumentParser(description="Project Training Script")

    # Add the mode argument to the parser
    parser.add_argument(
        "-m", "--mode",
        type=str,
        default="train",
        choices=["train"],
        help="Operation mode: 'train' is the only current option.",
    )
    # Add the config file argument to the parser
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to file with experiment configuration settings",
    )

    # Get the parsed arguments
    args = parser.parse_args()

    # Run the main function.
    main(args)
