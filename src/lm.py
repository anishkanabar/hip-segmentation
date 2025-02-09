##########################################################################################
# Description: Script containing the Pytorch Lightning module for training.
##########################################################################################
from typing import Any
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import monai
from viz import plot_img_seg

import wandb

"""
Description: A Script containing the lightning module for training.
"""
class HipSegmenter(pl.LightningModule):
    def __init__(self, model, max_epochs, CLASS_NAMES, LR, NUM_CLASSES):
        super(HipSegmenter, self).__init__()
        self.model = model
        self.max_epochs = max_epochs
        self.LR = LR
        self.NUM_CLASSES = NUM_CLASSES
        self.CLASS_NAMES = CLASS_NAMES
        self.loss = monai.losses.DiceFocalLoss(
            include_background=True,
            sigmoid=True,
            weight=[0.2,5]
        )
        self.dice_metric = monai.metrics.DiceMetric(
            include_background=True,
            reduction="none"
        )
        self.iou_metric = monai.metrics.MeanIoU(
            include_background=True,
            reduction="none"
        )
    
    def forward(self, inputs, targets) -> Any:
        return self.model(inputs, targets)

    def training_step(self, batch, batch_idx):
        
        ## Run step
        loss, dice, per_class_dice, iou, per_class_iou = self.step(batch)

        ## Log
        self.log("Train/Loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("Train/Dice", dice, on_step=True, on_epoch=True, logger=True)
        self.log("Train/IoU", iou, on_step=True, on_epoch=True, logger=True)

        for i, label in enumerate(self.CLASS_NAMES.keys()):
            self.log(
                f"Train-PerStructDice/{label}",
                per_class_dice[i].item(),
                on_step=False,
                on_epoch=True,
                logger=True)
            
        for i, label in enumerate(self.CLASS_NAMES.keys()):
            self.log(
                f"Train-PerStructIoU/{label}", 
                per_class_iou[i].item(), 
                on_step=False,
                on_epoch=True,
                logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        ## Run step
        loss, dice, per_class_dice, iou, per_class_iou = self.step(batch, val=True)

        ## Log
        self.log("Val/Loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log("Val/Dice", dice, on_step=False, on_epoch=True, logger=True)
        self.log("Val/IoU", iou, on_step=False, on_epoch=True, logger=True)

        for i, label in enumerate(self.CLASS_NAMES.keys()):
            self.log(
                f"Val-PerStructDice/{label}",
                per_class_dice[i].item(),
                on_step=False,
                on_epoch=True,
                logger=True
            )

        # Log per-class IoU scores
        for i, label in enumerate(self.CLASS_NAMES.keys()):
            self.log(
                f"Val-PerStructIoU/{label}",  
                per_class_iou[i].item(), 
                on_step=False,
                on_epoch=True,
                logger=True
            )
        
        return loss
    
    def step(self, batch, val=False):
        
        inputs = batch["image"]
        labels = batch["label"]
        #print(inputs.shape)
        #print(labels.shape)
        outputs = self.model(inputs)
        #print(f"outputs: {outputs.shape}")
        loss = self.loss(outputs, labels)

        ## Calculate Dice     
        post_pred = monai.transforms.Compose([monai.transforms.AsDiscrete(threshold=0.5, to_onehot=None)])
        post_label = monai.transforms.Compose([monai.transforms.AsDiscrete(to_onehot=None)])
        
        outputs_proc = [post_pred(i) for i in monai.data.decollate_batch(outputs)]
        labels_proc = [post_label(i) for i in monai.data.decollate_batch(labels)]
        
        self.dice_metric(outputs_proc, labels_proc)
        dice = self.dice_metric.aggregate(reduction="mean")
        per_class_dice = self.dice_metric.aggregate(reduction="mean_batch")
        self.dice_metric.reset()

        ## Calculate IoU
        self.iou_metric(outputs_proc, labels_proc)
        iou = self.iou_metric.aggregate(reduction="mean")
        per_class_iou = self.iou_metric.aggregate(reduction="mean_batch")
        self.iou_metric.reset()

        if val:
            self.validation_step_input = inputs
            self.validation_step_output = outputs_proc 
            self.validation_step_label = labels_proc
        
        return loss, dice, per_class_dice, iou, per_class_iou
    
    # def on_validation_epoch_end(self) -> None:
        
    #     fig = plot_img_seg(
    #         image=self.validation_step_input[0],
    #         label=self.validation_step_label[0],
    #         seg=self.validation_step_output[0]
    #     )
    #     tb_logger = self.logger.experiment
    #     tb_logger.add_figure("Model Output", fig, self.current_epoch)
        
    #     return
    

    def on_validation_epoch_end(self) -> None:
    
        for i in range(3):
            fig = plot_img_seg(
                image=self.validation_step_input[i],
                label=self.validation_step_label[i],
                seg=self.validation_step_output[i]
            )
            # Use wandb to log the figures
            wandb.log({f"Model Output{i+1}": [wandb.Image(fig, caption=f"Model Output{i+1}")]}, commit=False)

        return
        
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max= self.max_epochs
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }