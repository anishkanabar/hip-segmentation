"""
Description: A script containing the data module for the project.
"""
# import pytorch_lightning as pl
import lightning as L

import monai
import copy
import numpy as np
import fnmatch
import os
import slicerio
import nrrd
import logging
import osail_utils
import math

import monai.transforms as transforms
from monai.data.utils import pad_list_data_collate, no_collation

class HipSegDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "./", 
            cache_dir: str = "./", 
            batch_size: int = 16,
            # batch_size can only be 8 or 16
            val_frac: float = 0.15, 
            test_frac: float = 0.15,
            seg_labels: dict = {}
        ):
        super().__init__()
        self.data_dir = data_dir
        self.CACHE_DIR = cache_dir
        self.batch_size = batch_size
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seg_labels = seg_labels
        self.train_transforms = build_train_transforms(seg_labels=seg_labels)
        self.val_transforms = build_val_transforms(seg_labels=seg_labels)

    def prepare_data(self):

        #list_subfolders_with_paths = [f.path for f in os.scandir(self.data_dir) if f.is_dir()]
        ds = []

        seg_pattern = '*.seg.nrrd'
        seg_matches = fnmatch.filter(os.listdir(self.data_dir), seg_pattern)
        
        for seg_match in seg_matches:
            if 'dcm' in seg_match:
                img_match = seg_match.replace(".dcm_Segmentation.seg.nrrd", ".nrrd")
            else: 
                img_match = seg_match.replace("_Segmentation.seg.nrrd", ".nrrd")
            ds.append({
                "label": os.path.join(self.data_dir, seg_match),
                "image": os.path.join(self.data_dir, img_match)
            })

        # for folder in list_subfolders_with_paths:
        #     if 'MG' in folder:
        #         seg_pattern = '*SEG*.nrrd'
        #     elif 'SS' in folder:
        #         seg_pattern = '*Template*.nrrd'
        #     else:
        #         seg_pattern = '*.seg.nrrd'
        #         #seg_pattern = 'none'
        #     seg_match = fnmatch.filter(os.listdir(folder), seg_pattern)
            
        #     if len(seg_match)==0:
        #         continue
        #     else:
        #         seg_match = seg_match[0]

        #     if 'MG' in folder: 
        #         img_match = seg_match.replace("MG SEG ", "img_")
        #     elif 'SS' in folder:
        #         temp_seg_match = seg_match[6:]
        #         img_match = temp_seg_match.replace("mJSW_Hip_Template", "img_")
        #     else:
        #         img_match = seg_match.replace(".dcm_Segmentation.seg.nrrd", ".nrrd") 

        #     ds.append({
        #         "label": os.path.join(folder, seg_match),
        #         "image": os.path.join(folder, img_match)
        #     })
        self.dataset = ds

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train_list, val_list, test_list = get_train_val_test_splits(
                data_dict=self.dataset,
                val_frac=self.val_frac,
                test_frac=self.test_frac
            )
            logging.info(f"""
            Length of training data: {len(train_list)}
            Length of validation data: {len(val_list)}
            """)
            self.train_ds = monai.data.Dataset(
                data=train_list, transform=self.train_transforms
            )

            self.val_ds = monai.data.Dataset(
                data=val_list, transform=self.val_transforms
            )
        return train_list
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.test_ds = monai.data.Dataset(
                data=test_list, transform=self.val_transforms
            )
        if stage == "predict":
            self.predict_ds = monai.data.Dataset(
                data=self.dataset, transform=self.val_transforms
            )
    def train_dataloader(self):
        return monai.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            shuffle=True, # change back to True
            num_workers=2
            #collate_fn=pad_list_data_collate # remove 
        )

    def val_dataloader(self):
        return monai.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self):
        return monai.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def predict_dataloader(self):
        return monai.data.DataLoader(
            data=self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )

def build_train_transforms(**kwargs):

    img_height = kwargs.get("img_height", 1024)
    img_width = kwargs.get("img_width", 1024)
    seg_labels = kwargs.get("seg_labels", {})

    train_trans = transforms.Compose(
        [
            #osail_utils.io.LoadImageD(keys=['image'], pad=True, target_shape=(1024, 1024), percentile_clip=2.5, normalize=True, standardize=True, dtype=None),
            transforms.LoadImageD(keys=["image"], image_only=True, prune_meta_pattern=".*|.*", prune_meta_sep=" "),
            LoadSegmentationD(keys=["label"], labels=seg_labels),
            transforms.EnsureChannelFirstD(keys=['image'], channel_dim=0),
            #transforms.Rotated(keys=["label"], angle=-math.pi/2),

            transforms.TransposeD(keys=["image"], indices=[2,1,0]),
            transforms.TransposeD(keys=["label"], indices=[0,2,1]),
            Pad2Square(keys=["image", "label"]),
            transforms.ResizeD(keys=["image", "label"], spatial_size=(img_height, img_width), mode="nearest"),

            transforms.NormalizeIntensityD(keys=["image"]),
            transforms.ScaleIntensityD(keys=["image"]),
            
            transforms.RandFlipD(keys=["image", "label"], prob=0.5),
            #transforms.RandGaussianNoiseD(keys=["image"], prob=0.5),
            #transforms.RandRotateD(keys=["image", "label"], prob=0.5, range_x=0.6, mode="nearest"),
            #transforms.RandAdjustContrastD(keys=["image"], gamma=(0.2,4))
        ]
    )
    
    return train_trans

def build_val_transforms(**kwargs):
    
    img_height = kwargs.get("img_height", 1024)
    img_width = kwargs.get("img_width", 1024)
    seg_labels = kwargs.get("seg_labels", {})
    
    val_trans = transforms.Compose(
        [
 #osail_utils.io.LoadImageD(keys=['image'], pad=True, target_shape=(1024, 1024), percentile_clip=2.5, normalize=True, standardize=True, dtype=None),
            transforms.LoadImageD(keys=["image"], image_only=True, prune_meta_pattern=".*|.*", prune_meta_sep=" "),
            LoadSegmentationD(keys=["label"], labels=seg_labels),
            transforms.EnsureChannelFirstD(keys=['image'], channel_dim=0),
            #transforms.Rotated(keys=["label"], angle=-math.pi/2),

            transforms.TransposeD(keys=["image"], indices=[2,1,0]),
            transforms.TransposeD(keys=["label"], indices=[0,2,1]),
            Pad2Square(keys=["image", "label"]),
            transforms.ResizeD(keys=["image", "label"], spatial_size=(img_height, img_width), mode="nearest"),

            transforms.NormalizeIntensityD(keys=["image"]),
            transforms.ScaleIntensityD(keys=["image"]),
            
            # transforms.RandGaussianNoiseD(keys=["image"], prob=0.5),
            # transforms.RandRotateD(keys=["image", "label"], prob=0.5, range_x=0.6, mode="nearest"),
            # transforms.RandAdjustContrastD(keys=["image"], gamma=(0.2,4))
        ]
    )
    
    return val_trans

class Pad2Square(transforms.Transform):
    def __init__(self, keys: list[str]) -> None:
        super().__init__()
        self.keys = keys

    def __call__(self, data):
        data_copy = copy.deepcopy(data)
        for key in self.keys:
            if key in data:
                img = data[key]
                s = list(img.shape)
                max_wh = np.max([s[-1], s[-2]])
                hp = int((max_wh - s[-1]) / 2)
                vp = int((max_wh - s[-2]) / 2)
                padding = ((0,0), (vp, vp), (hp, hp))
                img = np.pad(img, padding, mode="constant")
                data_copy[key] = img
        return data_copy

def get_train_val_test_splits(data_dict, val_frac, test_frac):
    
    length = len(data_dict)
    indices = np.arange(length)
    np.random.shuffle(indices)

    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]

    train = [data_dict[i] for i in train_indices]
    val = [data_dict[i] for i in val_indices]
    test = [data_dict[i] for i in test_indices]

    return train, val, test

class LoadSegmentationD(transforms.Transform):
    def __init__(self, keys: list[str], labels: dict) -> None:
        super().__init__()
        self.keys = keys
        self.labels = labels
    
    def __call__(self, data):
        data_copy = copy.deepcopy(data)
        for key in self.keys:
            if key in data:
                output, path = load_segmentations(data[key], self.labels)
                data_copy[key] = output
        return data_copy       

def load_segmentations(path, labels):
    
    ## Load Data
    segmentation_info = slicerio.read_segmentation(path)
    voxels, _ = nrrd.read(path)
    
    ## Add channels dim if not present
    if len(voxels.shape) == 3:
        voxels = np.expand_dims(voxels, axis=0)

    ## Prep Empty Volume
    x = voxels.shape[1]
    y = voxels.shape[2]
    channels = len(segmentation_info["segments"])

    # Comment below print statement out when I am done debugging
    #print(f"Segmentation: {path} has {channels} channels with dims {voxels.shape}")
    output = np.zeros((x,y,channels))
    
    ## Loop through layers
    for i, segment in enumerate(segmentation_info["segments"]):
        
        ## Extract Metadata
        layer = segment["layer"]
        layer_name = segment["name"]
        labelValue = segment["labelValue"]
        
        ## Set up new layer based on voxel value from segmentation info
        layer_voxels = np.moveaxis(voxels, 0, -1)
        layer_voxels = np.squeeze(layer_voxels, axis=-2)
        indx = (layer_voxels[..., layer] == labelValue).nonzero()
        new_layer = np.zeros(layer_voxels[..., layer].shape)
        new_layer[indx] = labelValue
        
        ## Assign the new layer to the output based on defined channel order
        output[...,labels[str.lower(layer_name)]] = new_layer
    
    output = np.where(np.moveaxis(output, -1, 0) > 0.0, 1, 0)
    return output, path