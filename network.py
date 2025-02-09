"""
Description: A script containing the network for the model.
"""
import monai

def create_model(IMG_HEIGHT, IMG_WIDTH, IN_CHANNELS, OUT_CHANNELS):
    model = monai.networks.nets.SwinUNETR(
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        spatial_dims= 2
    )
    return model

def create_unet_model(IN_CHANNELS, OUT_CHANNELS):
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        channels=(
            16, 16, 16, 
            32, 32, 32, 
            64, 64, 64, 
            128, 128, 128, 
            512, 512, 512, 
            1024, 1023, 1024,
            2048
        ),
        strides=(
            1, 1, 2,
            1, 1, 2, 
            1, 1, 2,
            1, 1, 2,
            1, 1, 2,
            1, 1, 2,
        ),
        dropout=0.15,
        num_res_units=2
    )
    return model