from model.enc import encoder
from model.enc import unet
from templates import *
from experiment import *

class identityConfig(unet.BeatGANsUNetConfig):
    image_size = 128
    channel_mult = (1, 1, 2, 2, 4, 4, 4)

identityEncoder = encoder.BeatGANsAutoencModel(identityConfig)

gpus = [0]
conf = autoenc_72M()
train(conf, gpus=gpus)