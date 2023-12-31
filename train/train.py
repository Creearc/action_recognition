import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo.models.resnet

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)


from torch.utils.data import DistributedSampler


alpha = 4
train_loss_file = "train_loss.txt"
val_loss_file = "val_loss.txt"
with open(train_loss_file, "w") as f:
    f.write("")
with open(val_loss_file, "w") as f:
    f.write("")

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

num_frames = 32
sampling_rate = 2
frames_per_second = 30
    
class KineticsDataModule(pytorch_lightning.LightningDataModule):

    # Dataset configuration
    

    _DATA_PATH = "./"
    _CLIP_DURATION = (num_frames * sampling_rate)/frames_per_second
    _BATCH_SIZE = 8
    _NUM_WORKERS = 0  # Number of parallel processes fetching data


    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train.csv. Add transform that subsamples and
        normalizes the video before applying the scale, crop and flip augmentations.
        """
        train_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                    PackPathway()
                  ]
                ),
              ),
            ]
        )
        train_dataset = pytorchvideo.data.Kinetics(
            data_path="train.csv",
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform,
            #video_sampler=DistributedSampler,
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(num_frames),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    #RandomHorizontalFlip(p=0.5),
                    PackPathway()
                  ]
                ),
              ),
            ]
        )
        val_dataset = pytorchvideo.data.Kinetics(
            data_path="test.csv",
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=val_transform,
            #video_sampler=DistributedSampler,
            decode_audio=False,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )


def make_kinetics_resnet():
    return pytorchvideo.models.resnet.create_resnet(
        input_channel=3, # RGB input from Kinetics
        model_depth=50, # For the tutorial let's just use a 50 layer network
        model_num_class=24, # Kinetics has 400 classes so we need out final head to align
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        #self.model = make_kinetics_resnet()
        model_name = "slowfast_r50"
        #self.model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name,
        #                            pretrained=True)
        self.model = torch.load("action_recognition/models/model_scripted_5ep.pth") # <-- TODO

        print(dir(self.model))
##        for d in self.model.state_dict():
##            print(d)

##        for i in range(1, 6):
##            self.model.blocks[i].requires_grad_ = False
        
        print(self.model.blocks[6].proj)
        self.model.blocks[6].proj = nn.Linear(in_features=2304, out_features=24, bias=True)
        print(self.model.blocks[6].proj)

        print(self.model.blocks[0])


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        with open(train_loss_file, "a") as f:
            f.write("{}\n".format(loss.item()))
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        with open(val_loss_file, "a") as f:
            f.write("{}\n".format(loss.item()))
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-2)


if __name__ == '__main__':
    def train():
        classification_module = VideoClassificationLightningModule()
        data_module = KineticsDataModule()
        trainer = pytorch_lightning.Trainer(accelerator='gpu', devices=1, max_epochs=10)
        trainer.fit(classification_module, data_module)
        torch.save(classification_module.model, "./model_scripted_10ep.pth")


    train()
