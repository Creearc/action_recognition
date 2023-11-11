import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict

import pytorchvideo.models.resnet
import torch.nn as nn

import model_config


# Выбор устройства запуска нейронки
device = "cpu"

# Путь до тестового файла
video_path = "../data/test.avi"

model = torch.hub.load(model_config.MODEL_WEB_PATH,
                       model=model_config.MODEL_NAME,
                       pretrained=True)

model.blocks[6].proj = nn.Linear(in_features=2304,
                                 out_features=model_config.CLASSES_NUMER,
                                 bias=True)

checkpoint = torch.load(model_config.MODEL)
model.load_state_dict(checkpoint['state_dict'], strict=False)

model = model.to(device)
model = model.eval()

with open(model_config.CLASSES, "r") as f:
    kinetics_classnames = json.load(f)

kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // model_config.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(model_config.NUM_FRAMES),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(model_config.MEAN, model_config.STD),
            ShortSideScale(
                size=model_config.SIDE_SIZE
            ),
            CenterCropVideo(model_config.CROP_SIZE),
            PackPathway()
        ]
    ),
)


start_sec = 0
end_sec = start_sec + model_config.CLIP_DURATION

video = EncodedVideo.from_path(video_path)
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
video_data = transform(video_data)

inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs]

preds = model(inputs)

post_act = torch.nn.Softmax(dim=1)
preds = post_act(preds)
pred_classes = preds.topk(k=5).indices

# Вывод предсказанных классов
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
print("Predicted labels: %s" % ", ".join(pred_class_names))

