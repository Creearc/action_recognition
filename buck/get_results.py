import os
import json
import cv2
import numpy as np
import model_config # <--TODO

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

# Для обработки видео на вход нейронной сети
class PackPathway(torch.nn.Module):
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

start_sec = 0
end_sec = start_sec + model_config.CLIP_DURATION

# Загрузка модели
#model = torch.load('./action_recognition/models/model_scripted_5ep.pth')

model_name = "slowfast_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name,
                            pretrained=True)

model = model.to(model_config.DEVICE)
model = model.eval()

# Загрузка классов
with open(model_config.CLASSES, "r") as f: # <-- TODO
    kinetics_classnames = json.load(f)

kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# Подготовка обработчиков видео
transform = ApplyTransformToKey(
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




PATH = '../data/train_dataset_train/videos'
OUTPUT = 'result.csv'

result = dict()

data = dict()
for folder in os.listdir(PATH):
    result[folder] = []
    for file in os.listdir('{}/{}'.format(PATH, folder)):
        if (file[-len('.avi'):]=='.avi'):
            video_file_path = '{}/{}/{}'.format(PATH, folder, file)
            print(video_file_path)

            try:
                video = EncodedVideo.from_path(video_file_path)
            except Exception as e:
                print('_' * 100)
                print(e)
                print('_' * 100)
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
            video_data = transform(video_data)

            inputs = video_data["video"]
            inputs = [i.to(model_config.DEVICE)[None, ...] for i in inputs]

            preds = model(inputs)

            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=model_config.OUTPUT_CLASSES_NUMBER).indices
            pred_class_names = [int(i) for i in pred_classes[0]]

            data[video_file_path] = pred_class_names[0]

            result[folder].append(pred_class_names[0])

with open(OUTPUT, 'w') as f: # <-- TODO path
    for file in data:
        f.write('{} {}\n'.format(file, data[file]))


with open('result.json', 'w') as f:
    json.dump(result, f)
