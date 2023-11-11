import backend_config

from flask import Flask, request

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


app = Flask(__name__)


@app.route('/get', methods=['GET'])
def index():
    global start_sec, end_sec, model
    
    if request.method == 'GET':
        video_path = request.args.get('video')

        if video_path is None or not (video_path.split('.')[-1] in backend_config.VIDEO_FORMATS):
            return 'Файл не найден или не поддерживается'

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = transform(video_data)

        inputs = video_data["video"]
        inputs = [i.to(backend_config.DEVICE)[None, ...] for i in inputs]

        preds = model(inputs)

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=backend_config.OUTPUT_CLASSES_NUMBER).indices

        # Вывод предсказанных классов
        pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]

        return ", ".join(pred_class_names)


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
                0, frames.shape[1] - 1, frames.shape[1] // backend_config.ALPHA
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


# Загрузка модели
model = torch.hub.load(backend_config.MODEL_WEB_PATH,
                       model=backend_config.MODEL_NAME,
                       pretrained=True)

model.blocks[6].proj = nn.Linear(in_features=2304,
                                 out_features=backend_config.CLASSES_NUMER,
                                 bias=True)

checkpoint = torch.load(backend_config.MODEL)
print(checkpoint.keys())
model.load_state_dict(checkpoint['state_dict'], strict=False)

print(dir(model))

model = model.to(backend_config.DEVICE)
model = model.eval()

# Загрузка классов
with open(backend_config.CLASSES, "r") as f:
    kinetics_classnames = json.load(f)

kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

# Подготовка обработчиков видео
transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(backend_config.NUM_FRAMES),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(backend_config.MEAN, backend_config.STD),
            ShortSideScale(
                size=backend_config.SIDE_SIZE
            ),
            CenterCropVideo(backend_config.CROP_SIZE),
            PackPathway()
        ]
    ),
)

start_sec = 0
end_sec = start_sec + backend_config.CLIP_DURATION


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
