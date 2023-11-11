import backend_config

import threading
from flask import Flask, request, render_template, Response

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

import numpy as np
import cv2
import imutils


lock = threading.Lock()
img_q = None

app = Flask(__name__)

@app.route("/")
def index0():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")


def generate():
    global lock, img_q
    frame = None

    while True:
        with lock:
            if not (img_q is None):
                frame = img_q.copy()
            else:
                continue
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


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



@app.route('/long', methods=['GET'])
def index2():
    global start_sec, end_sec, model
    
    if request.method == 'GET':
        video_path = request.args.get('video')

        if video_path is None or not (video_path.split('.')[-1] in backend_config.VIDEO_FORMATS):
            return 'Файл не найден или не поддерживается'

        tr1 = threading.Thread(target=long_video, args=(video_path, ))
        tr1.start()
        
        return "ok"


def long_video(video_path):
    global lock, img_q
    
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for i in range(0, frame_count, backend_config.VIDEO_PART_FRAMES):
        video_out = cv2.VideoWriter(backend_config.VIDEO_PART_SAVE,
                                codec,
                                frame_rate, (w, h))

        for frame_num in range(backend_config.VIDEO_PART_FRAMES):
            video.set(1, i + frame_num)
            _, frame = video.read()
            if frame is None:
                continue

            video_out.write(frame)
        video_out.release()

        video_nn = EncodedVideo.from_path(backend_config.VIDEO_PART_SAVE)
        video_data = video_nn.get_clip(start_sec=start_sec, end_sec=end_sec)
        video_data = transform(video_data)

        inputs = video_data["video"]
        inputs = [i.to(backend_config.DEVICE)[None, ...] for i in inputs]

        preds = model(inputs)

        post_act = torch.nn.Softmax(dim=1)
        preds = post_act(preds)
        pred_classes = preds.topk(k=backend_config.OUTPUT_CLASSES_NUMBER).indices

        # Вывод предсказанных классов
        pred_class_name = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
        print(pred_class_name)

        for frame_num in range(backend_config.VIDEO_PART_FRAMES):
            video.set(1, i + frame_num)
            _, img_q = video.read()
            if img_q is None:
                continue

            cv2.putText(img_q, str(pred_class_name), (5, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   0.7, (255, 0, 255), 2, cv2.LINE_AA) 

    video.release()
    print('End Thread')
    return
    

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
model = torch.load(backend_config.MODEL)

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
