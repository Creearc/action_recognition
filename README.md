# action_recognition
## Model
```
https://pytorchvideo.org/docs/tutorial_classification
```
## Установка библиотек

Для нейронной сети
```
pip install pytorch-lightning==1.8.0
pip install pytorchvideo==0.1.5
```

Для бэкэнда
```
pip install opencv-python==4.8.1.78
pip install numpy==1.26.1
pip install Flask==3.0.0
```

Для веба
```
pip install dash==2.14.1
pip install dash_daq==2.32.0
pip install imageio
```

## Cuda for Windows
CUDA Toolkit
```
https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
```

Cudnn archive
```
https://developer.nvidia.com/rdp/cudnn-archive 
```
8.9.5 version
```
https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.5/local_installers/11.x/cudnn-windows-x86_64-8.9.5.30_cuda11-archive.zip
```

Torch versions with CUDA versions
```
https://pytorch.org/get-started/previous-versions/
```

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Check CUDA
```
python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available')" 
```
