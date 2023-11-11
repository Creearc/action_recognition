# action_recognition
## Описание проекта
Проект создан в рамках хакатона. Задача - разработка модели классификации действий людей по видео.

Программы для подготовки датасетов расположены в папке ```dataset_preparation```.  
Чтобы обучить модель нейронной сети и проверить качество ее работы, необходимо воспользоваться программами из папки ```train```.  
Сервис с пользовательским интерфейсом расположен в папке ```service```.  



## Model
[Ссылка на документацию по модели нейронной сети](https://pytorchvideo.org/docs/tutorial_classification)

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
pip install dash_daq
pip install imageio
```

## Cuda for Windows
CUDA Toolkit  
[Прямая ссылка на скачивание](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)

Cudnn archive  
[Архив версий cudnn](https://developer.nvidia.com/rdp/cudnn-archive)

Cudnn 8.9.5 version  
[Ссылка на скачивание. Потребуется авторизация.](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.5/local_installers/11.x/cudnn-windows-x86_64-8.9.5.30_cuda11-archive.zip)

Torch versions with CUDA versions  
[Ссылка на список версий pytorch и совместимых с ними версий CUDA](https://pytorch.org/get-started/previous-versions/)

```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Check CUDA
```
python -c "import torch; print('CUDA available' if torch.cuda.is_available() else 'CUDA not available')" 
```
