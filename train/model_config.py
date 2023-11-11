
CLASSES = '../data/h_classnames.json'
CLASSES_NUMER = 24

MODEL_NAME = "slowfast_r50"
MODEL_WEB_PATH = "facebookresearch/pytorchvideo"
MODEL = '../models/epoch=0-step=1131.ckpt'


SIDE_SIZE = 256
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CROP_SIZE = 256
NUM_FRAMES = 32
SAMPLING_RATE = 2
FRAMES_PER_SECOND = 30
ALPHA = 4

CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FRAMES_PER_SECOND
