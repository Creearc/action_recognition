import os
import random

PATH_VIDEOS = 'train_dataset_train/videos'
PATH_CLASSES = 'train_dataset_train/classes.csv'

val_percent = 0.2
PATH_TRAIN_OUTPUT = 'annotations/train.csv'
PATH_TEST_OUTPUT = 'annotations/test.csv'

random.seed(8)

with open(PATH_CLASSES, 'r') as f:
    classes = f.read().split('\n')[1:-1]

classes_dict = dict()
for cl in classes:
    classes_dict[cl.split(',')[1]] = cl.split(',')[0]
    
print(classes_dict)

with open('h_classnames.json', 'w') as f:
    f.write('{')
    for key, value in classes_dict.items():
        f.write('"{}": {}, '.format(key, value))
    f.write('}')
    
data = dict()

for folder in os.listdir(PATH_VIDEOS):
    for file in os.listdir('{}/{}'.format(PATH_VIDEOS, folder)):
        video_file_path = '{}/{}/{}'.format(PATH_VIDEOS, folder, file)

        if video_file_path[-len('.mp4'):] == '.mp4':
            data[video_file_path] = classes_dict[folder]
        

files = [i for i in data.keys()]
random.shuffle(files)

train_files = files[:-int(len(files) * val_percent)]
test_files = files[-int(len(files) * val_percent):]

with open(PATH_TRAIN_OUTPUT, 'w') as f:
    for file in train_files:
        f.write('{} {}\n'.format(file, data[file]))

with open(PATH_TEST_OUTPUT, 'w') as f:
    for file in test_files:
        f.write('{} {}\n'.format(file, data[file]))
