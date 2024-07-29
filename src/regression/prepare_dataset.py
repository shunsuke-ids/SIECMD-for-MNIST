import argparse
import os
import pickle

from src.preprocessing.handle_dataset import *
from src.preprocessing.augment import *

parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('data_dir', help='Path to dataset')
'''
The dataset should be formated as follows: 
data_dir
├── 0
│   ├── image_name0.format
│   └── image_name1.format
├── 1
│   ├── image_name2.format
│   ...
...
├── 360
│   ├── ...
│   ...
with 0, 1, ... being the GT angles
'''
parser.add_argument('dataset', help='Name of dataset')
parser.add_argument('save_dir', help='Save path')

parser.add_argument('--folds', '-f', type=int, default=4)

args = parser.parse_args()

X = []
y = []

for dir in os.listdir(args.data_dir):
    for filename in os.listdir(os.path.join(args.data_dir, dir)):
        img = cv.imread(os.path.join(args.data_dir, dir, filename), cv.IMREAD_GRAYSCALE)
        img = img.reshape((*img.shape, 1))

        # Here you can apply any image preprocessing e.g. reducing noise or contrast optimization

        X.append(img)
        y.append(int(dir))

X = np.array(X) / 255
y = np.array(y, dtype=np.float32)

for i in range(args.folds):
    (X_train, X_val, X_test), (y_train, y_val, y_test) = split_data(X, y, train_percent=.4, test_percent=.5, val_percent=.1)
    X_train, y_train = augment_images(X_train, y_train,
                                      rotate_angle=360,
                                      relative_shift=.2,
                                      zoom_factor=.1,
                                      flip=True,
                                      _augment=.8)

    with open(os.path.join(args.save_dir, f'{args.dataset}_{i}.pkl'), 'wb') as f:
        pickle.dump(((X_train, X_val, X_test), (y_train, y_val, y_test)), f)

