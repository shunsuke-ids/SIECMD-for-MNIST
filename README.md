# SIECMD
This repository provides the implementation of the presented solution in: 
[Single Image Estimation of Cell Migration Direction by Deep Circular Regression](https://arxiv.org/abs/2406.19162) (L. Bruns et al.) [preprint manuscript].

## Abstract
In this paper we study the problem of estimating the migration direction of cells based on a single image. To the best of our knowledge, there is only one related work that uses a classification CNN for four classes (quadrants). This approach does not allow detailed directional resolution. We solve the single image estimation problem using deep circular regression with special attention to cycle-sensitive methods. On two databases we achieve an average accuracy of ∼17 degrees, which is a significant improvement over the previous work. 

## Implementation
Following, the implementation of SIECMD is described as directory structure. 

### /src/DL
This directory contains all deep learning based solution parts. 
The evaluation metrics are implemented in [metrics.py](./src/DL/metrics.py). 
The activation and loss functions that support circular regression are implemented in [activation_functions.py](./src/DL/activation_functions.py) and [losses.py](./src/DL/losses.py). The file [models.py](./src/DL/models.py) contains the probing model for the described parameter probing. The file also features a classification model of similar size. 

### /src/preprocessing
This directory features all methods for preprocessing and handling datasets before training. The file [augment.py](./src/preprocessing/augment.py) implements die augmentation process for training data preperation as well as test-time augmentation (TTA). The methods in [handle_dataset.py](./src/preprocessing/handle_dataset.py) can be used to prepare datasets to split into train, test and val set or normal distribute angular representations. 

### /src/regression
This directory contains implementations of SIECMD regression task. The files [fine_tuning.py](./src/regression/fine_tuning.py) and [probing.py](./src/regression/probing.py) are examples on how to train and evaluate the circular regression models. New datasets can be prepared to match the format used in those examples by using [prepare_dataset.py](./src/regression/prepare_dataset.py). The remaining two files contain helper functions, also used in the two example applications. The file [format_gt.py](./src/regression/format_gt.py) supports the conversion between different ground truth encodings. Finally, the file [circular_operations.py](./src/regression/circular_operations.py) contains methods for circular averaging which are used for TTA.

### /weights
This is the default weight-file save directory. Files are saved as *.keras* files and can be loaded to keras models (see example [fine_tuning.py](./src/regression/fine_tuning.py)). 
