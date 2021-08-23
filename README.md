Tested with Python 3.8, Tensorflow 2.5, CUDA 11.2 and cuDNN 8.1. Note that even slightly different environment from ours may lead to errors.

Training command:

python landmark_detection_group1_size24.py

Inference command (this is only an example; specific path to model varies):

python landmark_detection_group1_size24.py --mode=inference --weight-path=./models/3d_faster_rcnn20210810T1758/3d_faster_rcnn.18.h5
