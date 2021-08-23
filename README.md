This is the code for our TMI paper Fast and Accurate Craniomaxillofacial Landmark Detection via 3D Faster R-CNN. Only coarse stage code is provided, since the implementation of the model for landmark location refinement is trivial. For more details, please refer

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9494574



Our code is tested with Python 3.8, Tensorflow 2.5, CUDA 11.2 and cuDNN 8.1. Note that even slightly different environment from ours may lead to training/inference errors.

>>> Training command:

python landmark_detection_group1_size24.py

>>> Inference command (this is only an example; specific path to model varies):

python landmark_detection_group1_size24.py --mode=inference --weight-path=./models/3d_faster_rcnn20210810T1758/3d_faster_rcnn.18.h5
