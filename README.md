This is the code for our TMI paper "Fast and Accurate Craniomaxillofacial Landmark Detection via 3D Faster R-CNN". Only coarse stage code is provided, since the implementation of the model for landmark location refinement is trivial. For more details, please refer

https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9494574

Our code is tested with Python 3.8, Tensorflow 2.5, CUDA 11.2 and cuDNN 8.1. To use the default settings, the same environment as ours is required to train/test models. Note that even slightly different environment from ours may lead to training/inference errors. If you want to use them in a different environment, you must recompile the CUDA source codes in folders CropAndResize3D and non_max_suppression to obtain the new crop_and_resize_op_gpu.so and non_max_suppression_op.so! We have provided the instructions in ReadMe.txt inside both CropAndResize3D and non_max_suppression for recompiling. After this, you should either change the paths in landmark_detection_group1_size24.py (lines 48 and 51) or replace the old ones in the current folder with the new ones.

The algorithm is designed to be general. That is, users can use any imaging modalities that are prefered in their applications. To use our code in a different application, an user should prepare the data (i.e., images and ground truth landmark locations) the same way as ours (see folders images_1.6 and coordinates_1.6; the coordinates are in the order of x, y and z) and save them in folders for images and ground truth respectively. Note that users should carefully choose the down-sampling rate, as a too large image spacing may lose a lot of image details that are essential for accurate landmark localization. Depending on the sizes of your data, users should choose the smallest down-sampling rate possible. Please also note that the default hyper-parameters should give some reasonable predictions but they are not guarantteed to be optimal in different datasets and applications. We encourage the users to tune the parameters when using this code.

### Training command:

```python landmark_detection_group1_size24.py```

### Inference command (this is only an example; specific path to model varies):

```python landmark_detection_group1_size24.py --mode=inference --weight-path=./models/3d_faster_rcnn20210810T1758/3d_faster_rcnn.18.h5```


If you find our work useful, please consider citing our paper:

```
@article{chen2021fast,
title={Fast and Accurate Craniomaxillofacial Landmark Detection via 3D Faster R-CNN},
author={Chen, Xiaoyang and Lian, Chunfeng and Deng, Hannah H and Kuang, Tianshu and Lin, Hung-Ying and Xiao, Deqiang and Gateno, Jaime and Shen, Dinggang and Xia, James J and Yap, Pew-Thian},
journal={IEEE Transactions on Medical Imaging},
year={2021},
publisher={IEEE}
}
```
