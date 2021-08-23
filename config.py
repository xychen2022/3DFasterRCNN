"""
Mask R-CNN
Base Configurations class.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import numpy as np
from itertools import permutations, combinations_with_replacement

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = '3d_faster_rcnn'  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Data path
    TRAIN_DATA_FOLDER = os.getcwd() # '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/'
    VAL_DATA_FOLDER = os.getcwd() # '/shenlab/lab_stor4/xychen/original_based_heatmap_prediction/updated_data/'

    # Train (Val) image ids
    TRAIN_IDS = ['case_100_ct_patient', 'case_102_ct_patient', 'case_103_ct_patient', 'case_105_ct_patient', 'case_107_ct_patient', 
                 'case_110_ct_patient', 'case_112_ct_patient', 'case_113_ct_patient', 'case_114_ct_patient', 'case_115_ct_patient', 
                 'case_116_ct_patient', 'case_117_ct_patient', 'case_118_ct_normal', 'case_119_ct_normal', 'case_120_ct_normal', 
                 'case_121_ct_normal', 'case_122_ct_normal', 'case_123_ct_normal', 'case_124_ct_normal', 'case_125_ct_normal', 
                 'case_126_ct_normal', 'case_127_ct_normal', 'case_128_ct_normal', 'case_129_ct_normal', 'case_130_ct_normal', 
                 'case_132_ct_normal', 'case_133_ct_normal', 'case_134_ct_normal', 'case_136_ct_normal', 'case_137_ct_normal', 
                 'case_138_ct_normal', 'case_139_ct_normal', 'case_140_ct_normal', 'case_141_ct_normal', 'case_142_ct_normal', 
                 'case_143_ct_normal', 'case_144_ct_normal', 'case_145_ct_normal', 'case_146_ct_normal', 'case_147_ct_normal', 
                 'case_149_ct_normal', 'case_150_ct_normal', 'case_151_ct_normal', 'case_152_ct_normal', 'case_153_ct_normal', 
                 'case_154_ct_normal', 'case_155_ct_normal', 'case_156_ct_normal', 'case_157_ct_normal', 'case_158_ct_normal', 
                 'case_160_ct_normal', 'case_161_ct_normal', 'case_162_ct_normal', 'case_163_ct_normal', 'case_164_ct_normal', 
                 'case_165_ct_normal', 'case_166_ct_normal', 'case_167_ct_normal', 'case_168_ct_normal', 'case_170_ct_normal', 
                 'case_171_ct_normal', 'case_172_ct_normal', 'case_173_ct_normal', 'case_174_ct_normal', 'case_175_ct_normal', 
                 'case_176_ct_normal', 'case_177_ct_normal', 'case_178_ct_normal', 'case_180_ct_normal', 'case_181_ct_normal', 
                 'case_18_cbct_patient', 'case_19_cbct_patient', 'case_20_cbct_patient', 'case_21_cbct_patient', 'case_22_cbct_patient',
                 'case_23_cbct_patient', 'case_24_cbct_patient', 'case_25_cbct_patient', 'case_26_cbct_patient', 'case_27_cbct_patient', 
                 'case_28_cbct_patient', 'case_29_cbct_patient', 'case_30_cbct_patient', 'case_31_cbct_patient', 'case_34_cbct_patient', 
                 'case_35_cbct_patient', 'case_36_cbct_patient', 'case_37_cbct_patient', 'case_39_cbct_patient', 'case_40_cbct_patient', 
                 'case_41_cbct_patient', 'case_42_cbct_patient', 'case_43_cbct_patient', 'case_44_cbct_patient', 'case_46_cbct_patient', 
                 'case_48_cbct_patient', 'case_49_cbct_patient', 'case_50_cbct_patient', 'case_51_cbct_patient', 'case_52_cbct_patient', 
                 'case_56_cbct_patient', 'case_57_cbct_patient', 'case_58_cbct_patient', 'case_59_cbct_patient', 'case_60_cbct_patient', 
                 'case_61_cbct_patient', 'case_62_cbct_patient', 'case_63_cbct_patient', 'case_65_cbct_patient']
    
    VAL_IDS = ['case_101_ct_patient', 'case_111_ct_patient', 'case_131_ct_normal', 'case_135_ct_normal', 'case_148_ct_normal', 
               'case_159_ct_normal', 'case_169_ct_normal', 'case_179_ct_normal', 'case_17_cbct_patient', 'case_32_cbct_patient', 
               'case_38_cbct_patient', 'case_47_cbct_patient', 'case_54_cbct_patient', 'case_64_cbct_patient', 'case_66_cbct_patient']
        
    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 20*len(TRAIN_IDS) #2400

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 20*len(VAL_IDS) #200
    
    EPOCHS = 18

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    BACKBONE = "VGGlike"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # Number of downsampling in the backbone
    NUM_OF_DOWNSAMPLING = 3

    # The strides of each layer of the FPN Pyramid. These values
    # are based on backbone.
    BACKBONE_STRIDES = [4, 8] # [4, 8, 16]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background)
    NUM_CLASSES = 25

    # Length of square anchor side in pixels
    LENGTH = 24 # Should be an even number - I did not take special care to odd number in the algorithm
    RPN_ANCHOR_SCALES = [LENGTH, LENGTH]

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [1.0]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8 # 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256 ## A smaller or a larger value is more suitable ??? ##
    
    RPN_IOU_LOWER = 0.3
    RPN_IOU_HIGHER = 0.5

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 1200
    POST_NMS_ROIS_INFERENCE = 600

    IMAGE_FORMAT = 'channels_last'
    IMAGE_NUM_CHS = 1
    IMAGE_DEPTH = 112 + LENGTH
    IMAGE_HEIGHT = 144 + LENGTH
    IMAGE_WIDTH = 144 + LENGTH

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 120 # Used in 'detection_targets_graph'; positive_roi + negative_roi ##

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.5 # 0.333

    # Pooled ROIs
    POOL_SIZE = 5 # 7

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 24

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 24 ## Same as MAX_GT_INSTANCES ##

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.8

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.4

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.0005
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 5.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 2.,
        "mrcnn_bbox_loss": 1.
    }

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    def __init__(self):

        def anchors_per_location(ratios):
            total = set()
        
            for i in combinations_with_replacement(ratios, 3):
                total |= set(permutations(i))
            
            total = [list(x) for x in sorted(total)]
            return len(total)

        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.IMAGE_SHAPE = np.array([self.IMAGE_DEPTH, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_NUM_CHS])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 1 + 3 + 6 + self.NUM_CLASSES
        self.ANCHORS_PER_LOCATION = anchors_per_location(self.RPN_ANCHOR_RATIOS)
    
        self.display()

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
