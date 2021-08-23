#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 15:24:23 2018

@author: xiaoyangchen
"""

import os
import random
import datetime
import re
import math
import h5py
import logging
import SimpleITK as sitk
from collections import OrderedDict
import multiprocessing
import numpy as np

import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

from mrcnn import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import tensorflow as tf

import warnings
warnings.filterwarnings('ignore')

# Requires TensorFlow 2.4.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("2.4")

tf.compat.v1.disable_eager_execution()

nms_opt = tf.load_op_library('./non_max_suppression_op.so') ## cpu version
ops.NotDifferentiable('NonMaxSuppressionV6')

box_opt = tf.load_op_library('./crop_and_resize_op_gpu.so')

@ops.RegisterGradient("CropAndResizeV3")
def _CropAndResizeV3Grad(op, grad):
  """The derivatives for crop_and_resize_v3.
  We back-propagate to the image only when the input image tensor has floating
  point dtype but we always back-propagate to the input boxes tensor.
  Args:
    op: The CropAndResizeV3 op.
    grad: The tensor representing the gradient w.r.t. the output.
  Returns:
    The gradients w.r.t. the input image, boxes, as well as the always-None
    gradients w.r.t. box_ind and crop_size.
  """
  image = op.inputs[0]
  if image.get_shape().is_fully_defined():
    image_shape = image.get_shape().as_list()
  else:
    image_shape = array_ops.shape(image)

  allowed_types = [dtypes.float16, dtypes.float32, dtypes.float64]
  if op.inputs[0].dtype in allowed_types:
    # pylint: disable=protected-access
    grad0 = box_opt.crop_and_resize_v3_grad_image(
        grad, op.inputs[1], op.inputs[2], image_shape, T=op.get_attr("T"),
        method=op.get_attr("method"))
    # pylint: enable=protected-access
  else:
    grad0 = None

  # `grad0` is the gradient to the input image pixels and it
  # has been implemented for trilinear sampling.
  # `grad1` is the gradient to the input crop boxes' coordinates.
  # When using nearest neighbor sampling (not defined in 3D version actually), the gradient to crop boxes'
  # coordinates are not well defined. In practice, we still approximate
  # grad1 using the gradient derived from trilinear sampling.
  grad1 = box_opt.crop_and_resize_v3_grad_boxes(
      grad, op.inputs[0], op.inputs[1], op.inputs[2])

  return [grad0, grad1, None, None]

def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    
    Returns:
        [N, (depth, height, width)]. Where N is the number of stages
    """

    # Currently supports VGGlike only
    assert config.BACKBONE in ["VGGlike"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
            int(math.ceil(image_shape[1] / stride)),
            int(math.ceil(image_shape[2] / stride))]
            for stride in config.BACKBONE_STRIDES])

############################################################
#  Group Normalization Layer
############################################################
class GroupNormalization(KL.Layer):
    
    def __init__(self,
                 groups=1,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
    
    def build(self, input_shape):
        dim = input_shape[self.axis]
        
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
    
        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')
        
        if dim % self.groups != 0:
            raise ValueError('Number of channels must be divisible by the number of groups')
        
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)
        
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        if self.axis == -1 or self.axis == 4:
            inputs = K.permute_dimensions(inputs, [0, 4, 1, 2, 3])
        
        input_shape = K.int_shape(inputs)  
        original_shape = tf.shape(inputs) #[tf.shape(inputs)[0],] + list(input_shape[1:])
        
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[1] = input_shape[1]

        reshape_group_shape = [tf.shape(inputs)[i] for i in range(len(input_shape))]
        reshape_group_shape[1] = tf.shape(inputs)[1] // self.groups
        group_shape = [tf.shape(inputs)[0], self.groups]
        group_shape.extend(reshape_group_shape[1:])
        
        inputs = K.reshape(inputs, group_shape)
        
        mean = K.mean(inputs, axis=[2, 3, 4, 5], keepdims=True)
        variance = K.var(inputs, axis=[2, 3, 4, 5], keepdims=True)
        
        outputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))
        outputs = K.reshape(outputs, original_shape)
            
        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma
            
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        if self.axis == -1 or self.axis == 4:
            outputs = K.permute_dimensions(outputs, [0, 2, 3, 4, 1])
        
        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
            }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape

class AnchorsLayer(KL.Layer):
    def __init__(self, name="anchors", **kwargs):
        super(AnchorsLayer, self).__init__(name=name, **kwargs)
    
    def call(self, anchor):
        return anchor
    
    def get_config(self) :
        config = super(AnchorsLayer, self).get_config()
        return config

############################################################
#  Backbone
############################################################

class conv_block:
    def __init__(self, in_channels, inter_channels, out_channels, name='cb', train_bn=False, **kwargs):
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.name = name
        self.train_bn = train_bn
        
    def __call__(self, input_):
        x = KL.Conv3D(self.inter_channels, (3, 3, 3), padding='same', data_format='channels_last', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv1')(input_)
        x = GroupNormalization(groups=self.inter_channels, axis=-1, name=self.name + '_gn1')(x)
        # x = KL.BatchNormalization(axis=-1, name=self.name + '_bn1')(x, training=self.train_bn)
        x = KL.Activation('relu', name=self.name + '_acti1')(x)

        x = KL.Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_last', dilation_rate=(2, 2, 2), kernel_initializer='he_normal', name=self.name + '_conv2')(x)
        x = GroupNormalization(groups=self.out_channels, axis=-1, name=self.name + '_gn2')(x)
        # x = KL.BatchNormalization(axis=-1, name=self.name + '_bn2')(x, training=self.train_bn)
        x = KL.Activation('relu', name=self.name + '_acti2')(x)
        return x

class res_block:
    def __init__(self, in_channels, out_channels, name='cb', train_bn=False, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.train_bn = train_bn

    def __call__(self, input_):

        x = KL.Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_last',
                      kernel_initializer='he_normal', name=self.name + '_conv1')(input_)
        x = GroupNormalization(groups=self.out_channels, axis=-1, name=self.name + '_gn1')(x)
        # x = KL.BatchNormalization(axis=-1, name=self.name + '_bn1')(x, training=self.train_bn)

        if self.in_channels == self.out_channels:
            x = KL.Activation('relu', name=self.name + '_acti')(x)
            x = KL.Add(name=self.name + '_add')([input_, x])

        else:
            y = KL.Conv3D(self.out_channels, (3, 3, 3), padding='same', data_format='channels_last',
                          kernel_initializer='he_normal', name=self.name + '_conv2')(input_)
            y = GroupNormalization(groups=self.out_channels, axis=-1, name=self.name + '_gn2')(y)
            # y = KL.BatchNormalization(axis=-1, name=self.name + '_bn2')(y, training=self.train_bn)
            x = KL.Add(name=self.name + '_add')([y, x])
            x = KL.Activation('relu', name=self.name + '_acti')(x)
        return x

def Backbone(img_input, train_bn):
    """Backbone"""
    
    conv0 = KL.Conv3D(16, (3, 3, 3), padding='same', data_format='channels_last', kernel_initializer='he_normal', name='conv0')(img_input)
    conv0 = GroupNormalization(groups=16, axis=-1, name='gn0')(conv0)
    # conv0 = KL.BatchNormalization(axis=-1, name='bn0')(conv0, training=train_bn)
    conv0 = KL.Activation('relu', name='acti0')(conv0)
    
    pool1 = KL.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(conv0)
    C1 = res_block(16, 32, name='cb1', train_bn=train_bn)(pool1)

    conv2 = res_block(32, 64, name='cb2', train_bn=train_bn)(C1)
    conv2 = res_block(64, 128, name='cb3', train_bn=train_bn)(conv2)
    conv2 = res_block(128, 256, name='cb4', train_bn=train_bn)(conv2)
    #conv2 = res_block(256, 256, name='cb5', train_bn=train_bn)(conv2)
    pool2 = KL.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(conv2)
    C2 = res_block(256, 256, name='cb6', train_bn=train_bn)(pool2)

    conv3 = res_block(256, 256, name='cb7', train_bn=train_bn)(C2)
    C3 = KL.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), data_format='channels_last')(conv3)
    
    return [C1, C2, C3]

############################################################
#  Region Proposal Network (RPN)
############################################################

def rpn_graph(feature_map, anchors_per_location, anchor_stride):
    """Builds the computation graph of Region Proposal Network.

    feature_map: backbone features [batch, channels, depth, height, width] if data_format = 'channels_first'
    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).

    Returns:
        rpn_logits: [batch, Depth * Height * Width * anchors_per_location, 2]             Anchor classifier logits (before softmax)
        rpn_probs:  [batch, Depth * Height * Width * anchors_per_location, 2]             Anchor classifier probabilities.
        rpn_bbox:   [batch, Depth * Height * Width * anchors_per_location, (dz, dy, dx)]  Deltas to be applied to anchors.
    """
    
    # Shared convolutional base of the RPN
    shared = KL.Conv3D(256, (3, 3, 3), padding='same', strides=anchor_stride, data_format='channels_last', name='rpn_conv_shared')(feature_map)
    # shared = GroupNormalization(groups=256, axis=-1, name='rpn_conv_shared_gn')(shared)
    # shared = KL.BatchNormalization(axis=-1, name='rpn_conv_shared_bn')(shared, training=False)
    shared = KL.Activation('relu', name='rpn_conv_shared_acti')(shared)
    
    # Anchor Score. 
    # Because I use data_format = 'channels_last', the shape of x is [batch, depth, height, width, anchors per location * 2]
    x = KL.Conv3D(2 * anchors_per_location, (1, 1, 1), padding='valid', activation='linear', data_format='channels_last', name='rpn_class_raw')(shared)
    rpn_class_logits = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)
    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation("softmax", name="rpn_anchor_classification")(rpn_class_logits)

    # Bounding box refinement.
    x = KL.Conv3D(anchors_per_location * 3, (1, 1, 1), padding="valid", activation='linear', data_format='channels_last', name='rpn_bbox_pred')(shared)
    # Reshape to [batch, anchors, 3], the last dimension is [x, y, z]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 3]))(x) # HAVING BEEN TESTED. This is okay to reshape a placeholder

    return [rpn_class_logits, rpn_probs, rpn_bbox]

def build_rpn_model(anchor_stride, anchors_per_location, depth):
    """Builds a Keras model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    anchors_per_location: number of anchors per pixel in the feature map
    anchor_stride: Controls the density of anchors. Typically 1 (anchors for
                   every pixel in the feature map), or 2 (every other pixel).
    depth: Depth of the backbone feature map.

    Returns a Keras Model object. The model outputs, when called, are:
    rpn_logits: [batch, Depth * Height * Width * anchors_per_location, 2]             Anchor classifier logits (before softmax)
    rpn_probs:  [batch, Depth * Height * Width * anchors_per_location, 2]             Anchor classifier probabilities.
    rpn_bbox:   [batch, Depth * Height * Width * anchors_per_location, (dz, dy, dx)]  Deltas to be applied to anchors.
    """
    input_feature_map = KL.Input(shape=[None, None, None, depth], name="input_rpn_feature_map")
    outputs = rpn_graph(input_feature_map, anchors_per_location, anchor_stride)
    return KM.Model([input_feature_map], outputs, name="rpn_model")

############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas, Size=24):
    """Applies the given deltas to the given boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] boxes to update
    deltas: [N, (dz, dy, dx)] refinements to apply
    """
    
    # center_z, center_y, center_x are the (normalized) coordinates of the centers
    center_z = 0.5 * (boxes[:, 0] +  boxes[:, 3])
    center_y = 0.5 * (boxes[:, 1] +  boxes[:, 4])
    center_x = 0.5 * (boxes[:, 2] +  boxes[:, 5])
    
    # Apply deltas
    center_z += deltas[:, 0]
    center_y += deltas[:, 1]
    center_x += deltas[:, 2]

    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * Size
    y1 = center_y - 0.5 * Size
    x1 = center_x - 0.5 * Size
    z2 = center_z + 0.5 * Size
    y2 = center_y + 0.5 * Size
    x2 = center_x + 0.5 * Size
    result = tf.stack([z1, y1, x1, z2, y2, x2], axis=1, name="apply_box_deltas_out")
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (z1, y1, x1, z2, y2, x2)]
    window: [1, 6] in the form z1, y1, x1, z2, y2, x2
    """
    # Split
    wz1, wy1, wx1, wz2, wy2, wx2 = tf.split(window, 6, axis=1)
    z1, y1, x1, z2, y2, x2 = tf.split(boxes, 6, axis=1)
    # Clip
    z1 = tf.maximum(tf.minimum(z1, wz2), wz1)
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    z2 = tf.maximum(tf.minimum(z2, wz2), wz1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([z1, y1, x1, z2, y2, x2], axis=1, name="clipped_boxes")

    clipped.set_shape((clipped.shape[0], 6))
    return clipped

class ProposalLayer(KL.Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, anchors, (dz, dy, dx)]
        anchors: [batch, (z1, y1, x1, z2, y2, x2)]

    Returns:
        Proposals: [batch, rois, (z1, y1, x1, z2, y2, x2)]
    """

    def __init__(self, proposal_count, nms_threshold, config=None, **kwargs):
        super(ProposalLayer, self).__init__(**kwargs)
        self.config = config
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold

    def call(self, inputs):
        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]
        # Box deltas [batch, num_rois, 3]
        deltas = inputs[1]
        # Anchors
        anchors = inputs[2]
        # Image meta
        image_meta = inputs[3]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(6000, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name="top_anchors").indices
        scores = utils.batch_slice([scores, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        deltas = utils.batch_slice([deltas, ix], lambda x, y: tf.gather(x, y),
                                   self.config.IMAGES_PER_GPU)
        pre_nms_anchors = utils.batch_slice([anchors, ix], lambda a, x: tf.gather(a, x),
                                    self.config.IMAGES_PER_GPU,
                                    names=["pre_nms_anchors"])

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (z1, y1, x1, z2, y2, x2)]
        boxes = utils.batch_slice([tf.cast(pre_nms_anchors, tf.float32), deltas],
                                  lambda x, y: apply_box_deltas_graph(x, y, Size=self.config.LENGTH),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors"])

        # Clip to image boundaries. Note coordiantes are not normalized
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape']
        window = norm_boxes_graph(m['window'], image_shape[:3])
        #print('boxes in ProposalLayer: ', boxes)
        #print('window in ProposalLayer: ', window)
        boxes = utils.batch_slice(boxes,
                                  lambda x: clip_boxes_graph(x, window),
                                  self.config.IMAGES_PER_GPU,
                                  names=["refined_anchors_clipped"])

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.

        # Non-max suppression
        def nms(boxes, scores):
            indices = nms_opt.non_max_suppression_v6(
                boxes, scores, self.proposal_count,
                self.nms_threshold, 0., name="rpn_non_max_suppression") # Changed here # 
            proposals = tf.gather(boxes, indices)
            # Pad if needed
            padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
            proposals = tf.pad(proposals, [(0, padding), (0, 0)])
            return proposals

        proposals = utils.batch_slice([boxes, scores], nms, self.config.IMAGES_PER_GPU)
        
        return proposals

    def compute_output_shape(self, input_shape):
        return (None, self.proposal_count, 6)

############################################################
#  Detection Target Layer
############################################################

def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].
    """
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    #b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1), [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b1 = tf.reshape(tf.tile(boxes1, [1, tf.shape(boxes2)[0]]), [-1, 6])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_z1, b1_y1, b1_x1, b1_z2, b1_y2, b1_x2 = tf.split(b1, 6, axis=1)
    b2_z1, b2_y1, b2_x1, b2_z2, b2_y2, b2_x2 = tf.split(b2, 6, axis=1)
    z1 = tf.maximum(b1_z1, b2_z1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    z2 = tf.minimum(b1_z2, b2_z2)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(z2 - z1, 0) * tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_volume = tf.maximum(b1_z2 - b1_z1, 0) * tf.maximum(b1_y2 - b1_y1, 0) * tf.maximum(b1_x2 - b1_x1, 0)
    b2_volume = tf.maximum(b2_z2 - b2_z1, 0) * tf.maximum(b2_y2 - b2_y1, 0) * tf.maximum(b2_x2 - b2_x1, 0)
    union = b1_volume + b2_volume - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = (intersection + 1e-8) / (union + 1e-6)
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps

def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
        box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)
    
    gt_center_z = 0.5 * (gt_box[:, 0] + gt_box[:, 3])
    gt_center_y = 0.5 * (gt_box[:, 1] + gt_box[:, 4])
    gt_center_x = 0.5 * (gt_box[:, 2] + gt_box[:, 5])

    center_z = 0.5 * (box[:, 0] + box[:, 3])
    center_y = 0.5 * (box[:, 1] + box[:, 4])
    center_x = 0.5 * (box[:, 2] + box[:, 5])
    
    dz = gt_center_z - center_z
    dy = gt_center_y - center_y
    dx = gt_center_x - center_x

    result = tf.stack([dz, dy, dx], axis=1)
    return result

def detection_targets_graph(proposals, gt_class_ids, gt_boxes, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates. Might be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)] in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts.
        rois:      [TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)]     in normalized coordinates
        class_ids: [TRAIN_ROIS_PER_IMAGE].                              Integer class IDs. Zero padded.
        deltas:    [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dz, dy, dx)]    Class-specific bbox refinements.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros, name="trim_gt_class_ids")
    
    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)
    
    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.where(roi_iou_max < 0.5)[:, 0]

    # Subsample ROIs. Aim for 33% positive
    # Positive ROIs
    positive_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
    positive_count = tf.shape(positive_indices)[0]
    # Negative ROIs. Add enough to maintain positive:negative ratio.
    r = 1.0 / config.ROI_POSITIVE_RATIO
    negative_count = tf.cast(r * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    negative_indices = tf.random.shuffle(negative_indices)[:negative_count]
    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        tf.greater(tf.shape(positive_overlaps)[1], 0),
        true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
        false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    # print('positive_rois: ', positive_rois)
    # print('roi_gt_boxes: ', roi_gt_boxes)
    deltas = box_refinement_graph(positive_rois, roi_gt_boxes)

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(negative_rois)[0]
    P = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(rois)[0], 0)
    rois = tf.pad(rois, [(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
    roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
    deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
    
    return rois, roi_gt_class_ids, deltas


class DetectionTargetLayer(KL.Layer):
    """Subsamples proposals and generates target box refinement, class_ids

    Inputs:
    proposals:    [batch, N, (z1, y1, x1, z2, y2, x2)]                  in normalized coordinates. Might be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES]                             Integer class IDs.
    gt_boxes:     [batch, MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)]   in normalized coordinates.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts
        rois:             [batch, TRAIN_ROIS_PER_IMAGE, (z1, y1, x1, z2, y2, x2)]   in normalized coordinates
        target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]                             Integer class IDs.
        target_deltas:    [batch, TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dz, dy, dx)]  Class-specific bbox refinements.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(DetectionTargetLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        
        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois", "target_class_ids", "target_bbox"]
        outputs = utils.batch_slice(
            [proposals, gt_class_ids, gt_boxes],
            lambda w, x, y: detection_targets_graph(
                w, x, y, self.config),
            self.config.IMAGES_PER_GPU, names=names)
        
        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 6),  # rois
            (None, self.config.TRAIN_ROIS_PER_IMAGE),     # class_ids
            (None, self.config.TRAIN_ROIS_PER_IMAGE, 3),  # deltas
        ]

############################################################
#  ROIAlign Layer
############################################################

def log2_graph(x):
    """Implementation of Log2. TF doesn't have a native implementation."""
    return tf.math.log(x) / tf.math.log(2.0)


class PyramidROIAlign(KL.Layer):
    """Implements ROI Alignment on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [depth, height, width] of the output pooled regions. Usually [7, 7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - Feature maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, depth, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, depth, height, width, channels].
    The depth height and width are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape, **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)

    def call(self, inputs):
        # Crop boxes [batch, num_boxes, (z1, y1, x1, z2, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, depth, height, width, channels]
        feature_maps = inputs[2:]

        # Convert boxes from real coordinates to normalized coordinates <= required by box_opt.crop_and_resize_v3
        m = parse_image_meta_graph(image_meta)
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = m['image_shape'][0]
        boxes = boxes / tf.cast(tf.tile(image_shape, [2]), tf.float32)
        window = tf.cast(tf.identity([[0., 0., 0., 1., 1., 1.]]), tf.float32)

        # Clip the values in case when the values are less than 0 or more than 1.
        # Split
        wz1, wy1, wx1, wz2, wy2, wx2 = tf.split(window, 6, axis=1)
        z1, y1, x1, z2, y2, x2 = tf.split(boxes, 6, axis=2)
        # Clip
        z1 = tf.maximum(tf.minimum(z1, wz2), wz1)
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        z2 = tf.maximum(tf.minimum(z2, wz2), wz1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        boxes = tf.concat([z1, y1, x1, z2, y2, x2], axis=-1, name="clipped_boxes")
        
        # Assign each ROI to a level in the pyramid based on the ROI volume.
        z1, y1, x1, z2, y2, x2 = tf.split(boxes, 6, axis=2)
        d = z2 - z1
        h = y2 - y1
        w = x2 - x1
        d = tf.squeeze(d, 2)
        h = tf.squeeze(h, 2)
        w = tf.squeeze(w, 2)

        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        image_volume = tf.cast(image_shape[0] * image_shape[1] * image_shape[2], tf.float32)
        roi_level = log2_graph(tf.pow(d * h * w, 1.0/3) / (2000.0 / tf.pow(image_volume, 1.0/3)))
        # roi_level = tf.minimum(3, tf.maximum(3, 3 + tf.cast(tf.round(roi_level), tf.int32))) # Original on Level 3
        roi_level = tf.minimum(2, tf.maximum(2, 2 + tf.cast(tf.round(roi_level), tf.int32))) # Try on Level 2

        # Loop through levels and apply ROI pooling to each. P2 to P3.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 4)):
            ix = tf.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)
            
            # Box indices for crop_and_resize_v3.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(box_opt.crop_and_resize_v3(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="trilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.

        # WELL DONE!!
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)
        
        # Re-add the batch dimension
        # pooled = tf.expand_dims(pooled, 0)
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        
        return pooled

    def compute_output_shape(self, input_shape):
        return input_shape[0][:2] + self.pool_shape + (input_shape[2][-1], )

def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, train_bn=True,
                         fc_layers_size=128):
    """Builds the computation graph of the feature pyramid network classifier
        and regressor heads.
        
        rois: [batch, num_rois, (z1, y1, x1, z2, y2, x2)] Proposal boxes in normalized
              coordinates.
        feature_maps: List of feature maps from diffent layers of the pyramid,
                      [P2, P3, P4, P5]. Each has a different resolution.
        pool_size: The width of the square feature map generated from ROI Pooling.
        num_classes: number of classes, which determines the depth of the results; whose value is (1 + 2)
        train_bn: Boolean. Train or freeze Batch Norm layres
        
        Returns:
        logits:      [batch, num_rois, NUM_CLASSES]                classifier logits (before softmax); N is the number of 
        probs:       [batch, num_rois, NUM_CLASSES]                classifier probabilities
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dz, dy, dx)]  Deltas to apply to proposal boxes
    """
    # ROI Pooling
    # Shape: [batch, num_boxes, pool_depth, pool_height, pool_width, channels]
    x = PyramidROIAlign([pool_size, pool_size, pool_size], name="roi_align_classifier")([rois, image_meta] + feature_maps)
    
    # Two 1024 FC layers (implemented with Conv3D for consistency)
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (pool_size, pool_size, pool_size), padding="valid"), name="mrcnn_class_conv1")(x)
    # x = KL.TimeDistributed(KL.BatchNormalization(axis=-1), name='mrcnn_class_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    
    x = KL.TimeDistributed(KL.Conv3D(fc_layers_size, (1, 1, 1)), name="mrcnn_class_conv2")(x)
    # x = KL.TimeDistributed(KL.BatchNormalization(axis=-1), name='mrcnn_class_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # end #

    shared = KL.Lambda(lambda x: K.squeeze(K.squeeze(K.squeeze(x, 4), 3), 2), name="pool_squeeze")(x)
    
    # Classifier head
    mrcnn_class_logits = KL.TimeDistributed(KL.Dense(num_classes), name='mrcnn_class_logits')(shared)
    mrcnn_probs = KL.TimeDistributed(KL.Activation("softmax"), name="mrcnn_class")(mrcnn_class_logits)

    # BBox head
    # [batch, boxes, num_classes * (dz, dy, dx)]
    x = KL.TimeDistributed(KL.Dense(num_classes * 3, activation='linear'), name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, boxes, num_classes, (dz, dy, dx)]
    #mrcnn_bbox = KL.Reshape((tf.shape(x)[1], num_classes, 3), name="mrcnn_bbox")(x)
    mrcnn_bbox = KL.Lambda(lambda x: tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], num_classes, 3)), name="mrcnn_bbox")(x)
    
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox

############################################################
#  Detection Layer
############################################################

def apply_box_deltas_refine_graph(boxes, deltas, Size=24):
    """Applies the given deltas to the given boxes.
        boxes: [N, (z1, y1, x1, z2, y2, x2)] boxes to update
        deltas: [N, (dz, dy, dx)] refinements to apply
    """
    # center_z, center_y, center_x are the coordinates of the centers
    center_z = 0.5 * (boxes[:, 0] +  boxes[:, 3])
    center_y = 0.5 * (boxes[:, 1] +  boxes[:, 4])
    center_x = 0.5 * (boxes[:, 2] +  boxes[:, 5])
    
    # Apply deltas
    center_z += deltas[:, 0]
    center_y += deltas[:, 1]
    center_x += deltas[:, 2]
    
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * Size
    y1 = center_y - 0.5 * Size
    x1 = center_x - 0.5 * Size
    z2 = center_z + 0.5 * Size
    y2 = center_y + 0.5 * Size
    x2 = center_x + 0.5 * Size
    result = tf.stack([z1, y1, x1, z2, y2, x2], axis=1, name="apply_box_deltas_out")
    return result

def refine_detections_graph(rois, probs, deltas, window, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dz, dy, dx)]. Class-specific bounding box deltas.
        window: (z1, y1, x1, z2, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (z1, y1, x1, z2, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)
    # Class probability of the top class of each ROI
    indices = tf.stack([tf.range(tf.shape(probs)[0]), class_ids], axis=1)
    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    refined_rois = apply_box_deltas_refine_graph(rois, deltas_specific, Size=config.LENGTH)
#    # Clip boxes to image window
#    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = nms_opt.non_max_suppression_v6(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=config.DETECTION_MAX_INSTANCES,
                iou_threshold=config.DETECTION_NMS_THRESHOLD, score_threshold=0.)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return detections


class DetectionLayer(KL.Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        rois = inputs[0]
        mrcnn_class = inputs[1]
        mrcnn_bbox = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:3])
        window = tf.expand_dims(window, axis=1) # TODO: change this line using clever way
        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, mrcnn_class, mrcnn_bbox, window],
            lambda x, y, w, z: refine_detections_graph(x, y, w, z, self.config),
            self.config.IMAGES_PER_GPU)

        # Reshape output
        # [batch, num_detections, (z1, y1, x1, z2, y2, x2, class_score)] in
        # normalized coordinates
        return tf.reshape(
            detections_batch,
            [self.config.BATCH_SIZE, self.config.DETECTION_MAX_INSTANCES, 8])

    def compute_output_shape(self, input_shape):
        return (None, self.config.DETECTION_MAX_INSTANCES, 8)

#######################
#      Generator      #
#######################

def load_image_gt(config, data_path, image_id, side_length = 24, augmentation=False, max_num_of_instances=18):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [depth, height, width, 1]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (z1, y1, x1, z2, y2, x2)]
    mask: [depth, height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = sitk.ReadImage(data_path + '/images_1.6/{0}.nii.gz'.format(image_id))
    image = sitk.GetArrayFromImage(image)
    
    with h5py.File(data_path + '/coordinates_1.6/{0}.hdf5'.format(image_id), 'r') as f:
        lmks = f['landmark_i'][()][:, ::-1] # - 1.0
    
    num_of_lmks_of_interest = lmks.shape[0]
    assert num_of_lmks_of_interest == 24
    
    z_max, y_max, x_max = image.shape
    
    cropStart = []
    if z_max > 40:
        z_range_min = random.randint(0, int(z_max*0.2)) # Difference
        z_range_max = random.randint(int(z_max*0.5), z_max)
        if z_range_max > z_range_min + config.IMAGE_DEPTH - config.LENGTH:
            z_range_max = z_range_min + config.IMAGE_DEPTH - config.LENGTH
        cropStart.append(z_range_min)
        z_range = slice(z_range_min, z_range_max)
        image = image[z_range]
    else:
        cropStart.append(0)
        image = image

    if y_max > 60:
        y_range_min = random.randint(0, int(y_max*0.2))
        y_range_max = random.randint(int(y_max*0.5), y_max)
        if y_range_max > y_range_min + config.IMAGE_HEIGHT - config.LENGTH:
            y_range_max = y_range_min + config.IMAGE_HEIGHT - config.LENGTH
        cropStart.append(y_range_min)
        y_range = slice(y_range_min, y_range_max)
        image = image[:, y_range]
    else:
        cropStart.append(0)
        image = image

    if x_max > 60:  # Difference
        x_range_min = random.randint(0, int(x_max*0.2))
        x_range_max = random.randint(int(x_max*0.8), x_max)
        if x_range_max > x_range_min + config.IMAGE_WIDTH - config.LENGTH:
            x_range_max = x_range_min + config.IMAGE_WIDTH - config.LENGTH
        cropStart.append(x_range_min)
        x_range = slice(x_range_min, x_range_max)
        image = image[:, :, x_range]
    else:
        cropStart.append(0)
        image = image

    cropStart = np.array(cropStart)
    lmks = lmks - cropStart

    image_size = np.array(image.shape)

    Expand_to = np.array([config.IMAGE_DEPTH - config.LENGTH, config.IMAGE_HEIGHT - config.LENGTH, config.IMAGE_WIDTH - config.LENGTH])
    residual = Expand_to - image_size

    pad_front = np.array([np.random.randint(0, cor+1) + side_length//2 for cor in residual]) # Option3

    pad_front = list(pad_front.astype(np.int32)) # pad_front is a list now
    padded_image = np.zeros(Expand_to + np.array([side_length, side_length, side_length]), dtype=np.float32)
    padded_image[pad_front[0]:(pad_front[0]+image_size[0]),
                 pad_front[1]:(pad_front[1]+image_size[1]),
                 pad_front[2]:(pad_front[2]+image_size[2])] = image
    image = padded_image

    lmks = lmks + pad_front
    
    present = []
    numLandmarks = lmks.shape[0]
    
    for idx in range(numLandmarks):
        landmark_i = lmks[idx]
        if np.all(np.logical_and(landmark_i > pad_front, landmark_i < pad_front + image_size)):
            present.append(1)
        else:
            present.append(0)
    
    present = np.array(present)
    lmks = lmks[np.where(present==1)]
    landmark_ids = np.arange(1, numLandmarks+1)[np.where(present==1)]
    
    img_shape = np.array(image.shape)
    lmks = lmks.astype(np.float32)
    num_of_lmks = lmks.shape[0]
    
    # Data augmentation: to remove some local regions around selected landmarks
    mask = np.random.randint(2, size=num_of_lmks)    
    loc_for_deletion = lmks[np.where(mask==1)].astype(np.int32)

    mask = np.zeros_like(image, dtype=np.int32)
    for idx in range(loc_for_deletion.shape[0]):
        
        center_block_i = loc_for_deletion[idx]

        block_size = np.array([np.random.randint(6, 12) for i in range(3)])
        block_size = block_size - np.mod(block_size, 2)

        if int(center_block_i[0]-block_size[0]/2) < 0:
            z_lower_bound = 0
        else:
            z_lower_bound = int(center_block_i[0]-block_size[0]/2)
        if int(center_block_i[0]+block_size[0]/2) >= img_shape[0]:
            z_upper_bound = None
        else:
            z_upper_bound = int(center_block_i[0]+block_size[0]/2)
        if int(center_block_i[1]-block_size[1]/2) < 0:
            y_lower_bound = 0
        else:
            y_lower_bound = int(center_block_i[1]-block_size[1]/2)
        if int(center_block_i[1]+block_size[1]/2) >= img_shape[1]:
            y_upper_bound = None
        else:
            y_upper_bound = int(center_block_i[1]+block_size[1]/2)
        if int(center_block_i[2]-block_size[2]/2) < 0:
            x_lower_bound = 0
        else:
            x_lower_bound = int(center_block_i[2]-block_size[2]/2)
        if int(center_block_i[2]+block_size[2]/2) >= img_shape[2]:
            x_upper_bound = None
        else:
            x_upper_bound = int(center_block_i[2]+block_size[2]/2)

        mask[z_lower_bound:z_upper_bound,
             y_lower_bound:y_upper_bound,
             x_lower_bound:x_upper_bound] = 1

    # image
    image = image * (1 - mask)
    
    del_or_not = np.zeros([num_of_lmks,], dtype=np.int32)
    for idx in range(num_of_lmks):
        if mask[int(lmks[idx][0])][int(lmks[idx][1])][int(lmks[idx][2])] == 1:
            del_or_not[idx] = 1
    
    lmks = lmks[np.where(del_or_not==0)]
    num_of_lmks = lmks.shape[0]
    landmark_ids = landmark_ids[np.where(del_or_not==0)]

    # bboxes
    bboxes = np.concatenate([lmks - side_length//2, lmks + side_length//2], axis=1) # gt_boxes: 24 * 24 * 24

    # image_meta
    active_class_ids = np.zeros([num_of_lmks_of_interest + 1,], dtype=np.int32)
    active_class_ids[0] = 1
    active_class_ids[landmark_ids] = 1

    # Image meta data
    img_size = image.shape
    image_meta = compose_image_meta( image_id.split('_')[1], img_size, [0, 0, 0, img_size[0], img_size[1], img_size[2]], active_class_ids)

    return image, image_meta, landmark_ids, bboxes

def build_rpn_targets(anchors, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    ### PARAMETERS THAT NEED TO BE CAREFULLY CHOSEN INCLUDE "lower threshold" 
        and "higher threshold" ###

    anchors: [num_anchors, (z1, y1, x1, z2, y2, x2)]
    gt_boxes: [num_gt_boxes, (z1, y1, x1, z2, y2, x2)]

    Returns:
    rpn_match: [N] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [N, (dz, dy, dx)] Anchor bbox deltas.
    """
    
    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dz, dy, dx)]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 3))

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]

    rpn_match[anchor_iou_max < config.RPN_IOU_LOWER] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # TODO: If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argmax(overlaps, axis=0)
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= config.RPN_IOU_HIGHER] = 1

    unique, counts = np.unique(rpn_match, return_counts=True)
    unique, counts = np.unique(anchor_iou_argmax[np.where(rpn_match == 1)], return_counts=True)

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    
    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here

    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_center_z = 0.5 * (gt[0] + gt[3])
        gt_center_y = 0.5 * (gt[1] + gt[4])
        gt_center_x = 0.5 * (gt[2] + gt[5])
        # Anchor
        a_center_z = 0.5 * (a[0] + a[3])
        a_center_y = 0.5 * (a[1] + a[4])
        a_center_x = 0.5 * (a[2] + a[5])

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            gt_center_z - a_center_z,
            gt_center_y - a_center_y,
            gt_center_x - a_center_x,
        ]
        ix += 1
    
    return rpn_match, rpn_bbox

def data_generator(data_path, image_ids, config, shuffle=True, augmentation=False,
                   random_rois=0, batch_size=1, detection_targets=False):
    """
    1. random_rois: If > 0 then generate proposals to be used to train the
    network classifier and mask heads. Useful if training
    the Mask RCNN part without the RPN.
    2. detection_targets: If True, generate detection targets (class IDs, bbox
    deltas, and masks). Typically for debugging or visualizations because
    in trainig detection targets are generated by DetectionTargetLayer.
    
    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dz, dy, dx)] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (z1, y1, x1, z2, y2, x2)]

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1

    # Anchors: [anchor_count, (z1, y1, x1, z2, y2, x2)]
    backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            image, image_meta, gt_class_ids, gt_boxes = \
                load_image_gt(config, data_path, image_id, side_length = config.LENGTH, augmentation=augmentation, max_num_of_instances=config.MAX_GT_INSTANCES) # Note gt_boxes is not normalized

            image = np.expand_dims(image, axis=-1)
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(anchors, gt_boxes, config)
            
            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 3], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 6), dtype=np.int32)
            
            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = image
            assert gt_class_ids.shape[0] == gt_boxes.shape[0]
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            b += 1
            
            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox, batch_gt_class_ids, batch_gt_boxes]
                outputs = []

                yield inputs, outputs
                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise

############################################################
#  Loss Functions
############################################################

def custom_sparse_categorical_crossentropy(gt, pred, from_logits=True, config=None):
    if config is not None:
        one_hot_gt = tf.one_hot(gt, config.NUM_CLASSES, dtype=tf.float32)
    else:
        one_hot_gt = tf.one_hot(gt, 2, dtype=tf.float32)
    
    if from_logits:
        pred_softmax = tf.nn.softmax(pred)
    else:
        pred_softmax = pred
    
    # manual computation of crossentropy
    epsilon = 1e-6
    pred_softmax = tf.clip_by_value(pred_softmax, epsilon, 1. - epsilon)
    return - tf.reduce_mean(one_hot_gt * tf.math.log(pred_softmax) + (1. - one_hot_gt) * tf.math.log(1. - pred_softmax), axis=-1)

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 6], but could be any shape.
    """
    diff = K.abs(y_true - y_pred)
    less_than_one = K.cast(K.less(diff, 1.0), "float32")
    loss = (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)
    return loss

def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = K.cast(K.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(K.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    zeros_logits = tf.gather_nd(rpn_class_logits, tf.where(K.equal(anchor_class, 0)))
    ones_logits = tf.gather_nd(rpn_class_logits, tf.where(K.equal(anchor_class, 1)))
    zeros = tf.gather_nd(anchor_class, tf.where(K.equal(anchor_class, 0)))
    ones = tf.gather_nd(anchor_class, tf.where(K.equal(anchor_class, 1)))

    loss1 = K.sparse_categorical_crossentropy(target=zeros,
                                              output=zeros_logits,
                                              from_logits=True)

    loss2 = K.sparse_categorical_crossentropy(target=ones,
                                              output=ones_logits,
                                              from_logits=True)

    losses = []

    # This is a work around to compute class-specific losses; the fact that computed losses will never be less than 0 are utilized.
    losses.append(K.switch(tf.logical_and(tf.size(loss1) > 0, tf.size(loss2) > 0),
                           K.mean(tf.stack([K.mean(loss1), K.mean(loss2)])), tf.constant(-1, dtype=tf.float32)))
    losses.append(K.switch(tf.logical_and(tf.size(loss1) > 0, tf.equal(tf.size(loss2), 0)), K.mean(loss1),
                           tf.constant(-1, dtype=tf.float32)))
    losses.append(K.switch(tf.logical_and(tf.equal(tf.size(loss1), 0), tf.size(loss2) > 0), K.mean(loss2),
                           tf.constant(-1, dtype=tf.float32)))

    losses = tf.stack(losses)
    losses = tf.gather(losses, tf.where(tf.greater_equal(losses, 0.)))

    return tf.reduce_mean(losses)


def rpn_bbox_loss_graph(config, target_bbox, rpn_match, rpn_bbox):
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = K.squeeze(rpn_match, -1)
    indices = tf.where(K.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss. Negative and neutral anchors do not contribute to the loss.
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = K.sum(K.cast(K.equal(rpn_match, 1), tf.int32), axis=1)
    target_bbox = batch_pack_graph(target_bbox, batch_counts, config.IMAGES_PER_GPU)

    loss = K.abs(target_bbox - rpn_bbox)
    loss = K.switch(tf.size(loss) > 0, K.mean(K.mean(loss, axis=-1)), tf.constant(0.0))
    
    return loss

def mrcnn_class_loss_graph(config, target_class_ids, pred_class_logits, active_class_ids):
    target_class_ids = tf.cast(target_class_ids, 'int64')
    uniqueValues, _, counts = tf.unique_with_counts(tf.reshape(target_class_ids, [-1]))

    losses = []
    for idx in range(config.NUM_CLASSES):
        count = tf.gather_nd(counts, tf.where(tf.equal(uniqueValues, idx)))

        logits = tf.gather_nd(pred_class_logits, tf.where(tf.equal(target_class_ids, idx)))
        target = tf.gather_nd(target_class_ids, tf.where(tf.equal(target_class_ids, idx)))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)

        losses.append(
            K.switch(tf.size(count) > 0, tf.cast(K.mean(loss), dtype=tf.float32), tf.constant(-1, dtype=tf.float32)))

    losses = tf.stack(losses)
    losses = tf.gather(losses, tf.where(tf.greater_equal(losses, 0.)))

    return tf.reduce_mean(losses)

def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox:      [batch, num_rois, (dz, dy, dx)]
    target_class_ids: [batch, num_rois].                                       Integer class IDs.
    pred_bbox:        [batch, num_rois, num_classes, (dz, dy, dx)]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = K.reshape(target_class_ids, (-1,))
    target_bbox = K.reshape(target_bbox, (-1, 3))
    pred_bbox = K.reshape(pred_bbox, (-1, K.int_shape(pred_bbox)[2], 3))

    # Only positive ROIs contribute to the loss. And only
    # the right class_id of each ROI. Get their indices.
    positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    loss = K.switch(tf.size(target_bbox) > 0,
                    K.mean(K.mean(K.abs(target_bbox - pred_bbox), axis=-1)),
                    tf.constant(0.0))
    return loss

############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, start_from_ckpt, weight_path, model_dir='./models'):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.keras_model = self.build(mode=mode, config=config)

        if self.mode == 'training':
            if start_from_ckpt:
                self.keras_model.load_weights(weight_path)
                self.set_log_dir(weight_path)
            else:
                self.set_log_dir()
        else:
            self.keras_model.load_weights(weight_path, by_name=True)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']
        
        # Inputs
        if config.IMAGE_FORMAT == 'channels_first':
            input_image = KL.Input(shape=[config.IMAGE_NUM_CHS, None, None, None], name="input_image")
        else:
            input_image = KL.Input(shape=[None, None, None, config.IMAGE_NUM_CHS], name="input_image")

        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")
        
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(shape=[None, 3], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (center_z, center_y, center_x)] in image coordinates
            input_gt_boxes = KL.Input(shape=[None, 6], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            if config.IMAGE_FORMAT == 'channels_first':
                gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[2:5]))(input_gt_boxes)
            else:
                gt_boxes = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_gt_boxes)

        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 6], name="input_anchors")

        # Build the shared convolutional layers.
        # Bottom-up Layers
        _, C2, C3 = Backbone(input_image, train_bn=config.TRAIN_BN) # train_bn=config.TRAIN_BN

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        if K.int_shape(C3)[-1] != config.TOP_DOWN_PYRAMID_SIZE:
            P3 = KL.Conv3D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", data_format='channels_last', name="fpn_c3p3")(C3)
        else:
            P3 = C3
        
        P2 = KL.Add(name="fpn_p3add")([
                                       KL.UpSampling3D(size=(2, 2, 2), name="fpn_p4upsampled")(P3),
                                       KL.Conv3D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3, 3), padding="SAME", data_format='channels_last', name='fpn_c2p2')(C2)])

        # Note that P4 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3]
        mrcnn_feature_maps = [P2, P3]

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
            # A hack to get around Keras's bad support for constants
            anchors = AnchorsLayer(name="anchors")(anchors)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE, config.ANCHORS_PER_LOCATION, config.TOP_DOWN_PYRAMID_SIZE)
        
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))

        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" else config.POST_NMS_ROIS_INFERENCE

        rpn_rois = ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=config.RPN_NMS_THRESHOLD,
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors, input_image_meta])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image came from.
            active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x)["active_class_ids"])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 6], name="input_roi", dtype=np.int32)
                # Normalize coordinates
                # target_rois = KL.Lambda(lambda x: norm_boxes_graph(x, K.shape(input_image)[1:4]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox =\
                DetectionTargetLayer(config, name="proposal_targets")([target_rois, input_gt_class_ids, gt_boxes])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, 
                                     mrcnn_feature_maps, 
                                     input_image_meta,
                                     config.POOL_SIZE, 
                                     config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
            
            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            
            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")([input_rpn_match, rpn_class_logits])
            rpn_bbox_loss  = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")([input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss     = KL.Lambda(lambda x: mrcnn_class_loss_graph(config, *x), name="mrcnn_class_loss")([target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss      = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")([target_bbox, target_class_ids, mrcnn_bbox])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes]

            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, 
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]
                        
            model = KM.Model(inputs, outputs, name='mask_rcnn')
        else:
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, 
                                     mrcnn_feature_maps, 
                                     input_image_meta,
                                     config.POOL_SIZE, 
                                     config.NUM_CLASSES,
                                     train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            detections = DetectionLayer(config, name="mrcnn_detection")([rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            
            model = KM.Model([input_image, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox, rpn_rois, rpn_class, rpn_bbox],
                             name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model
    
    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = tf.keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                tf.reduce_mean(layer.output, keepdims=True) 
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            tf.keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            
            self.keras_model.add_metric(loss, name=name, aggregation='mean') # This is to explicitly add metrics to the model

    def set_log_dir(self, model_path=None):
        """
        Called in MaskRCNN.__init__(). Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()
        
        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = regex = r".*/3d\_faster\_rcnn(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/3d\_faster\_rcnn\.(\d{2})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)
        
        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))
        
        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Save best weights only
        self.checkpoint_path = os.path.join(self.log_dir, self.config.NAME + ".{epoch:02d}.h5")
        
    def train(self):

        assert self.mode == "training", "Create model in training mode."
 
        # Data generators
        train_generator = data_generator(self.config.TRAIN_DATA_FOLDER, 
                                         self.config.TRAIN_IDS, 
                                         self.config, 
                                         shuffle=True,
                                         batch_size=self.config.BATCH_SIZE)
        
        val_generator = data_generator(self.config.VAL_DATA_FOLDER, 
                                       self.config.VAL_IDS, 
                                       self.config, 
                                       shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False, profile_batch=0),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_best_only=False, save_weights_only=True, save_freq='epoch'),
        ]

        # Train
        utils.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, self.config.LEARNING_RATE))
        # utils.log("Checkpoint Path: {}".format(self.checkpoint_path))

        self.compile(self.config.LEARNING_RATE, self.config.LEARNING_MOMENTUM)
        
        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=self.config.EPOCHS,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=1,
            workers=1,
            use_multiprocessing=False,
        )

    def detect(self, data_path, test_image_ids, verbose=0):
        
        import timeit
        import pandas as pd
        from groups import group1
        from id2name import id2name
        
        assert self.mode == "inference", "Create model in inference mode."

        for idx in range(len(test_image_ids)):
            image_id = test_image_ids[idx]
            print('Processing subject {}'.format(image_id))
            # Load image and mask
            image = sitk.ReadImage(data_path + '/images_1.6/{0}.nii.gz'.format(image_id))
            image = sitk.GetArrayFromImage(image)
            
            with h5py.File(data_path + '/coordinates_1.6/{0}.hdf5'.format(image_id), 'r') as f:
                lmks = f['landmark_i'][()][:, ::-1]
            
            image_size = np.array(image.shape)
            
            toLarge = 0
            cropStart = []
            if image_size[0] > self.config.IMAGE_DEPTH - self.config.LENGTH:
                toLarge = 1
                cropStart.append(image_size[0]//2 - (self.config.IMAGE_DEPTH - self.config.LENGTH)//2)
                z_range = slice(image_size[0]//2 - (self.config.IMAGE_DEPTH - self.config.LENGTH)//2, image_size[0]//2 + (self.config.IMAGE_DEPTH - self.config.LENGTH)//2)
                image = image[z_range]
            else:
                cropStart.append(0)
                image = image
        
            if image_size[1] > self.config.IMAGE_HEIGHT - self.config.LENGTH:
                toLarge = 1
                cropStart.append(image_size[1]//2 - (self.config.IMAGE_HEIGHT - self.config.LENGTH)//2)
                y_range = slice(image_size[1]//2 - (self.config.IMAGE_HEIGHT - self.config.LENGTH)//2, image_size[1]//2 + (self.config.IMAGE_HEIGHT - self.config.LENGTH)//2)
                image = image[:, y_range]
            else:
                cropStart.append(0)
                image = image
            
            if image_size[2] > self.config.IMAGE_WIDTH - self.config.LENGTH:
                toLarge = 1
                cropStart.append(image_size[2]//2 - (self.config.IMAGE_WIDTH - self.config.LENGTH)//2)
                x_range = slice(image_size[2]//2 - (self.config.IMAGE_WIDTH - self.config.LENGTH)//2, image_size[2]//2 + (self.config.IMAGE_WIDTH - self.config.LENGTH)//2)
                image = image[:, :, x_range]
            else:
                cropStart.append(0)
                image = image

            if toLarge:
                print('WARNING: Image too large! Be careful, landmarks out of bound may not be detected.')
            
            cropStart = np.array(cropStart)
            #print(cropStart)
            lmks = lmks - cropStart
            #print(lmks)
            
            image_size = np.array(image.shape)
            #print(image_size)
            Expand_to = np.array([self.config.IMAGE_DEPTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH])
            residual = Expand_to - image_size
            #print(residual)
            
            side_length = self.config.LENGTH
            pad_front = np.array(residual) // 2 + side_length//2
            
            pad_front = list(pad_front.astype(np.int16)) # pad_front is a list now
            padded_image = np.zeros(Expand_to + np.array([side_length, side_length, side_length]), dtype=np.float32)
            padded_image[pad_front[0]:(pad_front[0]+image_size[0]),
                         pad_front[1]:(pad_front[1]+image_size[1]),
                         pad_front[2]:(pad_front[2]+image_size[2])] = image
            image = padded_image
            
            lmks = lmks + pad_front # Record ground truth
            
            class_ids = np.array(np.arange(1, self.config.NUM_CLASSES)) # class 0 represents background

            """Ground truth"""
            print('Ground truth in order of z, y, and x (physical space):')
            for idx in range(len(class_ids)):
                landmark_i = lmks[idx]
                if np.all(np.logical_and(landmark_i > pad_front, landmark_i < pad_front + image_size)):
                    print('Landmark {0}: {1}'.format(class_ids[idx], 1.6*(landmark_i - pad_front + cropStart)))
            print('\n')
            
            # Image meta data
            img_size = image.shape
            # image_meta = compose_image_meta(image_id, img_size, [0, 0, 0, img_size[0], img_size[1], img_size[2]], active_class_ids)
            image_meta = compose_image_meta(image_id.split('_')[1], img_size, [0, 0, 0, img_size[0], img_size[1], img_size[2]], np.zeros([self.config.NUM_CLASSES], dtype=np.int32))

            image_shape = np.array(image.shape)
            assert np.all(image_shape == np.array([self.config.IMAGE_DEPTH, self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH]) + np.array([side_length, side_length, side_length]))

            anchors = self.get_anchors(image_shape)
            anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
            
            if verbose:
                utils.log("padded_images", image)
                utils.log("image_metas", image_meta)
                utils.log("anchors", anchors)
    
            image      = np.expand_dims(np.expand_dims(image, axis=-1), axis=0)
            image_meta = np.expand_dims(image_meta, axis=0)
            
            # Run object detection
            start = timeit.default_timer()
            
            detections, _, _, _, _, _ =\
            self.keras_model.predict([image, image_meta, anchors], verbose=0)
            
            end = timeit.default_timer()
            
            assert detections.shape[0] == 1
            detections = detections[0]
            indexes = detections[:, 6]
            argsort = np.argsort(indexes)
            detections = detections[argsort]

            print('Prediction:')
            
            predictions = np.zeros([self.config.MAX_GT_INSTANCES, 3], dtype=np.float32)
            
            rows = [id2name[x] for x in group1]
            
            for idx in range(detections.shape[0]):
                detection = detections[idx]
                [z1, y1, x1, z2, y2, x2, ind, prob] = list(detection)
                if ind > 0 and np.all(predictions[int(ind)-1] == np.array([0, 0, 0])):
                    predictions[int(ind)-1] = 1.6*np.array([(z1+z2)/2-pad_front[0]+cropStart[0], (y1+y2)/2-pad_front[1]+cropStart[1], (x1+x2)/2-pad_front[2]+cropStart[2]])
                    print(id2name[group1[int(ind)-1]], 1.6*np.array([(z1+z2)/2-pad_front[0]+cropStart[0], (y1+y2)/2-pad_front[1]+cropStart[1], (x1+x2)/2-pad_front[2]+cropStart[2]]))
                    
            if not os.path.exists('predictions'):
                os.makedirs('predictions')
            
            df = pd.DataFrame(data=predictions, columns=['z', 'y', 'x'])
            df.index = rows
            df.to_csv('./predictions/{0}.csv'.format(image_id))
                        
            print('Time consumed: ', end - start, '\n')
            
    def get_anchors(self, image_shape):
        """Returns anchor pyramid for the given image size."""
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        assert len(backbone_shapes) == len(self.config.BACKBONE_STRIDES), 'Vital error: Please make sure the num of levels of feature pyramid and the num of backbone strides are consistent'

        # Cache anchors and reuse if image shape is the same
        if not hasattr(self, "_anchor_cache"):
            self._anchor_cache = {}
        if not tuple(image_shape) in self._anchor_cache:
            # Generate Anchors
            a = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                backbone_shapes,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE)
            # Keep a copy of the latest anchors in pixel coordinates because
            # it's used in inspect_model notebooks.
            # TODO: Remove this after the notebook are refactored to not use it
            self.anchors = a
            # Normalize coordinates
            self._anchor_cache[tuple(image_shape)] = a
        return self._anchor_cache[tuple(image_shape)]

############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(image_shape) +           # size=3
        list(window) +                # size=6
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:10]  # (z1, y1, x1, z2, y2, x2) window of image in in pixels (not normalized coordinates)
    active_class_ids = meta[:, 10:]
    return {
        "image_id": image_id.astype(np.int32),
        "image_shape": image_shape.astype(np.int32),
        "window": window.astype(np.int32),
        "active_class_ids": active_class_ids.astype(np.int32)
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    image_shape = meta[:, 1:4]
    window = meta[:, 4:10]  # (z1, y1, x1, z2, y2, x2) window of image in in pixels (not normalized coordinates)
    active_class_ids = meta[:, 10:]
    return {
        "image_id": image_id,
        "image_shape": image_shape,
        "window": window,
        "active_class_ids": active_class_ids
    }

############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name=None):
    """Often boxes are represented with matrices of shape [N, 6] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 6] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """To be deleted later
    """
    return boxes # - tf.constant([0., 0., 0., 1., 1., 1.])


def denorm_boxes_graph(boxes, shape):
    """To be deleted later
    """
    return boxes # + tf.constant([0., 0., 0., 1., 1., 1.])

import argparse

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="3dFasterRCNN")
    parser.add_argument("--mode", type=str, default="training",
                        help="either 'training' or 'inference'.")
    parser.add_argument("--start-epoch", type=int, default=0,
                        help="the starting epoch.'")
    parser.add_argument("--weight-path", type=str, default='./models/3d_faster_rcnn20210328T2226/3d_faster_rcnn.15.h5',
                        help="path to weight for contuning training or inference.'")
    parser.add_argument("--test-data-path", type=str, default=os.getcwd(),
                        help="path to where testing data are stored (may be different from training/val data).'")
    return parser.parse_args()

if __name__ == '__main__':
    from config import Config
    config = Config()
    
    args = get_arguments()
    
    mrcnn = MaskRCNN(mode=args.mode, config=config, start_from_ckpt=args.start_epoch>0, weight_path=args.weight_path)
    #mrcnn.keras_model.summary()
    
    if args.mode == 'training':
        mrcnn.train()
    else:
        mrcnn.detect(args.test_data_path, config.VAL_IDS)

