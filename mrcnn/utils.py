import sys
import os
import math
import h5py
import random
import numpy as np
import tensorflow as tf
import scipy
import shutil
import warnings
from itertools import permutations, combinations_with_replacement

# URL from which to download the latest COCO trained weights
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
        prints it's shape, min, and max values.
        """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
                                                                         str(array.shape),
                                                                         array.min() if array.size else "",
                                                                         array.max() if array.size else "",
                                                                         array.dtype))
    print(text)

############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 6], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, :, i]
        # Bounding box.
        depth_indicies = np.where(np.any(m, axis=(1, 2)))[0]
        vertical_indicies = np.where(np.any(m, axis=(0, 2)))[0]
        horizontal_indicies = np.where(np.any(m, axis=(0, 1)))[0]
        if depth_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            z1, z2 = depth_indicies[[0, -1]]
            # x2, y2, z2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
            z2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2, z1, z2 = 0, 0, 0, 0, 0, 0
        boxes[i] = np.array([z1, y1, x1, z2, y2, x2])
    return boxes.astype(np.int32)


def compute_iou(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [z1, y1, x1, z2, y2, x2]
    boxes: [boxes_count, (z1, y1, x1, z2, y2, x2)]
    box_volume: float. the volume of 'box'
    boxes_volume: array of length boxes_count.

    Note: the volumes are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection volumes
    z1 = np.maximum(box[0], boxes[:, 0])
    z2 = np.minimum(box[3], boxes[:, 3])
    y1 = np.maximum(box[1], boxes[:, 1])
    y2 = np.minimum(box[4], boxes[:, 4])
    x1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = (intersection + 1e-9) / (union + 1e-6)
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (z1, y1, x1, z2, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Volumes of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, volume2[i], volume1)
    
    return overlaps


def compute_overlaps_masks(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Depth, Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[0] == 0 or masks2.shape[0] == 0:
        return np.zeros((masks1.shape[0], masks2.shape[-1]))
    # flatten masks and compute their volumes
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    volume1 = np.sum(masks1, axis=0)
    volume2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = volume1[:, None] + volume2[None, :] - intersections
    overlaps = intersections / union

    return overlaps


def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)

    # Compute box areas
    z1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x1 = boxes[:, 2]
    z2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x2 = boxes[:, 5]
    volume = (z2 - z1) * (y2 - y1) * (x2 - x1)

    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]

    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], volume[i], volume[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def apply_box_deltas(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (z1, y1, x1, z2, y2, x2)]. Note that (z2, y2, x2) is outside the box.
    deltas: [N, (dz, dy, dx, log(dd), log(dh), log(dw))]
    """
    boxes = boxes.astype(np.float32)
    # Convert to z, y, x, d, h, w
    depth = boxes[:, 3] - boxes[:, 0]
    height = boxes[:, 4] - boxes[:, 1]
    width = boxes[:, 5] - boxes[:, 2]
    center_z = boxes[:, 0] + 0.5 * depth
    center_y = boxes[:, 1] + 0.5 * height
    center_x = boxes[:, 2] + 0.5 * width
    # Apply deltas
    center_z += deltas[:, 0] * depth
    center_y += deltas[:, 1] * height
    center_x += deltas[:, 2] * width
    depth *= np.exp(deltas[:, 3])
    height *= np.exp(deltas[:, 4])
    width *= np.exp(deltas[:, 5])
    # Convert back to z1, y1, x1, z2, y2, x2
    z1 = center_z - 0.5 * depth
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    z2 = z1 + depth
    y2 = y1 + height
    x2 = x1 + width
    return np.stack([z1, y1, x1, z2, y2, x2], axis=1)


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = tf.log(gt_depth / depth)
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dz, dy, dx, dd, dh, dw], axis=1)
    return result

# Numpy version
def box_refinement(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (z1, y1, x1, z2, y2, x2)]. (z2, y2, x2) is
    assumed to be outside the box.
    """
    box = box.astype(np.float32)
    gt_box = gt_box.astype(np.float32)

    depth = box[:, 3] - box[:, 0]
    height = box[:, 4] - box[:, 1]
    width = box[:, 5] - box[:, 2]
    center_z = box[:, 0] + 0.5 * depth
    center_y = box[:, 1] + 0.5 * height
    center_x = box[:, 2] + 0.5 * width

    gt_depth = gt_box[:, 3] - gt_box[:, 0]
    gt_height = gt_box[:, 4] - gt_box[:, 1]
    gt_width = gt_box[:, 5] - gt_box[:, 2]
    gt_center_z = gt_box[:, 0] + 0.5 * gt_depth
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height
    gt_center_x = gt_box[:, 2] + 0.5 * gt_width

    dz = (gt_center_z - center_z) / depth
    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dd = np.log(gt_depth / depth)
    dh = np.log(gt_height / height)
    dw = np.log(gt_width / width)

    return np.stack([dz, dy, dx, dd, dh, dw], axis=1)

############################################################
#  Anchors
############################################################

def all_combinations(ratios):
    total = set()
    
    for i in combinations_with_replacement(ratios, 3):
        total |= set(permutations(i))

    total = [list(x) for x in sorted(total)]
    return total

def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    scales: 1D array of anchor sizes in pixels. Example: [32, 64, 128]
    ratios: 1D array of anchor ratios of width/height. Example: [0.5, 1, 2]
    shape: [depth, height, width] spatial shape of the feature map over which
            to generate anchors.
    feature_stride: Stride of the feature map relative to the image in pixels. # Should be the same as that in RPN part
    anchor_stride: Stride of anchors on the feature map. For example, if the   # Should be the same as that in RPN part
        value is 2 then generate anchors for every other feature map pixel.
    """
    # Get all combinations of scales and ratios
    ratios = np.array(all_combinations(ratios))
    #print(ratios)
    #print(ratios.shape)
    ratios = np.transpose(ratios, [1, 0])

    final_boxes = []
    for scale in scales:
        scale = np.array([scale] * (ratios.shape[1]))
        
        # Enumerate depths, heights and widths from scales and ratios
        depths = scale * ratios[0]
        heights = scale * ratios[1]
        widths = scale * ratios[2]
        
        # Enumerate shifts in feature space
        shifts_z = np.arange(0, shape[0], anchor_stride) * feature_stride
        shifts_y = np.arange(0, shape[1], anchor_stride) * feature_stride
        shifts_x = np.arange(0, shape[2], anchor_stride) * feature_stride
        shifts_y, shifts_z, shifts_x = np.meshgrid(shifts_y, shifts_z, shifts_x)

        # Enumerate combinations of shifts, widths, and heights
        # EXAMPLE:
        # heights = np.array([10, 20, 30]) and widths = np.array([10, 20, 30])
        # shifts_y = np.array([2, 4]) and shifts_y = np.array([2, 4])
        # then
        #   box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
        #   box_widths, box_centers_x = np.meshgrid(widths, shifts_x)
        #   box_centers = np.stack([box_centers_y, box_centers_x], axis=1).reshape([-1, 2])
        #   box_sizes = np.stack([box_heights, box_widths], axis=1).reshape([-1, 2])
        # ==>
        # box_centers = array([[10, 10], and box_sizes = array([[2, 4],
        #                      [10, 10],                        [2, 4],
        #                      [20, 20],                        [2, 4],
        #                      [20, 20],                        [2, 4],
        #                      [30, 30],                        [2, 4],
        #                      [30, 30]])                       [2, 4]])
        
        box_depths, box_centers_z = np.meshgrid(depths, shifts_z)
        box_heights, box_centers_y = np.meshgrid(heights, shifts_y)
        box_widths, box_centers_x = np.meshgrid(widths, shifts_x)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = np.stack([box_centers_z, box_centers_y, box_centers_x], axis=2).reshape([-1, 3])
        box_sizes = np.stack([box_depths, box_heights, box_widths], axis=2).reshape([-1, 3])

        # Convert to corner coordinates (z1, y1, x1, z2, y2, x2)
        boxes = np.concatenate([box_centers - 0.5 * box_sizes,
                                box_centers + 0.5 * box_sizes], axis=1)
        # print(boxes.shape)
        final_boxes.append(boxes)
        
    return np.concatenate(final_boxes, axis=0)


def generate_pyramid_anchors(scales, ratios, feature_shapes, feature_strides,
                             anchor_stride):
    """Generate anchors at different levels of a feature pyramid. Each scale
    is associated with a level of the pyramid, but each ratio is used in
    all levels of the pyramid.

    Returns:
    anchors: [N, (z1, y1, x1, z2, y2, x2)]. All generated anchors in one array. Sorted
        with the same order of the given scales. So, anchors of scale[0] come
        first, then anchors of scale[1], and so on.
    """
    # Anchors
    # [anchor_count, (z1, y1, x1, z2, y2, x2)]
    anchors = []
    for i in range(feature_shapes.shape[0]):
        anchors.append(generate_anchors(scales, ratios, feature_shapes[i],
                                        feature_strides[i], anchor_stride))
    # print(np.concatenate(anchors, axis=0).shape)
    return np.concatenate(anchors, axis=0)


############################################################
#  Miscellaneous
############################################################

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

# ## Batch Slicing
# Some custom layers support a batch size of 1 only, and require a lot of work
# to support batches greater than 1. This function slices an input tensor
# across the batch dimension and feeds batches of size 1. Effectively,
# an easy way to support batches > 1 quickly with little code modification.
# In the long run, it's more efficient to modify the code to support large
# batches and getting rid of this function. Consider this a temporary solution
def batch_slice(inputs, graph_fn, batch_size, names=None):
    """Splits inputs into slices and feeds each slice to a copy of the given
    computation graph and then combines the results. It allows you to run a
    graph on a batch of inputs even if the graph is written to support one
    instance only.

    inputs: list of tensors. All must have the same first dimension length
    graph_fn: A function that returns a TF tensor that's part of a graph.
    batch_size: number of slices to divide the data into.
    names: If provided, assigns names to the resulting tensors.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]

    outputs = []
    for i in range(batch_size):
        inputs_slice = [x[i] for x in inputs]
        output_slice = graph_fn(*inputs_slice)
        if not isinstance(output_slice, (tuple, list)):
            output_slice = [output_slice]
        outputs.append(output_slice)
    # Change outputs from a list of slices where each is
    # a list of outputs to a list of outputs and each has
    # a list of slices
    outputs = list(zip(*outputs))

    if names is None:
        names = [None] * len(outputs)

    result = [tf.stack(o, axis=0, name=n)
              for o, n in zip(outputs, names)]
    if len(result) == 1:
        result = result[0]

    return result

# NOT useful here
def download_trained_weights(coco_model_path, verbose=1):
    """Download COCO trained weights from Releases.

    coco_model_path: local path of COCO trained weights
    """
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")


def norm_boxes(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    shape: [..., (depth, height, width)] in pixels

    Note: In pixel coordinates (z2, y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    """
    d, h, w = shape
    scale = np.array([d - 1, h - 1, w - 1, d - 1, h - 1, w - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    return np.divide((boxes - shift), scale).astype(np.float32)


def denorm_boxes(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [N, (z1, y1, x1, z2, y2, x2)] in normalized coordinates
    shape: [..., (depth, height, width)] in pixels

    Note: In pixel coordinates (z2, y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [N, (z1, y1, x1, z2, y2, x2)] in pixel coordinates
    """
    d, h, w = shape
    scale = np.array([d - 1, h - 1, w - 1, d - 1, h - 1, w - 1])
    shift = np.array([0, 0, 0, 1, 1, 1])
    return np.around(np.multiply(boxes, scale) + shift).astype(np.int32)
