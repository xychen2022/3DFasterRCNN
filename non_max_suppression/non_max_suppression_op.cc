/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/image_ops.cc

#define EIGEN_USE_THREADS

#include "non_max_suppression_op.h"
#include "bounds_check.h"
#include "stl_util.h"

#include <functional>
#include <queue>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

//#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
//#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/platform/logging.h"


namespace tensorflow {

using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::InferenceContext;

Status NMSShapeFn(InferenceContext* c) {
  // Get inputs and validate ranks.
  ShapeHandle boxes;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &boxes));
  ShapeHandle scores;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &scores));
  ShapeHandle max_output_size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &max_output_size));
  ShapeHandle iou_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &iou_threshold));
  ShapeHandle score_threshold;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &score_threshold));
  // The boxes is a 2-D float Tensor of shape [num_boxes, 6].
  DimensionHandle unused;
  // The boxes[0] and scores[0] are both num_boxes.
  TF_RETURN_IF_ERROR(c->Merge(c->Dim(boxes, 0), c->Dim(scores, 0), &unused));
  // The boxes[1] is 4.
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 6, &unused));

  c->set_output(0, c->Vector(c->UnknownDim()));
  return Status::OK();
}

REGISTER_OP("NonMaxSuppressionV6")
    .Input("boxes: float")
    .Input("scores: float")
    .Input("max_output_size: int32")
    .Input("iou_threshold: float")
    .Input("score_threshold: float")
    .Output("selected_indices: int32")
    .SetShapeFn(NMSShapeFn);
}

namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

static inline void CheckScoreSizes(OpKernelContext* context, int num_boxes,
                                   const Tensor& scores) {
  // The shape of 'scores' is [num_boxes]
  OP_REQUIRES(context, scores.dims() == 1,
              errors::InvalidArgument("scores must be 1-D",
                                      scores.shape().DebugString()));
  OP_REQUIRES(context, scores.dim_size(0) == num_boxes,
              errors::InvalidArgument("scores has incompatible shape"));
}

static inline void ParseAndCheckBoxSizes(OpKernelContext* context,
                                         const Tensor& boxes, int* num_boxes) {
  // The shape of 'boxes' is [num_boxes, 6]
  OP_REQUIRES(context, boxes.dims() == 2,
              errors::InvalidArgument("boxes must be 2-D",
                                      boxes.shape().DebugString()));
  *num_boxes = boxes.dim_size(0);
  OP_REQUIRES(context, boxes.dim_size(1) == 6,
              errors::InvalidArgument("boxes must have 6 columns"));
}

// Return intersection-over-union overlap between boxes i and j
static inline float IOUGreaterThanThreshold(
    typename TTypes<float, 2>::ConstTensor boxes, int i, int j,
    float iou_threshold) {

  const float zmin_i = std::min<float>(boxes(i, 0), boxes(i, 3));
  const float ymin_i = std::min<float>(boxes(i, 1), boxes(i, 4));
  const float xmin_i = std::min<float>(boxes(i, 2), boxes(i, 5));

  const float zmax_i = std::max<float>(boxes(i, 0), boxes(i, 3));
  const float ymax_i = std::max<float>(boxes(i, 1), boxes(i, 4));
  const float xmax_i = std::max<float>(boxes(i, 2), boxes(i, 5));

  const float zmin_j = std::min<float>(boxes(j, 0), boxes(j, 3));
  const float ymin_j = std::min<float>(boxes(j, 1), boxes(j, 4));
  const float xmin_j = std::min<float>(boxes(j, 2), boxes(j, 5));

  const float zmax_j = std::max<float>(boxes(j, 0), boxes(j, 3));
  const float ymax_j = std::max<float>(boxes(j, 1), boxes(j, 4));
  const float xmax_j = std::max<float>(boxes(j, 2), boxes(j, 5));

  const float volume_i = (zmax_i - zmin_i) * (ymax_i - ymin_i) * (xmax_i - xmin_i);
  const float volume_j = (zmax_j - zmin_j) * (ymax_j - ymin_j) * (xmax_j - xmin_j);

  if (volume_i <= 0 || volume_j <= 0) return 0.0;

  const float intersection_zmin = std::max<float>(zmin_i, zmin_j);
  const float intersection_ymin = std::max<float>(ymin_i, ymin_j);
  const float intersection_xmin = std::max<float>(xmin_i, xmin_j);

  const float intersection_zmax = std::min<float>(zmax_i, zmax_j);
  const float intersection_ymax = std::min<float>(ymax_i, ymax_j);
  const float intersection_xmax = std::min<float>(xmax_i, xmax_j);

  const float intersection_volume =
      std::max<float>(intersection_zmax - intersection_zmin, 0.0) *
      std::max<float>(intersection_ymax - intersection_ymin, 0.0) *
      std::max<float>(intersection_xmax - intersection_xmin, 0.0);

  const float iou = intersection_volume / (volume_i + volume_j - intersection_volume);

  return iou > iou_threshold;
}

static inline bool OverlapsGreaterThanThreshold(
    typename TTypes<float, 2>::ConstTensor overlaps, int i, int j,
    float overlap_threshold) {
  return overlaps(i, j) > overlap_threshold;
}

static inline std::function<bool(int, int)> CreateIOUSuppressCheckFn(
    const Tensor& boxes, float threshold) {
  typename TTypes<float, 2>::ConstTensor boxes_data = boxes.tensor<float, 2>();
  return std::bind(&IOUGreaterThanThreshold, boxes_data, std::placeholders::_1,
                   std::placeholders::_2, threshold);
}

static inline std::function<bool(int, int)> CreateOverlapsSuppressCheckFn(
    const Tensor& overlaps, float threshold) {
  typename TTypes<float, 2>::ConstTensor overlaps_data =
      overlaps.tensor<float, 2>();
  return std::bind(&OverlapsGreaterThanThreshold, overlaps_data,
                   std::placeholders::_1, std::placeholders::_2, threshold);
}

void DoNonMaxSuppressionOp(
    OpKernelContext* context, const Tensor& scores, int num_boxes,
    const Tensor& max_output_size, const float score_threshold,
    const std::function<bool(int, int)>& suppress_check_fn,
    bool pad_to_max_output_size = false, int* ptr_num_valid_outputs = nullptr) {
  const int output_size = max_output_size.scalar<int>()();

  std::vector<float> scores_data(num_boxes);
  std::copy_n(scores.flat<float>().data(), num_boxes, scores_data.begin());

  // Data structure for selection candidate in NMS.
  struct Candidate {
    int box_index;
    float score;
  };

  auto cmp = [](const Candidate bs_i, const Candidate bs_j) {
    return bs_i.score < bs_j.score;
  };
  std::priority_queue<Candidate, std::deque<Candidate>, decltype(cmp)>
      candidate_priority_queue(cmp);
  for (int i = 0; i < scores_data.size(); ++i) {
    if (scores_data[i] > score_threshold) {
      candidate_priority_queue.emplace(Candidate({i, scores_data[i]}));
    }
  }

  std::vector<int> selected;
  std::vector<float> selected_scores;
  Candidate next_candidate;

  while (selected.size() < output_size && !candidate_priority_queue.empty()) {
    next_candidate = candidate_priority_queue.top();
    candidate_priority_queue.pop();

    // Overlapping boxes are likely to have similar scores,
    // therefore we iterate through the previously selected boxes backwards
    // in order to see if `next_candidate` should be suppressed.
    bool should_select = true;
    for (int j = selected.size() - 1; j >= 0; --j) {
      if (suppress_check_fn(next_candidate.box_index, selected[j])) {
        should_select = false;
        break;
      }
    }

    if (should_select) {
      selected.push_back(next_candidate.box_index);
      selected_scores.push_back(next_candidate.score);
    }
  }

  int num_valid_outputs = selected.size();
  if (pad_to_max_output_size) {
    selected.resize(output_size, 0);
    selected_scores.resize(output_size, 0);
  }
  if (ptr_num_valid_outputs) {
    *ptr_num_valid_outputs = num_valid_outputs;
  }

  // Allocate output tensors
  Tensor* output_indices = nullptr;
  TensorShape output_shape({static_cast<int>(selected.size())});
  OP_REQUIRES_OK(context,
                 context->allocate_output(0, output_shape, &output_indices));
  TTypes<int, 1>::Tensor output_indices_data = output_indices->tensor<int, 1>();
  std::copy_n(selected.begin(), selected.size(), output_indices_data.data());
}

}  // namespace


template <typename Device>
class NonMaxSuppressionV6Op : public OpKernel {
 public:
  explicit NonMaxSuppressionV6Op(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // boxes: [num_boxes, 6]
    const Tensor& boxes = context->input(0);
    // scores: [num_boxes]
    const Tensor& scores = context->input(1);
    // max_output_size: scalar
    const Tensor& max_output_size = context->input(2);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(max_output_size.shape()),
        errors::InvalidArgument("max_output_size must be 0-D, got shape ",
                                max_output_size.shape().DebugString()));
    // iou_threshold: scalar
    const Tensor& iou_threshold = context->input(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(iou_threshold.shape()),
                errors::InvalidArgument("iou_threshold must be 0-D, got shape ",
                                        iou_threshold.shape().DebugString()));
    const float iou_threshold_val = iou_threshold.scalar<float>()();

    // score_threshold: scalar
    const Tensor& score_threshold = context->input(4);
    OP_REQUIRES(
        context, TensorShapeUtils::IsScalar(score_threshold.shape()),
        errors::InvalidArgument("score_threshold must be 0-D, got shape ",
                                score_threshold.shape().DebugString()));
    const float score_threshold_val = score_threshold.scalar<float>()();

    OP_REQUIRES(context, iou_threshold_val >= 0 && iou_threshold_val <= 1,
                errors::InvalidArgument("iou_threshold must be in [0, 1]"));
    int num_boxes = 0;
    ParseAndCheckBoxSizes(context, boxes, &num_boxes);
    CheckScoreSizes(context, num_boxes, scores);
    if (!context->status().ok()) {
      return;
    }
    auto suppress_check_fn = CreateIOUSuppressCheckFn(boxes, iou_threshold_val);

    DoNonMaxSuppressionOp(context, scores, num_boxes, max_output_size,
                          score_threshold_val, suppress_check_fn);
  }
};

/*
#define REGISTER_KERNEL(T)  \
  REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV6").Device(DEVICE_CPU), \
                          NonMaxSuppressionV6Op<CPUDevice>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
*/

REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV6").Device(DEVICE_CPU),
                        NonMaxSuppressionV6Op<CPUDevice>);


#if GOOGLE_CUDA

// Forward declaration of the CheckValidBoxIndexHelper specialization for GPU.
namespace functor {

#define REGISTER_KERNEL()                                         \
  REGISTER_KERNEL_BUILDER(Name("NonMaxSuppressionV6")           \
                              .Device(DEVICE_GPU)                  \
                          NonMaxSuppressionV6Op<GPUDevice>);

//REGISTER_GPU_KERNEL(float);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA
} // namespace tensorflow
