#define EIGEN_USE_THREADS

#include "crop_and_resize_op.h"
#include "bounds_check.h"

#include <functional>
#include <string>
#include <cstring>

// #include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

//#include "tensorflow/core/kernels/bounds_check.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/core/platform/stream_executor.h"

using stream_executor::cuda::ScopedActivateExecutorContext;
#endif  // GOOGLE_CUDA

namespace tensorflow {
using ::tensorflow::shape_inference::DimensionHandle;
using ::tensorflow::shape_inference::ShapeHandle;
using ::tensorflow::shape_inference::InferenceContext;

Status SetOutputToSizedImage(InferenceContext* c, DimensionHandle batch_dim,
	                     int size_input_idx, DimensionHandle channel_dim) {
  // Verify shape of size input.
  ShapeHandle size;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(size_input_idx), 1, &size));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(c->WithValue(c->Dim(size, 0), 3, &unused));

  // Get size values from the size tensor.
  const Tensor* size_tensor = c->input_tensor(size_input_idx);
  DimensionHandle depth;
  DimensionHandle width;
  DimensionHandle height;
  if (size_tensor == nullptr) {
    width = c->UnknownDim();
    height = c->UnknownDim();
    depth = c->UnknownDim();
  } else {
    // TODO(petewarden) - Remove once we have constant evaluation in C++ only.
    if (size_tensor->dtype() != DT_INT32) {
      return errors::InvalidArgument(
	  "Bad size input type for SetOutputToSizedImage: Expected DT_INT32 "
	  "but got ",
	  DataTypeString(size_tensor->dtype()), " for input #", size_input_idx,
	  " in ", c->DebugString());
    }
    auto vec = size_tensor->vec<int32>();
    depth = c->MakeDim(vec(0));
    height = c->MakeDim(vec(1));
    width = c->MakeDim(vec(2));
  }
  c->set_output(0, c->MakeShape({batch_dim, depth, height, width, channel_dim}));
  return Status::OK();
}


REGISTER_OP("CropAndResizeV3")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("crop_size: int32")
    .Output("crops: float")
    .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
    .Attr("method: {'trilinear'} = 'trilinear'")
    .Attr("extrapolation_value: float = 0")
    .SetShapeFn([](InferenceContext* c) {
      // Get inputs and validate ranks.
      //const Shape* input;
      ShapeHandle input;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 5, &input));
      ShapeHandle boxes;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &boxes));
      ShapeHandle box_ind;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &box_ind));

      // boxes[0] and box_ind[0] are both num_boxes.
      DimensionHandle num_boxes_dim;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(boxes, 0), c->Dim(box_ind, 0), &num_boxes_dim));

      // boxes.dim(1) is 6.
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(boxes, 1), 6, &unused));

      return SetOutputToSizedImage(c, num_boxes_dim, 3 /* size_input_idx */,
                                   c->Dim(input, 4));
    })
    .Doc(R"doc(
Extracts crops from the input image tensor and trilinearly resizes them (possibly
with aspect ratio change) to a common output size specified by `crop_size`. This
is more general than the `crop_to_bounding_box` op which extracts a fixed size
slice from the input image and does not allow resizing or aspect ratio change.
Returns a tensor with `crops` from the input `image` at positions defined at the
bounding box locations in `boxes`. The cropped boxes are all resized (with
trilinear interpolation) to a fixed `size = [crop_depth, crop_height, crop_width]`. 
The result is a 5-D tensor `[num_boxes, crop_depth, crop_height, crop_width, depth]`.

image: A 5-D tensor of shape `[batch, image_depth, image_height, image_width, depth]`.
   `image_depth`, `image_height` and `image_width` need to be positive.

boxes: A 2-D tensor of shape `[num_boxes, 6]`. The `i`-th row of the tensor
  specifies the coordinates of a box in the `box_ind[i]` image and is specified
  in normalized coordinates `[z1, y1, x1, z2, y2, x2]`. A normalized coordinate value 
  of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
  `[0, 1]` interval of normalized image height is mapped to
  `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
  which case the sampled crop is an up-down flipped version of the original
  image. The width dimension is treated similarly. Normalized coordinates
  outside the `[0, 1]` range are allowed, in which case we use
  `extrapolation_value` to extrapolate the input image values.

box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
  The value of `box_ind[i]` specifies the image that the `i`-th box refers to.

crop_size: A 1-D tensor of 3 elements, `size = [crop_depth, crop_height, crop_width]`. 
  All cropped image patches are resized to this size. The aspect ratio of the image
  content is not preserved. `crop_depth`, `crop_height` and `crop_width` need to be
  positive.

crops: A 5-D tensor of shape `[num_boxes, crop_depth, crop_height, crop_width, depth]`.

method: A string specifying the interpolation method. Only 'trilinear' is
  supported for now.

extrapolation_value: Value used for extrapolation, when applicable.
)doc");

REGISTER_OP("CropAndResizeV3GradImage")
    .Input("grads: float")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Input("image_size: int32")
    .Output("output: T")
    .Attr("T: {float, half, double}")
    .Attr("method: {'trilinear'} = 'trilinear'")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(3, &out));
      TF_RETURN_IF_ERROR(c->WithRank(out, 5, &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of the crop_and_resize op wrt the input image tensor.
grads: A 5-D tensor of shape `[num_boxes, crop_depth, crop_height, crop_width, depth]`.

boxes: A 2-D tensor of shape `[num_boxes, 6]`. The `i`-th row of the tensor
  specifies the coordinates of a box in the `box_ind[i]` image and is specified
  in normalized coordinates `[z1, y1, x1, z2, y2, x2]`. A normalized coordinate value 
  of `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
  `[0, 1]` interval of normalized image height is mapped to
  `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
  which case the sampled crop is an up-down flipped version of the original
  image. The width dimension is treated similarly. Normalized coordinates
  outside the `[0, 1]` range are allowed, in which case we use
  `extrapolation_value` to extrapolate the input image values.

box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
  The value of `box_ind[i]` specifies the image that the `i`-th box refers to.

image_size: A 1-D tensor with value `[batch, image_depth, image_height, image_width, depth]`
  containing the original image size. `image_depth`, `image_height` and `image_width` need
  to be positive.

output: A 5-D tensor of shape `[batch, image_depth, image_height, image_width, depth]`.

method: A string specifying the interpolation method. Only 'trilinear' is
  supported for now.
)doc");

REGISTER_OP("CropAndResizeV3GradBoxes")
    .Input("grads: float")
    .Input("image: T")
    .Input("boxes: float")
    .Input("box_ind: int32")
    .Output("output: float")
    .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
    .Attr("method: {'trilinear'} = 'trilinear'")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of the crop_and_resize op wrt the input boxes tensor.
grads: A 5-D tensor of shape `[num_boxes, crop_depth, crop_height, crop_width, depth]`.

image: A 5-D tensor of shape `[batch, image_depth, image_height, image_width, depth]`.
  `image_depth`, `image_height` and `image_width` need to be positive.

boxes: A 2-D tensor of shape `[num_boxes, 6]`. The `i`-th row of the tensor
  specifies the coordinates of a box in the `box_ind[i]` image and is specified
  in normalized coordinates `[z1,y1,x1,z2,y2,x2]`. A normalized coordinate value of
  `y` is mapped to the image coordinate at `y * (image_height - 1)`, so as the
  `[0, 1]` interval of normalized image height is mapped to
  `[0, image_height - 1] in image height coordinates. We do allow y1 > y2, in
  which case the sampled crop is an up-down flipped version of the original
  image. The width dimension is treated similarly. Normalized coordinates
  outside the `[0, 1]` range are allowed, in which case we use
  `extrapolation_value` to extrapolate the input image values.

box_ind: A 1-D tensor of shape `[num_boxes]` with int32 values in `[0, batch)`.
  The value of `box_ind[i]` specifies the image that the `i`-th box refers to.

output: A 2-D tensor of shape `[num_boxes, 6]`.

method: A string specifying the interpolation method. Only 'trilinear' is
  supported for now.
)doc");
}


namespace tensorflow {
namespace {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
using Callback = std::function<void()>;

static inline Status ParseAndCheckBoxSizes(const Tensor& boxes,
                                           const Tensor& box_index,
                                           int* num_boxes) {
  if (boxes.NumElements() == 0 && box_index.NumElements() == 0) {
    *num_boxes = 0;
    return Status::OK();
  }
  // The shape of 'boxes' is [num_boxes, 6].
  if (boxes.dims() != 2) {
    return errors::InvalidArgument("boxes must be 2-D",
                                   boxes.shape().DebugString());
  }
  *num_boxes = boxes.dim_size(0);
  if (boxes.dim_size(1) != 6) {
    return errors::InvalidArgument("boxes must have 6 columns");
  }
  // The shape of 'box_index' is [num_boxes].
  if (box_index.dims() != 1) {
    return errors::InvalidArgument("box_index must be 1-D",
                                   box_index.shape().DebugString());
  }
  if (box_index.dim_size(0) != *num_boxes) {
    return errors::InvalidArgument("box_index has incompatible shape");
  }
  return Status::OK();
}

// Conditionally calls the compute callback if all values in box_index are in
// [0, batch_size) then calls done.
template <typename Device>
inline void RunIfBoxIndexIsValid(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done);

// Specialization of CheckValidBoxIndex for a CPUDevice.
template <>
inline void RunIfBoxIndexIsValid<CPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done) {
  const int num_boxes = box_index.dimension(0);
  for (int b = 0; b < num_boxes; ++b) {
    OP_REQUIRES_ASYNC(
        context, FastBoundsCheck(box_index(b), batch_size),
        errors::OutOfRange("box_index has values outside [0, batch_size)"),
        done);
  }
  if (compute) {
    compute();
  }
  if (done) {
    done();
  }
}

}  // namespace

template <typename Device, typename T>
class CropAndResizeV3Op : public AsyncOpKernel {
 public:
  explicit CropAndResizeV3Op(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES(context, method_ == "trilinear" || method_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'trilinear' or 'nearest'", method_));
    OP_REQUIRES_OK(context, context->GetAttr("extrapolation_value",
                                             &extrapolation_value_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'image' is [batch_size, image_depth, image_height, image_width, channels].
    const Tensor& image = context->input(0);
    // The shape of 'boxes' is [num_boxes, 6].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'crop_size' is [3].
    const Tensor& crop_size = context->input(3);

    // Validate inputs dimensions.
    OP_REQUIRES_ASYNC(context, image.dims() == 5,
                      errors::InvalidArgument("input image must be 5-D",
                                              image.shape().DebugString()),
                      done);
    const int batch_size = image.dim_size(0);
    const int image_depth = image.dim_size(1);
    const int image_height = image.dim_size(2);
    const int image_width = image.dim_size(3);
    const int depth = image.dim_size(4);
    OP_REQUIRES_ASYNC(
        context, image_depth > 0 && image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);

    OP_REQUIRES_ASYNC(context, crop_size.dims() == 1,
                      errors::InvalidArgument("crop_size must be 1-D",
                                              crop_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(
        context, crop_size.dim_size(0) == 3,
        errors::InvalidArgument("crop_size must have three elements",
                                crop_size.shape().DebugString()),
        done);

    // Copy and validate crop sizes.
    auto crop_size_vec = crop_size.vec<int32>();
    const int crop_depth = internal::SubtleMustCopy(crop_size_vec(0));
    const int crop_height = internal::SubtleMustCopy(crop_size_vec(1));
    const int crop_width = internal::SubtleMustCopy(crop_size_vec(2));
    OP_REQUIRES_ASYNC(
        context, crop_depth > 0 && crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("crop dimensions must be positive"), done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({num_boxes, crop_depth, crop_height, crop_width, depth}),
            &output),
        done);

    auto compute_callback = [this, context, output]() {
      const Tensor& image = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status = functor::CropAndResizeV3<Device, T>()(
          context, image.tensor<T, 5>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), method_, extrapolation_value_,
          output->tensor<float, 5>());
      if (!status) {
        context->SetStatus(
            errors::Internal("Failed launch CropAndResizeV3Kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }

 private:
  float extrapolation_value_;
  string method_;
};

// Partial specialization of CropAndResizeV3 functor for a CPUDevice.
namespace functor {
        template <typename T>
        struct CropAndResizeV3<CPUDevice, T> {
                bool operator()(const OpKernelContext* context,
                                typename TTypes<T, 5>::ConstTensor image,
                                typename TTypes<float, 2>::ConstTensor boxes,
                                typename TTypes<int32, 1>::ConstTensor box_index,
                                const string& method_name, float extrapolation_value,
                                typename TTypes<float, 5>::Tensor crops) {
                                const int batch_size = image.dimension(0);
                                const int image_depth = image.dimension(1);
                                const int image_height = image.dimension(2);
                                const int image_width = image.dimension(3);

                                const int num_boxes = crops.dimension(0);
                                const int crop_depth = crops.dimension(1);
                                const int crop_height = crops.dimension(2);
                                const int crop_width = crops.dimension(3);
                                const int depth = crops.dimension(4);

				//changed mark
				//auto CropAndResizePerBox = [&](int start_box, int limit_box)
				//for (int b = start_box; b < limit_box; ++b)

				for (int b = 0; b < num_boxes; ++b) {
					const float z1 = boxes(b, 0);
					const float y1 = boxes(b, 1);
					const float x1 = boxes(b, 2);
					const float z2 = boxes(b, 3);
					const float y2 = boxes(b, 4);
					const float x2 = boxes(b, 5);

					const int32 b_in = box_index(b);
					if (!FastBoundsCheck(b_in, batch_size)) {
					  continue;
					}

					const float depth_scale =
					    (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1) : 0;
					const float height_scale =
					    (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
					const float width_scale =
					    (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;


					for (int z = 0; z < crop_depth; ++z) {
						const float in_z = (crop_depth > 1)
								 ? z1 * (image_depth - 1) + z * depth_scale
								 : 0.5 * (z1 + z2) * (image_depth - 1);

						if (in_z < 0 || in_z > image_depth - 1) {
							for (int y = 0; y < crop_height; ++y) {
								for (int x = 0; x < crop_width; ++x) {
									for (int d = 0; d < depth; ++d) {
										crops(b, z, y, x, d) = extrapolation_value;
									}
								}
							}
							continue;
						}

						const int front_z_index = floorf(in_z);
						const int back_z_index = ceilf(in_z);
						const float z_lerp = in_z - front_z_index;

						for (int y = 0; y < crop_height; ++y) {
							const float in_y = (crop_height > 1)
									 ? y1 * (image_height - 1) + y * height_scale
									 : 0.5 * (y1 + y2) * (image_height - 1);

							if (in_y < 0 || in_y > image_height - 1) {
								for (int x = 0; x < crop_width; ++x) {
									for (int d = 0; d < depth; ++d) {
										crops(b, z, y, x, d) = extrapolation_value;
									}
								}
								continue;
							}

							const int top_y_index = floorf(in_y);
							const int bottom_y_index = ceilf(in_y);
							const float y_lerp = in_y - top_y_index;

							for (int x = 0; x < crop_width; ++x) {
								const float in_x = (crop_width > 1)
									? x1 * (image_width - 1) + x * width_scale
									: 0.5 * (x1 + x2) * (image_width - 1);
								if (in_x < 0 || in_x > image_width - 1) {
									for (int d = 0; d < depth; ++d) {
									    crops(b, z, y, x, d) = extrapolation_value;
									}
									continue;
								}
								const int left_x_index = floorf(in_x);
								const int right_x_index = ceilf(in_x);
								const float x_lerp = in_x - left_x_index;

								for (int d = 0; d < depth; ++d) {
									const float left_top_front(static_cast<float>(image(b_in, front_z_index, top_y_index, left_x_index, d)));
									const float right_top_front(static_cast<float>(image(b_in, front_z_index, top_y_index, right_x_index, d)));
									const float left_bottom_front(static_cast<float>(image(b_in, front_z_index, bottom_y_index, left_x_index, d)));
									const float right_bottom_front(static_cast<float>(image(b_in, front_z_index, bottom_y_index, right_x_index, d)));

								        const float left_top_back(static_cast<float>(image(b_in, back_z_index, top_y_index, left_x_index, d)));
									const float right_top_back(static_cast<float>(image(b_in, back_z_index, top_y_index, right_x_index, d)));
									const float left_bottom_back(static_cast<float>(image(b_in, back_z_index, bottom_y_index, left_x_index, d)));
									const float right_bottom_back(static_cast<float>(image(b_in, back_z_index, bottom_y_index, right_x_index, d)));

									const float top_front = left_top_front * (1 - x_lerp) + right_top_front * x_lerp;
									const float bottom_front = left_bottom_front * (1 - x_lerp) + right_bottom_front * x_lerp;
									const float top_back = left_top_back * (1 - x_lerp) + right_top_back * x_lerp;
									const float bottom_back = left_bottom_back * (1 - x_lerp) + right_bottom_back * x_lerp;

									const float front = top_front * (1 - y_lerp) + bottom_front * y_lerp;
									const float back = top_back * (1 - y_lerp) + bottom_back * y_lerp;

									crops(b, z, y, x, d) = front * (1 - z_lerp) + back * z_lerp;
								}
							}
						}
					}
			    };
			    return true;
			}
		};
}  // namespace functor

template <typename Device, typename T>
class CropAndResizeV3GradImageOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeV3GradImageOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("method", &method_));
    OP_REQUIRES(context, method_ == "trilinear" || method_ == "nearest",
                errors::InvalidArgument(
                    "method must be 'trilinear' or 'nearest'", method_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'grads' is [num_boxes, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 6].
    const Tensor& boxes = context->input(1);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(2);
    // The shape of 'image_size' is [5].
    const Tensor& image_size = context->input(3);

    // Validate input shapes.
    OP_REQUIRES_ASYNC(context, grads.dims() == 5,
                      errors::InvalidArgument("grads image must be 5-D",
                                              grads.shape().DebugString()),
                      done);
    const int crop_depth = grads.dim_size(1);
    const int crop_height = grads.dim_size(2);
    const int crop_width = grads.dim_size(3);
    OP_REQUIRES_ASYNC(
        context, crop_depth > 0 && crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("grads dimensions must be positive"), done);
    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);
    OP_REQUIRES_ASYNC(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"),
        done);

    OP_REQUIRES_ASYNC(context, image_size.dims() == 1,
                      errors::InvalidArgument("image_size must be 1-D",
                                              image_size.shape().DebugString()),
                      done);
    OP_REQUIRES_ASYNC(context, image_size.dim_size(0) == 5,
                      errors::InvalidArgument("image_size must have 5 elements",
                                              image_size.shape().DebugString()),
                      done);
    auto image_size_vec = image_size.vec<int32>();
    const int batch_size = internal::SubtleMustCopy(image_size_vec(0));
    const int image_depth = internal::SubtleMustCopy(image_size_vec(1));
    const int image_height = internal::SubtleMustCopy(image_size_vec(2));
    const int image_width = internal::SubtleMustCopy(image_size_vec(3));
    const int depth = internal::SubtleMustCopy(image_size_vec(4));
    OP_REQUIRES_ASYNC(
        context, image_depth > 0 && image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    OP_REQUIRES_ASYNC(
        context, grads.dim_size(4) == depth,
        errors::InvalidArgument("image_size and grads are incompatible"), done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, image_depth, image_height, image_width, depth}),
            &output),
        done);

    auto compute_callback = [this, context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& boxes = context->input(1);
      const Tensor& box_index = context->input(2);
      const bool status = functor::CropAndResizeV3BackpropImage<Device, T>()(
          context->eigen_device<Device>(), grads.tensor<float, 5>(),
          boxes.tensor<float, 2>(), box_index.tensor<int32, 1>(),
          output->tensor<T, 5>(), method_);
      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed launch CropAndResizeV3BackpropImage kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }

 private:
  string method_;
};

// Partial specialization of CropAndResizeV3BackpropImage functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeV3BackpropImage<CPUDevice, T> {
  bool operator()(const CPUDevice& d,
                  typename TTypes<float, 5>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  typename TTypes<T, 5>::Tensor grads_image,
                  const string& method_name) {
    const int batch_size = grads_image.dimension(0);
    const int image_depth = grads_image.dimension(1);
    const int image_height = grads_image.dimension(2);
    const int image_width = grads_image.dimension(3);

    const int num_boxes = grads.dimension(0);
    const int crop_depth = grads.dimension(1);
    const int crop_height = grads.dimension(2);
    const int crop_width = grads.dimension(3);
    const int depth = grads.dimension(4);

    grads_image.setZero();

    for (int b = 0; b < num_boxes; ++b) {
	const float z1 = boxes(b, 0);
	const float y1 = boxes(b, 1);
	const float x1 = boxes(b, 2);
	const float z2 = boxes(b, 3);
	const float y2 = boxes(b, 4);
	const float x2 = boxes(b, 5);

	const int32 b_in = box_index(b);
	if (!FastBoundsCheck(b_in, batch_size)) {
	continue;
	}

	const float depth_scale = (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1) : 0;
	const float height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
	const float width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

	for (int z = 0; z < crop_depth; ++z) {
		
		const float in_z = (crop_depth > 1)
		                       ? z1 * (image_depth - 1) + z * depth_scale
		                       : 0.5 * (z1 + z2) * (image_depth - 1);
		
		if (in_z < 0 || in_z > image_depth - 1) {
		  continue;
		}
		const int front_z_index = floorf(in_z);
		const int back_z_index = ceilf(in_z);
		const float z_lerp = in_z - front_z_index;

		for (int y = 0; y < crop_height; ++y) {
		        
		        
		        const float in_y = (crop_height > 1)
		                               ? y1 * (image_height - 1) + y * height_scale
		                               : 0.5 * (y1 + y2) * (image_height - 1);
		        
		        if (in_y < 0 || in_y > image_height - 1) {
		          continue;
		        }
		        const int top_y_index = floorf(in_y);
		        const int bottom_y_index = ceilf(in_y);
		        const float y_lerp = in_y - top_y_index;
		
		        for (int x = 0; x < crop_width; ++x) {
		                const float in_x = (crop_width > 1)
		                                     ? x1 * (image_width - 1) + x * width_scale
		                                     : 0.5 * (x1 + x2) * (image_width - 1);
		                if (in_x < 0 || in_x > image_width - 1) {
		                        continue;
		                }
		                if (method_name == "trilinear") {
		                        const int left_x_index = floorf(in_x);
		                        const int right_x_index = ceilf(in_x);
		                        const float x_lerp = in_x - left_x_index;
		                        
		                        for (int d = 0; d < depth; ++d) {
		                                
		                                const float dfront = (1 - z_lerp) * grads(b, z, y, x, d);

		                                grads_image(b_in, front_z_index, top_y_index, left_x_index, d) += static_cast<T>((1 - y_lerp) * (1 - x_lerp) * dfront);
		                                grads_image(b_in, front_z_index, top_y_index, right_x_index, d) += static_cast<T>((1 - y_lerp) * x_lerp * dfront);
		                                grads_image(b_in, front_z_index, bottom_y_index, left_x_index, d) += static_cast<T>(y_lerp * (1 - x_lerp) * dfront);
		                                grads_image(b_in, front_z_index, bottom_y_index, right_x_index, d) += static_cast<T>(y_lerp * x_lerp * dfront);


		                                const float dback = z_lerp * grads(b, z, y, x, d);

		                                grads_image(b_in, back_z_index, top_y_index, left_x_index, d) += static_cast<T>((1 - y_lerp) * (1 - x_lerp) * dback);
		                                grads_image(b_in, back_z_index, top_y_index, right_x_index, d) += static_cast<T>((1 - y_lerp) * x_lerp * dback);
		                                grads_image(b_in, back_z_index, bottom_y_index, left_x_index, d) += static_cast<T>(y_lerp * (1 - x_lerp) * dback);
		                                grads_image(b_in, back_z_index, bottom_y_index, right_x_index, d) += static_cast<T>(y_lerp * x_lerp * dback);
		                        }
		                }
		                else {  // method_name == "nearest"
		                      for (int d = 0; d < depth; ++d) {
		                          int closest_x_index = roundf(in_x);
		                          int closest_y_index = roundf(in_y);
		                          int closest_z_index = roundf(in_z);
		                          grads_image(b_in, closest_z_index, closest_y_index, closest_x_index, d) +=
		                              static_cast<T>(grads(b, z, y, x, d));
		                      }
		                }
		        }
		}
	}



    }
    return true;
  }
};

}  // namespace functor

template <typename Device, typename T>
class CropAndResizeV3GradBoxesOp : public AsyncOpKernel {
 public:
  explicit CropAndResizeV3GradBoxesOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    string method;
    OP_REQUIRES_OK(context, context->GetAttr("method", &method));
    OP_REQUIRES(context, method == "trilinear",
                errors::InvalidArgument("method must be 'trilinear'", method));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
    // The shape of 'grads' is [num_boxes, crop_depth, crop_height, crop_width, depth].
    const Tensor& grads = context->input(0);
    // The shape of 'boxes' is [num_boxes, 6].
    const Tensor& boxes = context->input(2);
    // The shape of 'box_index' is [num_boxes].
    const Tensor& box_index = context->input(3);
    // The shape of 'image' is [batch_size, image_depth, image_height, image_width, depth].
    const Tensor& image = context->input(1);

    // Validate input shapes.
    OP_REQUIRES_ASYNC(context, grads.dims() == 5,
                      errors::InvalidArgument("grads image must be 5-D",
                                              grads.shape().DebugString()),
                      done);
    const int crop_depth = grads.dim_size(1);
    const int crop_height = grads.dim_size(2);
    const int crop_width = grads.dim_size(3);
    const int depth = grads.dim_size(4);
    OP_REQUIRES_ASYNC(
        context, crop_depth > 0 && crop_height > 0 && crop_width > 0,
        errors::InvalidArgument("grads dimensions must be positive"), done);

    OP_REQUIRES_ASYNC(context, image.dims() == 5,
                      errors::InvalidArgument("input image must be 5-D",
                                              image.shape().DebugString()),
                      done);
    const int batch_size = image.dim_size(0);
    const int image_depth = image.dim_size(1);
    const int image_height = image.dim_size(2);
    const int image_width = image.dim_size(3);
    OP_REQUIRES_ASYNC(
        context, image_depth > 0 && image_height > 0 && image_width > 0,
        errors::InvalidArgument("image dimensions must be positive"), done);
    OP_REQUIRES_ASYNC(context, image.dim_size(4) == depth,
                      errors::InvalidArgument("image, grads depth differ"),
                      done);

    int num_boxes = 0;
    OP_REQUIRES_OK_ASYNC(
        context, ParseAndCheckBoxSizes(boxes, box_index, &num_boxes), done);

    OP_REQUIRES_ASYNC(
        context, grads.dim_size(0) == num_boxes,
        errors::InvalidArgument("boxes and grads have incompatible shape"),
        done);

    // Allocate output tensor.
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_output(0, TensorShape({num_boxes, 6}), &output),
        done);

    auto compute_callback = [context, output]() {
      const Tensor& grads = context->input(0);
      const Tensor& image = context->input(1);
      const Tensor& boxes = context->input(2);
      const Tensor& box_index = context->input(3);
      const bool status = functor::CropAndResizeV3BackpropBoxes<Device, T>()(
          context->eigen_device<Device>(), grads.tensor<float, 5>(),
          image.tensor<T, 5>(), boxes.tensor<float, 2>(),
          box_index.tensor<int32, 1>(), output->tensor<float, 2>());
      if (!status) {
        context->SetStatus(errors::Internal(
            "Failed launch CropAndResizeV3BackpropBoxes kernel."));
      }
    };

    RunIfBoxIndexIsValid<Device>(context, box_index.tensor<int32, 1>(),
                                 batch_size, std::move(compute_callback),
                                 std::move(done));
  }
};

// Partial specialization of CropAndResizeV3BackpropBoxes functor for a CPUDevice.
namespace functor {
template <typename T>
struct CropAndResizeV3BackpropBoxes<CPUDevice, T> {
  bool operator()(const CPUDevice& d,
                  typename TTypes<float, 5>::ConstTensor grads,
                  typename TTypes<T, 5>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_index,
                  typename TTypes<float, 2>::Tensor grads_boxes) {
               const int batch_size = image.dimension(0);
               const int image_depth = image.dimension(1);
               const int image_height = image.dimension(2);
               const int image_width = image.dimension(3);
               
               const int num_boxes = grads.dimension(0);
               const int crop_depth = grads.dimension(1);
               const int crop_height = grads.dimension(2);
               const int crop_width = grads.dimension(3);
               const int depth = grads.dimension(4);
               
               grads_boxes.setZero();
               
               for (int b = 0; b < num_boxes; ++b) {
                     const float z1 = boxes(b, 0);
                     const float y1 = boxes(b, 1);
                     const float x1 = boxes(b, 2);
                     const float z2 = boxes(b, 3);
                     const float y2 = boxes(b, 4);
                     const float x2 = boxes(b, 5);
                     
                     const int32 b_in = box_index(b);
                     if (!FastBoundsCheck(b_in, batch_size)) {
                             continue;
                     }
                     
                     const float depth_ratio = (crop_depth > 1) ? static_cast<float>(image_depth - 1) / (crop_depth - 1) : 0;
                     const float height_ratio = (crop_height > 1) ? static_cast<float>(image_height - 1) / (crop_height - 1) : 0;
                     const float width_ratio = (crop_width > 1) ? static_cast<float>(image_width - 1) / (crop_width - 1) : 0;
                     
                     const float depth_scale = (crop_depth > 1) ? (z2 - z1) * depth_ratio : 0;
                     const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
                     const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;

                    	for (int z = 0; z < crop_depth; ++z) {
                    		
                    		const float in_z = (crop_depth > 1)
                    		                       ? z1 * (image_depth - 1) + z * depth_scale
                    		                       : 0.5 * (z1 + z2) * (image_depth - 1);
                    		
                    		if (in_z < 0 || in_z > image_depth - 1) {
                    		  continue;
                    		}
                    		const int front_z_index = floorf(in_z);
                    		const int back_z_index = ceilf(in_z);
                    		const float z_lerp = in_z - front_z_index;
                    
                    		for (int y = 0; y < crop_height; ++y) {
                    		        
                    		        
                    		        const float in_y = (crop_height > 1)
                    		                               ? y1 * (image_height - 1) + y * height_scale
                    		                               : 0.5 * (y1 + y2) * (image_height - 1);
                    		        
                    		        if (in_y < 0 || in_y > image_height - 1) {
                    		          continue;
                    		        }
                    		        const int top_y_index = floorf(in_y);
                    		        const int bottom_y_index = ceilf(in_y);
                    		        const float y_lerp = in_y - top_y_index;
                    		
                    		        for (int x = 0; x < crop_width; ++x) {
                    		                const float in_x = (crop_width > 1)
                    		                                     ? x1 * (image_width - 1) + x * width_scale
                    		                                     : 0.5 * (x1 + x2) * (image_width - 1);
                    		                if (in_x < 0 || in_x > image_width - 1) {
                    		                        continue;
                    		                }

        					const int left_x_index = floorf(in_x);
						const int right_x_index = ceilf(in_x);
						const float x_lerp = in_x - left_x_index;
          		                        
            		                        for (int d = 0; d < depth; ++d) {
                                                const float left_top_front(static_cast<float>(image(b_in, front_z_index, top_y_index, left_x_index, d)));
                                                const float right_top_front(static_cast<float>(image(b_in, front_z_index, top_y_index, right_x_index, d)));
                                                const float left_bottom_front(static_cast<float>(image(b_in, front_z_index, bottom_y_index, left_x_index, d)));
                                                const float right_bottom_front(static_cast<float>(image(b_in, front_z_index, bottom_y_index, right_x_index, d)));
                                                const float left_top_back(static_cast<float>(image(b_in, back_z_index, top_y_index, left_x_index, d)));
                                                const float right_top_back(static_cast<float>(image(b_in, back_z_index, top_y_index, right_x_index, d)));
                                                const float left_bottom_back(static_cast<float>(image(b_in, back_z_index, bottom_y_index, left_x_index, d)));
                                                const float right_bottom_back(static_cast<float>(image(b_in, back_z_index, bottom_y_index, right_x_index, d)));
                                                
                                                // Compute the image gradient.
                                                float image_grad_z = (1- y_lerp) * (1 - x_lerp) * (left_top_back - left_top_front) + 
                                                                      (1 - y_lerp) * x_lerp * (right_top_back - right_top_front) + 
                                                                      y_lerp * (1 - x_lerp) * (left_bottom_back - left_bottom_front) + 
                                                                      y_lerp * x_lerp * (right_bottom_back - right_bottom_front);
                                                
                                                float image_grad_y = (1 - z_lerp) * (1 - x_lerp) * (left_bottom_front - left_top_front) + 
                                                                     (1 - z_lerp) * x_lerp * (right_bottom_front - right_top_front) +
                                                                     z_lerp * (1 - x_lerp) * (left_bottom_back - left_top_back) + 
                                                                     z_lerp * x_lerp * (right_bottom_back - right_top_back);
                                                
                                                float image_grad_x = (1 - z_lerp) * (1 - y_lerp) * (right_top_front - left_top_front) + 
                                                                     (1 - z_lerp) * y_lerp * (right_bottom_front - left_bottom_front) + 
                                                                     z_lerp * (1 - y_lerp) * (right_top_back - left_top_back) + 
                                                                     z_lerp * y_lerp * (right_bottom_back - left_bottom_back);

                                                // Modulate the image gradient with the incoming gradient.
                                                const float top_grad = grads(b, z, y, x, d);
                                                image_grad_z *= top_grad;
                                                image_grad_y *= top_grad;
                                                image_grad_x *= top_grad;

                                                if (crop_depth > 1) {
                                                        grads_boxes(b, 0) += image_grad_z * (image_depth - 1 - z * depth_ratio);
                                                        grads_boxes(b, 3) += image_grad_z * (z * depth_ratio);
                                                }
                                                else {
                                                        grads_boxes(b, 0) += image_grad_z * 0.5 * (image_depth - 1);
                                                        grads_boxes(b, 3) += image_grad_z * 0.5 * (image_depth - 1);
                                                }

                                                if (crop_height > 1) {
                                                        grads_boxes(b, 1) += image_grad_y * (image_height - 1 - y * height_ratio);
                                                        grads_boxes(b, 4) += image_grad_y * (y * height_ratio);
                                                }
                                                else {
                                                        grads_boxes(b, 1) += image_grad_y * 0.5 * (image_height - 1);
                                                        grads_boxes(b, 4) += image_grad_y * 0.5 * (image_height - 1);
                                                }                                                

                                                if (crop_width > 1) {
                                                        grads_boxes(b, 2) += image_grad_x * (image_width - 1 - x * width_ratio);
                                                        grads_boxes(b, 5) += image_grad_x * (x * width_ratio);
                                                }
                                                else {
                                                        grads_boxes(b, 2) += image_grad_x * 0.5 * (image_width - 1);
                                                        grads_boxes(b, 5) += image_grad_x * 0.5 * (image_width - 1);
                                                }
            		                        }
                    		        }
                            }
                    	}                 
            }
            return true;
    }
  };
}  // namespace functor

#define REGISTER_KERNEL(T)                                \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeV3")           \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("crop_size"),   \
                          CropAndResizeV3Op<CPUDevice, T>); \
                                                          \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeV3GradBoxes")  \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T"),    \
                          CropAndResizeV3GradBoxesOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_KERNEL(T)                               \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeV3GradImage") \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("image_size"), \
                          CropAndResizeV3GradImageOp<CPUDevice, T>);

TF_CALL_half(REGISTER_KERNEL);
TF_CALL_float(REGISTER_KERNEL);
TF_CALL_double(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA

// Forward declaration of the CheckValidBoxIndexHelper specialization for GPU.
namespace functor {
template <>
void CheckValidBoxIndexHelper<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, typename TTypes<bool, 0>::Tensor isvalid);
extern template struct CheckValidBoxIndexHelper<GPUDevice>;
}  // namespace functor

namespace {

// Specialization of CheckValidBoxIndex for a GPUDevice.
template <>
inline void RunIfBoxIndexIsValid<GPUDevice>(
    OpKernelContext* context, typename TTypes<int32, 1>::ConstTensor box_index,
    int batch_size, const Callback& compute, const Callback& done) {
  const int num_boxes = box_index.dimension(0);
  if (num_boxes == 0) {
    compute();
    done();
    return;
  }

  Tensor isvalid_dev_tensor;
  OP_REQUIRES_OK_ASYNC(
      context,
      context->allocate_temp(DataTypeToEnum<bool>::value, TensorShape({}),
                             &isvalid_dev_tensor),
      done);
  typename TTypes<bool, 0>::Tensor isvalid_dev =
      isvalid_dev_tensor.tensor<bool, 0>();

  // Run the actual box check on the device.
  functor::CheckValidBoxIndexHelper<GPUDevice>()(
      context->eigen_device<GPUDevice>(), box_index, batch_size, isvalid_dev);

  // Copy the result back to the host.
  auto* stream = context->op_device_context()->stream();
  OP_REQUIRES_ASYNC(context, stream,
                    errors::Internal("No GPU stream available."), done);
  Tensor isvalid_host_tensor;
  // Use pinned host memory on the host to avoid unnecessary
  // synchronization.
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  alloc_attr.set_gpu_compatible(true);
  OP_REQUIRES_OK_ASYNC(
      context,
      context->allocate_temp(DataTypeToEnum<bool>::value, TensorShape({}),
                             &isvalid_host_tensor, alloc_attr),
      done);
  se::DeviceMemoryBase wrapped(isvalid_dev.data(), sizeof(bool));
  const bool status =
      stream
          ->ThenMemcpy(
              isvalid_host_tensor.scalar<bool>().data() /* destination */,
              wrapped /* source */, sizeof(bool))
          .ok();
  OP_REQUIRES_ASYNC(
      context, status,
      errors::Internal("Failed to launch copy of isvalid from device to host."),
      done);

  // We capture both temporary tensors to prevent them from being deallocated
  // when ComputeAsync returns and before the closure runs.
  TensorReference isvalid_dev_ref(isvalid_dev_tensor);
  auto wrapped_callback = [context, isvalid_host_tensor, isvalid_dev_ref,
                           compute, done]() {
    auto stream = context->op_device_context()->stream();
    ScopedActivateExecutorContext scoped_activation{stream->parent()};
    const bool isvalid = isvalid_host_tensor.scalar<bool>()();
    isvalid_dev_ref.Unref();
    OP_REQUIRES_ASYNC(
        context, isvalid,
        errors::OutOfRange("box_index has values outside [0, batch_size)"),
        done);
    compute();
    done();
  };

  context->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
      stream, wrapped_callback);
}

}  // namespace

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeV3")                    \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("crop_size"),            \
                          CropAndResizeV3Op<GPUDevice, T>);          \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeV3GradImage")           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T")              \
                              .HostMemory("image_size"),           \
                          CropAndResizeV3GradImageOp<GPUDevice, T>); \
                                                                   \
  REGISTER_KERNEL_BUILDER(Name("CropAndResizeV3GradBoxes")           \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          CropAndResizeV3GradBoxesOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#endif  // GOOGLE_CUDA

} // namespace tensorflow
