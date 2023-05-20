#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "crop_and_resize_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

enum InterpolationMethod {
  TRILINEAR = 0,
  NEAREST = 1,
};

template <typename T>
__global__ void CropAndResizeV3Kernel(
    const int32 nthreads, const T* image_ptr, const float* boxes_ptr,
    const int32* box_ind_ptr, int num_boxes, int batch, 
    int image_depth, int image_height, int image_width, 
    int crop_depth, int crop_height, int crop_width, 
    int depth, int method_id, float extrapolation_value, float* crops_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = dd + depth * (w + crop_width * (h + crop_height * (d + crop_depth * b)))
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    idx /= crop_height;
    const int z = idx % crop_depth;
    const int b = idx / crop_depth;

    const float z1 = boxes_ptr[b * 6];
    const float y1 = boxes_ptr[b * 6 + 1];
    const float x1 = boxes_ptr[b * 6 + 2];
    const float z2 = boxes_ptr[b * 6 + 3];
    const float y2 = boxes_ptr[b * 6 + 4];
    const float x2 = boxes_ptr[b * 6 + 5];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      continue;
    }

    const float depth_scale = (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1) : 0;
    const float height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    const float in_z = (crop_depth > 1) ? z1 * (image_depth - 1) + z * depth_scale : 0.5 * (z1 + z2) * (image_depth - 1);
    if (in_z < 0 || in_z > image_depth - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    const float in_y = (crop_height > 1) ? y1 * (image_height - 1) + y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    const float in_x = (crop_width > 1) ? x1 * (image_width - 1) + x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    if (method_id == TRILINEAR) {

      const int front_z_index = floorf(in_z);
      const int back_z_index = ceilf(in_z);
      const float z_lerp = in_z - front_z_index;

      const int top_y_index = floorf(in_y);
      const int bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;

      const int left_x_index = floorf(in_x);
      const int right_x_index = ceilf(in_x);
      const float x_lerp = in_x - left_x_index;

      const float front_top_left(static_cast<float>(image_ptr[((b_in * image_depth + front_z_index) * image_height + top_y_index) * image_width + left_x_index) * depth + d]));
      const float front_top_right(static_cast<float>(image_ptr[((b_in * image_depth + front_z_index) * image_height + top_y_index) * image_width + right_x_index) * depth + d]));
      const float front_bottom_left(static_cast<float>(image_ptr[((b_in * image_depth + front_z_index) * image_height + bottom_y_index) * image_width + left_x_index) * depth + d]));
      const float front_bottom_right(static_cast<float>(image_ptr[((b_in * image_depth + front_z_index) * image_height + bottom_y_index) * image_width + right_x_index) * depth + d]));

      const float back_top_left(static_cast<float>(image_ptr[((b_in * image_depth + back_z_index) * image_height + top_y_index) * image_width + left_x_index) * depth + d]));
      const float back_top_right(static_cast<float>(image_ptr[((b_in * image_depth + back_z_index) * image_height + top_y_index) * image_width + right_x_index) * depth + d]));
      const float back_bottom_left(static_cast<float>(image_ptr[((b_in * image_depth + back_z_index) * image_height + bottom_y_index) * image_width + left_x_index) * depth + d]));
      const float back_bottom_right(static_cast<float>(image_ptr[((b_in * image_depth + back_z_index) * image_height + bottom_y_index) * image_width + right_x_index) * depth + d]));

      // const float top = top_left + (top_right - top_left) * x_lerp;
      // const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
      // crops_ptr[out_idx] = top + (bottom - top) * y_lerp;

      const float front_top = front_top_left * (1 - x_lerp) + front_top_right * x_lerp;
      const float front_bottom = front_bottom_left * (1 - x_lerp) + front_bottom_right * x_lerp;
      const float back_top = back_top_left * (1 - x_lerp) + back_top_right * x_lerp;
      const float back_bottom = back_bottom_left * (1 - x_lerp) + back_bottom_right * x_lerp;

      const float front = front_top * (1 - y_lerp) + front_bottom * y_lerp;
      const float back = back_top * (1 - y_lerp) + back_bottom * y_lerp;

      crops_ptr[out_idx] = front * (1 - z_lerp) + back * z_lerp;

    } else {  // method_id == kMethodNearestId
      const int closest_x_index = roundf(in_x);
      const int closest_y_index = roundf(in_y);
      const int closest_z_index = roundf(in_z);
      crops_ptr[out_idx] = static_cast<float>(
          image_ptr[(((b_in * image_depth + closest_z_index) * image_height + closest_y_index) * image_width +
                     closest_x_index) *
                        depth +
                    d]);
    }
  }
}

template <typename T>
__global__ void CropAndResizeV3BackpropImageKernel(
    const int32 nthreads, const float* grads_ptr, const float* boxes_ptr,
    const int32* box_ind_ptr, int num_boxes, int batch, 
    int image_depth, int image_height, int image_width, 
    int crop_depth, int crop_height, int crop_width, 
    int depth, T* grads_image_ptr, int method_id) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {



    // out_idx = dd + depth * (w + crop_width * (h + crop_height * (d + crop_depth * b)))
    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    idx /= crop_height;
    const int z = idx % crop_depth;
    const int b = idx / crop_depth;

    const float z1 = boxes_ptr[b * 6];
    const float y1 = boxes_ptr[b * 6 + 1];
    const float x1 = boxes_ptr[b * 6 + 2];
    const float z2 = boxes_ptr[b * 6 + 3];
    const float y2 = boxes_ptr[b * 6 + 4];
    const float x2 = boxes_ptr[b * 6 + 5];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      continue;
    }

    const float depth_scale = (crop_depth > 1) ? (z2 - z1) * (image_depth - 1) / (crop_depth - 1) : 0;
    const float height_scale = (crop_height > 1) ? (y2 - y1) * (image_height - 1) / (crop_height - 1) : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * (image_width - 1) / (crop_width - 1) : 0;

    const float in_z = (crop_depth > 1) ? z1 * (image_depth - 1) + z * depth_scale : 0.5 * (z1 + z2) * (image_depth - 1);
    if (in_z < 0 || in_z > image_depth - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    const float in_y = (crop_height > 1) ? y1 * (image_height - 1) + y * height_scale : 0.5 * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    const float in_x = (crop_width > 1) ? x1 * (image_width - 1) + x * width_scale : 0.5 * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      crops_ptr[out_idx] = extrapolation_value;
      continue;
    }

    if (method_id == TRILINEAR) {
      const int front_z_index = floorf(in_z);
      const int back_z_index = ceilf(in_z);
      const float z_lerp = in_z - front_z_index;

      const int top_y_index = floorf(in_y);
      const int bottom_y_index = ceilf(in_y);
      const float y_lerp = in_y - top_y_index;

      const int left_x_index = floorf(in_x);
      const int right_x_index = ceilf(in_x);
      const float x_lerp = in_x - left_x_index;

      const float dfront = (1 - z_lerp) * grads_ptr[out_idx];


      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + front_z_index) * image_height + top_y_index) * image_width + left_x_index) * depth + d,
                    static_cast<T>((1 - y_lerp) * (1 - x_lerp) * dfront));

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + front_z_index) * image_height + top_y_index) * image_width + right_x_index) * depth + d,
                    static_cast<T>((1 - y_lerp) * x_lerp * dfront));

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + front_z_index) * image_height + bottom_y_index) * image_width + left_x_index) * depth + d,
                    static_cast<T>(y_lerp * (1 - x_lerp) * dfront));

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + front_z_index) * image_height + bottom_y_index) * image_width + right_x_index) * depth + d,
                    static_cast<T>(y_lerp * x_lerp * dfront));

      const float dback = z_lerp * grads_ptr[out_idx];

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + back_z_index) * image_height + top_y_index) * image_width + left_x_index) * depth + d,
                    static_cast<T>((1 - y_lerp) * (1 - x_lerp) * dback));

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + back_z_index) * image_height + top_y_index) * image_width + right_x_index) * depth + d,
                    static_cast<T>((1 - y_lerp) * x_lerp * dback));

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + back_z_index) * image_height + bottom_y_index) * image_width + left_x_index) * depth + d,
                    static_cast<T>(y_lerp * (1 - x_lerp) * dback));

      CudaAtomicAdd(grads_image_ptr + (((b_in * image_depth + back_z_index) * image_height + bottom_y_index) * image_width + right_x_index) * depth + d,
                    static_cast<T>(y_lerp * x_lerp * dback));

    } else {  // method_id == NEAREST
      const int closest_x_index = roundf(in_x);
      const int closest_y_index = roundf(in_y);
      const int closest_z_index = roundf(in_z);
      CudaAtomicAdd(grads_image_ptr +
                        (((b_in * image_depth + closest_z_index) * image_height + closest_y_index) * image_width + closest_x_index) * depth + d,
                    static_cast<T>(grads_ptr[out_idx]));
    }
  }
}

template <typename T>
__global__ void CropAndResizeV3BackpropBoxesKernel(
    const int32 nthreads, const float* grads_ptr, const T* image_ptr,
    const float* boxes_ptr, const int32* box_ind_ptr, int num_boxes, int batch,
    int image_depth, int image_height, int image_width, int crop_depth, int crop_height, int crop_width,
    int depth, float* grads_boxes_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {

    int idx = out_idx;
    const int d = idx % depth;
    idx /= depth;
    const int x = idx % crop_width;
    idx /= crop_width;
    const int y = idx % crop_height;
    idx /= crop_height;
    const int z = idx % crop_depth;
    const int b = idx / crop_depth;

    const float z1 = boxes_ptr[b * 6];
    const float y1 = boxes_ptr[b * 6 + 1];
    const float x1 = boxes_ptr[b * 6 + 2];
    const float z2 = boxes_ptr[b * 6 + 3];
    const float y2 = boxes_ptr[b * 6 + 4];
    const float x2 = boxes_ptr[b * 6 + 5];

    const int32 b_in = box_ind_ptr[b];
    if (b_in < 0 || b_in >= batch) {
      continue;
    }

    const float depth_ratio =
        (crop_depth > 1)
            ? static_cast<float>(image_depth - 1) / (crop_depth - 1)
            : 0;
    const float height_ratio =
        (crop_height > 1)
            ? static_cast<float>(image_height - 1) / (crop_height - 1)
            : 0;
    const float width_ratio =
        (crop_width > 1)
            ? static_cast<float>(image_width - 1) / (crop_width - 1)
            : 0;

    const float depth_scale = (crop_depth > 1) ? (z2 - z1) * depth_ratio : 0;
    const float height_scale = (crop_height > 1) ? (y2 - y1) * height_ratio : 0;
    const float width_scale = (crop_width > 1) ? (x2 - x1) * width_ratio : 0;

    const float in_z = (crop_depth > 1)
                           ? z1 * (image_depth - 1) + z * depth_scale
                           : 0.5 * (z1 + z2) * (image_depth - 1);
    if (in_z < 0 || in_z > image_depth - 1) {
      continue;
    }

    const float in_y = (crop_height > 1)
                           ? y1 * (image_height - 1) + y * height_scale
                           : 0.5 * (y1 + y2) * (image_height - 1);
    if (in_y < 0 || in_y > image_height - 1) {
      continue;
    }

    const float in_x = (crop_width > 1)
                           ? x1 * (image_width - 1) + x * width_scale
                           : 0.5 * (x1 + x2) * (image_width - 1);
    if (in_x < 0 || in_x > image_width - 1) {
      continue;
    }

    const int front_z_index = floorf(in_z);
    const int back_z_index = ceilf(in_z);
    const float z_lerp = in_z - top_z_index;

    const int top_y_index = floorf(in_y);
    const int bottom_y_index = ceilf(in_y);
    const float y_lerp = in_y - top_y_index;

    const int left_x_index = floorf(in_x);
    const int right_x_index = ceilf(in_x);
    const float x_lerp = in_x - left_x_index;




    const float left_top_front(static_cast<float>(image_ptr[(((b_in * image_depth + front_z_index) * image_height + top_y_index) * image_width + left_x_index) * depth + d])));
    const float right_top_front(static_cast<float>(image_ptr[(((b_in * image_depth + front_z_index) * image_height + top_y_index) * image_width + right_x_index) * depth + d])));
    const float left_bottom_front(static_cast<float>(image_ptr[(((b_in * image_depth + front_z_index) * image_height + bottom_y_index) * image_width + left_x_index) * depth + d])));
    const float right_bottom_front(static_cast<float>(image_ptr[(((b_in * image_depth + front_z_index) * image_height + bottom_y_index) * image_width + right_x_index) * depth + d])));
    const float left_top_back(static_cast<float>(image_ptr[(((b_in * image_depth + back_z_index) * image_height + top_y_index) * image_width + left_x_index) * depth + d])));
    const float right_top_back(static_cast<float>(image_ptr[(((b_in * image_depth + back_z_index) * image_height + top_y_index) * image_width + right_x_index) * depth + d])));
    const float left_bottom_back(static_cast<float>(image_ptr[(((b_in * image_depth + back_z_index) * image_height + bottom_y_index) * image_width + left_x_index) * depth + d])));
    const float right_bottom_back(static_cast<float>(image_ptr[(((b_in * image_depth + back_z_index) * image_height + bottom_y_index) * image_width + right_x_index) * depth + d])));

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
    const float top_grad = grads_ptr[out_idx];
    image_grad_z *= top_grad;
    image_grad_y *= top_grad;
    image_grad_x *= top_grad;

    float dz1, dz2;
    if (crop_depth > 1) {
      dz1 = image_grad_z * (image_depth - 1 - z * depth_ratio);
      dz2 = image_grad_z * (z * depth_ratio);
    } else {
      dz1 = image_grad_z * 0.5 * (image_depth - 1);
      dz2 = image_grad_z * 0.5 * (image_depth - 1);
    }

    float dy1, dy2;
    if (crop_height > 1) {
      dy1 = image_grad_y * (image_height - 1 - y * height_ratio);
      dy2 = image_grad_y * (y * height_ratio);
    } else {
      dy1 = image_grad_y * 0.5 * (image_height - 1);
      dy2 = image_grad_y * 0.5 * (image_height - 1);
    }

    float dx1, dx2;
    if (crop_width > 1) {
      dx1 = image_grad_x * (image_width - 1 - x * width_ratio);
      dx2 = image_grad_x * (x * width_ratio);
    } else {
      dx1 = image_grad_x * 0.5 * (image_width - 1);
      dx2 = image_grad_x * 0.5 * (image_width - 1);
    }

    CudaAtomicAdd(grads_boxes_ptr + b * 6 + 0, dz1);
    CudaAtomicAdd(grads_boxes_ptr + b * 6 + 1, dy1);
    CudaAtomicAdd(grads_boxes_ptr + b * 6 + 2, dx1);
    CudaAtomicAdd(grads_boxes_ptr + b * 6 + 3, dz2);
    CudaAtomicAdd(grads_boxes_ptr + b * 6 + 4, dy2);
    CudaAtomicAdd(grads_boxes_ptr + b * 6 + 5, dx2);
  }
}

}  // namespace

namespace functor {

template <typename T>
struct CropAndResizeV3<GPUDevice, T> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 5>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  string method_name, float extrapolation_value,
                  typename TTypes<float, 5>::Tensor crops) {
    const int batch = image.dimension(0);
    const int image_depth = image.dimension(1);
    const int image_height = image.dimension(2);
    const int image_width = image.dimension(3);

    const int num_boxes = crops.dimension(0);
    const int crop_depth = crops.dimension(1);
    const int crop_height = crops.dimension(2);
    const int crop_width = crops.dimension(3);
    const int depth = crops.dimension(4);

    const int total_count = num_boxes * crop_depth * crop_height * crop_width * depth;
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    InterpolationMethod method = TRILINEAR;
    if (method_name == "nearest") {
      method = NEAREST;
    }

    if (total_count > 0) {
      CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
      CropAndResizeV3Kernel<<<config.block_count, config.thread_per_block, 0,
                            d.stream()>>>(
          config.virtual_thread_count, image.data(), boxes.data(),
          box_ind.data(), num_boxes, batch, image_depth, image_height, image_width,
          crop_depth, crop_height, crop_width, depth, method, extrapolation_value,
          crops.data());
    }
    return d.ok();
  }
};

template <typename T>
struct CropAndResizeV3BackpropImage<GPUDevice, T> {
  bool operator()(const GPUDevice& d,
                  typename TTypes<float, 5>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 5>::Tensor grads_image,
                  const string& method_name) {
    const int batch = grads_image.dimension(0);
    const int image_depth = grads_image.dimension(1);
    const int image_height = grads_image.dimension(2);
    const int image_width = grads_image.dimension(3);

    const int num_boxes = grads.dimension(0);
    const int crop_depth = grads.dimension(1);
    const int crop_height = grads.dimension(2);
    const int crop_width = grads.dimension(3);
    const int depth = grads.dimension(4);

    int total_count;
    CudaLaunchConfig config;

    // Initialize grads_image with all zeros.
    total_count = batch * image_height * image_width * depth;
    if (total_count > 0) {
      config = GetCudaLaunchConfig(total_count, d);
      SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, grads_image.data());
    }

    // Configurate interpolation method.
    InterpolationMethod method = TRILINEAR;
    if (method_name == "nearest") {
      method = NEAREST;
    }

    // Accumulate.
    total_count = num_boxes * crop_depth * crop_height * crop_width * depth;
    if (total_count > 0) {
      config = GetCudaLaunchConfig(total_count, d);
      CropAndResizeV3BackpropImageKernel<<<
          config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, grads.data(), boxes.data(),
          box_ind.data(), num_boxes, batch, image_depth, image_height, image_width,
          crop_depth, crop_height, crop_width, depth, grads_image.data(), method);
    }
    return d.ok();
  }
};

template <typename T>
struct CropAndResizeV3BackpropBoxes<GPUDevice, T> {
  bool operator()(const GPUDevice& d,
                  typename TTypes<float, 5>::ConstTensor grads,
                  typename TTypes<T, 5>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<float, 2>::Tensor grads_boxes) {
    const int batch = image.dimension(0);
    const int image_depth = image.dimension(1);
    const int image_height = image.dimension(2);
    const int image_width = image.dimension(3);

    const int num_boxes = grads.dimension(0);
    const int crop_depth = grads.dimension(1);
    const int crop_height = grads.dimension(2);
    const int crop_width = grads.dimension(3);
    const int depth = grads.dimension(4);

    int total_count;
    CudaLaunchConfig config;

    // Initialize grads_boxes with all zeros.
    total_count = num_boxes * 6;
    if (total_count > 0) {
      config = GetCudaLaunchConfig(total_count, d);
      SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, grads_boxes.data());
    }

    // Accumulate.
    total_count = num_boxes * crop_depth * crop_height * crop_width * depth;
    if (total_count > 0) {
      config = GetCudaLaunchConfig(total_count, d);
      CropAndResizeV3BackpropBoxesKernel<<<
          config.block_count, config.thread_per_block, 0, d.stream()>>>(
          config.virtual_thread_count, grads.data(), image.data(), boxes.data(),
          box_ind.data(), num_boxes, batch, image_depth, image_height, image_width,
          crop_depth, crop_height, crop_width, depth, grads_boxes.data());
    }
    return d.ok();
  }
};

#define DEFINE_GPU_SPECS(T)                                 \
  template struct CropAndResizeV3<GPUDevice, T>;              \
  template struct CropAndResizeV3BackpropImage<GPUDevice, T>; \
  template struct CropAndResizeV3BackpropBoxes<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

template struct CheckValidBoxIndexHelper<GPUDevice>;

}  // namespace functor
}  // namespace tensorflow

#endif // GOOGLE_CUDA
