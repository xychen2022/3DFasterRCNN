#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "non_max_suppression_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(float const * const a, float const * const b) {
  float left = max(a[0], b[0]), right = min(a[3], b[3]);
  float top = max(a[1], b[1]), bottom = min(a[4], b[4]);
  float front = max(a[2], b[2]), back = min(a[5], b[5]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f), depth = max(back - front + 1, 0.f);
  float interS = width * height * depth;
  float Sa = (a[3] - a[0] + 1) * (a[4] - a[1] + 1) * (a[5] - a[2] + 1);
  float Sb = (b[3] - b[0] + 1) * (b[4] - b[1] + 1) * (b[5] - b[2] + 1);
  return interS / (Sa + Sb - interS);
}

__global__ void NonMaxSuppressionV5Kernel(const float* boxes_ptr, const float* scores_ptr,
                           const int num_boxes, const float iou_threshold,  const float score_threshold,
			   const int max_output_size, const int* selected_indices_ptr) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(num_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(num_boxes - col_start * threadsPerBlock, threadsPerBlock);

  __shared__ float block_boxes[threadsPerBlock * 7];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 7 + 0] =
        boxes_ptr[(threadsPerBlock * col_start + threadIdx.x) * 6 + 0];
    block_boxes[threadIdx.x * 7 + 1] =
        boxes_ptr[(threadsPerBlock * col_start + threadIdx.x) * 6 + 1];
    block_boxes[threadIdx.x * 7 + 2] =
        boxes_ptr[(threadsPerBlock * col_start + threadIdx.x) * 6 + 2];
    block_boxes[threadIdx.x * 7 + 3] =
        boxes_ptr[(threadsPerBlock * col_start + threadIdx.x) * 6 + 3];
    block_boxes[threadIdx.x * 7 + 4] =
        boxes_ptr[(threadsPerBlock * col_start + threadIdx.x) * 6 + 4];
    block_boxes[threadIdx.x * 7 + 5] =
        boxes_ptr[(threadsPerBlock * col_start + threadIdx.x) * 6 + 5];
    block_boxes[threadIdx.x * 7 + 6] =
        scores_ptr[threadsPerBlock * col_start + threadIdx.x];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = boxes_ptr + cur_box_idx * 6;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 7) > iou_threshold) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(num_boxes, threadsPerBlock);
    selected_indices_ptr[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _set_device(int device_id) {
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) {
    return;
  }
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));
}
}

namespace functor {

#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

template <>
struct NonMaxSuppressionV5<GPUDevice> {
  bool operator()(const OpKernelContext* context,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<float, 1>::ConstTensor scores,
                  float iou_threshold, 
                  float score_threshold,
                  int max_output_size,
                  typename TTypes<int, 1>::Tensor selected_indices) {

    const int num_boxes = boxes.dimension(0);
    const int num_scores = scores.dimension(0);

    int const threadsPerBlock = sizeof(unsigned long long) * 8;
    const int col_blocks = DIVUP(num_boxes, threadsPerBlock);
    
    dim3 blocks(DIVUP(num_boxes, threadsPerBlock), DIVUP(num_boxes, threadsPerBlock));
    dim3 threads(threadsPerBlock);

    const GPUDevice& d = context->eigen_device<GPUDevice>();

    NonMaxSuppressionV5Kernel<<<blocks, threads, 0, d.stream()>>>(
          boxes.data(), scores.data(), num_boxes, iou_threshold,
          score_threshold, max_output_size, selected_indices.data()); // selected_indices ??
/*
    std::vector<unsigned long long> mask_host(num_boxes * col_blocks);
    CUDA_CHECK(cudaMemcpy(&mask_host[0],
                          mask_dev, // mask_dev ??
                          sizeof(unsigned long long) * num_boxes * col_blocks,
                          cudaMemcpyDeviceToHost));

    std::vector<unsigned long long> remv(col_blocks);
    memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

    int num_to_keep = 0;
    for (int i = 0; i < num_boxes; i++) {
      int nblock = i / threadsPerBlock;
      int inblock = i % threadsPerBlock;

      if (!(remv[nblock] & (1ULL << inblock))) {
        keep_out[num_to_keep++] = i;
        unsigned long long *p = &mask_host[0] + i * col_blocks;
        for (int j = nblock; j < col_blocks; j++) {
          remv[j] |= p[j];
        }
      }
    }
    *num_out = num_to_keep;
*/
    return d.ok();
  }
};

#define DEFINE_GPU_SPECS()                                 \
  template struct NonMaxSuppressionV5<GPUDevice>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

}  // namespace functor

}  // namespace tensorflow

#endif // GOOGLE_CUDA
