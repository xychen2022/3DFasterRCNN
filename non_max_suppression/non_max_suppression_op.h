#ifndef TENSORFLOW_CORE_KERNELS_NON_MAX_SUPPRESSION_V5_OP_H_
#define TENSORFLOW_CORE_KERNELS_NON_MAX_SUPPRESSION_V5_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

//#include "cuda.h"
//#include "/usr/include/cuda.h"
//#include "/usr/include/linux/cuda.h"
#include "/usr/local/cuda-11.2/include/cuda.h"
//#include "/usr/local/cuda-11.0/targets/x86_64-linux/include/cuda.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device>
struct NonMaxSuppressionV5 {
  bool operator()(const Device& d, typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<float, 1>::ConstTensor scores,
                  int max_output_size,
                  float iou_threshold, float score_threshold,
                  bool pad_to_max_output_size,
                  typename TTypes<int, 1>::Tensor selected_indices);
};

}  // namespace functor
}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_NON_MAX_SUPPRESSION_V5_OP_H_
