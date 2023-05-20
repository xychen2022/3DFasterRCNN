//#ifndef CROP_RESIZE_V3_OP_H
//#define CROP_RESIZE_V3_OP_H
#ifndef TENSORFLOW_CORE_KERNELS_CROP_RESIZE_V3_OP_H_
#define TENSORFLOW_CORE_KERNELS_CROP_RESIZE_V3_OP_H_

//#include "cuda.h"
#include "/usr/local/cuda-11.2/include/cuda.h"
//#include "/usr/include/cuda.h"
//#include "/usr/include/linux/cuda.h"
//#include "/usr/include/cuda.h"
//#include "/usr/local/cuda-11/targets/x86_64-linux/include/cuda.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct CropAndResizeV3 {
  // We assume that the tensor sizes are correct.
  bool operator()(const OpKernelContext* context,
                  typename TTypes<T, 5>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  string method_name, float extrapolation_value,
                  typename TTypes<float, 5>::Tensor crops);
};

template <typename Device, typename T>
struct CropAndResizeV3BackpropImage {
  // We assume that the tensor sizes are correct.
  bool operator()(const Device& d, typename TTypes<float, 5>::ConstTensor grads,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<T, 5>::Tensor grads_image,
                  const string& method_name);
};

template <typename Device, typename T>
struct CropAndResizeV3BackpropBoxes {
  // We assume that the tensor sizes are correct.
  bool operator()(const Device& d, typename TTypes<float, 5>::ConstTensor grads,
                  typename TTypes<T, 5>::ConstTensor image,
                  typename TTypes<float, 2>::ConstTensor boxes,
                  typename TTypes<int32, 1>::ConstTensor box_ind,
                  typename TTypes<float, 2>::Tensor grads_boxes);
};

template <typename Device>
struct CheckValidBoxIndexHelper {
  // Checks if all values in box_index are in [0, batch).
  void operator()(const Device& d,
                  typename TTypes<int32, 1>::ConstTensor box_index, int batch,
                  typename TTypes<bool, 0>::Tensor isvalid) {
    isvalid.device(d) = ((box_index >= 0) && (box_index < batch)).all();
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CROP_AND_RESIZE_V3_OP_H_
