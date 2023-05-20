How to compile? Assume you are use CUDA 11.2. You should make proper changes when you run in a different environment

First, add the following environment variables to path

    export PATH=/usr/local/cuda-11.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH


Then, run the follwing commands

    TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

    TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

    TF_IFLAGS=( $(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')  )

    nvcc -std=c++11 -c -o crop_and_resize_op.cu.o crop_and_resize_op_gpu.cu.cc -I/usr/local/cuda-11.2/include ${TF_CFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=0 ${TF_LFLAGS[@]}  -x cu -Xcompiler -fPIC
    g++ -std=c++11 -shared -o crop_and_resize_op_gpu.so crop_and_resize_op.cc crop_and_resize_op.cu.o -D_GLIBCXX_USE_CXX11_ABI=0 -I$TF_IFLAGS -I$TF_IFLAGS/external/nsync/public -L/usr/local/cuda-11.2/lib64 ${TF_LFLAGS[@]} -fPIC -lcudart
