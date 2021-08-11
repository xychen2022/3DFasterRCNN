# Network Installation

https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

# Manual Installation

Environment CUDA 11.2, cuDNN 8.1

1. Download CUDA 11.2 and cuDNN 8.1 to a specified folder

2. Change directory to the folder where CUDA and cuDNN are stored

3. Remove all nvidia packages, skip this if your system is fresh installed

sudo apt-get remove nvidia* && sudo apt autoremove

4. Install some packages for build kernel:

sudo apt-get install dkms build-essential linux-headers-generic

5. Now, block and disable nouveau kernel driver using command:

sudo vi /etc/modprobe.d/blacklist.conf

6. Insert following lines to the blacklist.conf:

blacklist nouveau

blacklist lbm-nouveau

options nouveau modeset=0

alias nouveau off

alias lbm-nouveau off

Then, save and exit.

7. Disable the Kernel nouveau by typing the following commands(nouveau-kms.conf may not exist,it is ok):

echo options nouveau modeset=0 | sudo tee -a /etc/modprobe.d/nouveau-kms.conf

8. Build the new kernel by:

sudo update-initramfs -u

9. Reboot (sudo reboot)

10. Install CUDA (the command may vary for different CUDA versions):

sudo sh cuda_11.2.0_460.27.04_linux.run

11. sudo apt install nvidia-cuda-toolkit

12. Copy cuDNN files to CUDA

sudo cp cuda/include/cudnn*.h /usr/local/cuda/include

sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-11.2/lib64 

sudo chmod a+r /usr/local/cuda-11.2/include/cudnn*.h /usr/local/cuda-11.2/lib64/libcudnn*

13. Possibly reboot (sudo reboot)


