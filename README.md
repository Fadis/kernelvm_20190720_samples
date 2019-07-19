# kernelvm_20190720_samples
Deep learning demo in Vulkan

* Requirements

CMake >= 2.8
Boost >= 1.65.0
Vulkan >= 1.1 ( VK\_LAYER\_LUNARG\_standard\_validation is required to run in validation mode )
OpenImageIO

* Build instruction

$ cd ${SOURCE\_DIR}
$ mkdir build
$ cd build
$ cmake ../
$ make
$ cd ../shaders
$ ./compile.sh
$ mv \*.spv ../build
$ cd ../build

* Dataset

Decompressed MNIST or compatible dataset is required.

MNIST
http://yann.lecun.com/exdb/mnist/

Fashion-MNIST
https://github.com/zalandoresearch/fashion-mnist/

