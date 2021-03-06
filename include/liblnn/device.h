#ifndef LIBLNN_INCLUDE_GET_DEVICE_H
#define LIBLNN_INCLUDE_GET_DEVICE_H
/*
Copyright (c) 2019 Naomasa Matsubayashi (aka. Fadis)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <memory>
#include <vulkan/vulkan.hpp>
#include <liblnn/config.h>

namespace liblnn {
  std::tuple<
    std::shared_ptr< vk::Device >,
    std::shared_ptr< vk::Queue >,
    std::shared_ptr< vk::CommandPool >
  > get_device(
    const configs_t &config,
    const vk::PhysicalDevice &physical_device,
    const std::vector< const char* > &dext,
    const std::vector< const char* > &dlayers
  );
}

#endif


