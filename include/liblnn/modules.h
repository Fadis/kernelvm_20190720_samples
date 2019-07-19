#ifndef LIBLNN_INCLUDE_MODULES_H
#define LIBLNN_INCLUDE_MODULES_H
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
namespace liblnn {
  struct modules {
    modules( const std::shared_ptr< vk::Device > &device );
    std::shared_ptr< vk::ShaderModule > init;
    std::shared_ptr< vk::ShaderModule > affine_forward;
    std::shared_ptr< vk::ShaderModule > affine_backward;
    std::shared_ptr< vk::ShaderModule > relu_forward;
    std::shared_ptr< vk::ShaderModule > relu_backward;
    std::shared_ptr< vk::ShaderModule > tanh_forward;
    std::shared_ptr< vk::ShaderModule > tanh_backward;
    std::shared_ptr< vk::ShaderModule > conv_forward;
    std::shared_ptr< vk::ShaderModule > conv_backward;
    std::shared_ptr< vk::ShaderModule > conv2_backward;
    std::shared_ptr< vk::ShaderModule > conv_straight_forward;
    std::shared_ptr< vk::ShaderModule > conv_straight_backward;
    std::shared_ptr< vk::ShaderModule > conv2_straight_backward;
    std::shared_ptr< vk::ShaderModule > maxpooling_forward;
    std::shared_ptr< vk::ShaderModule > maxpooling_backward;
    std::shared_ptr< vk::ShaderModule > add_forward;
    std::shared_ptr< vk::ShaderModule > add_backward;
    std::shared_ptr< vk::ShaderModule > softmax_combined;
  };
}
#endif

