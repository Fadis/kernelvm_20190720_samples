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

#include <liblnn/modules.h>
#include <liblnn/shader.h>
namespace liblnn {
  modules::modules(
    const std::shared_ptr< vk::Device > &device
  ) {
    init = liblnn::get_shader( device, "init.comp.spv" );
    affine_forward = liblnn::get_shader( device, "affine_forward.comp.spv" );
    affine_backward = liblnn::get_shader( device, "affine_backward.comp.spv" );
    relu_forward = liblnn::get_shader( device, "relu_forward.comp.spv" );
    relu_backward = liblnn::get_shader( device, "relu_backward.comp.spv" );
    tanh_forward = liblnn::get_shader( device, "tanh_forward.comp.spv" );
    tanh_backward = liblnn::get_shader( device, "tanh_backward.comp.spv" );
    conv_forward = liblnn::get_shader( device, "conv_forward.comp.spv" );
    conv_backward = liblnn::get_shader( device, "conv_backward.comp.spv" );
    conv2_backward = liblnn::get_shader( device, "conv2_backward.comp.spv" );
    conv_straight_forward = liblnn::get_shader( device, "conv_straight_forward.comp.spv" );
    conv_straight_backward = liblnn::get_shader( device, "conv_straight_backward.comp.spv" );
    conv2_straight_backward = liblnn::get_shader( device, "conv2_straight_backward.comp.spv" );
    maxpooling_forward = liblnn::get_shader( device, "maxpooling_forward.comp.spv" );
    maxpooling_backward = liblnn::get_shader( device, "maxpooling_backward.comp.spv" );
    softmax_combined = liblnn::get_shader( device, "softmax_combined.comp.spv" );
  }
}
