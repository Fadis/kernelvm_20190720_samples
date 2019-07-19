#ifndef LIBLNN_INCLUDE_LAYER_DEF_H
#define LIBLNN_INCLUDE_LAYER_DEF_H
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
#include <cstdint>
#include <array>
#include <vulkan/vulkan.hpp>
#include <liblnn/setter.h>
#include <liblnn/buffer.h>
#include <liblnn/buffer_view.h>
#include <glm/vec4.hpp>
namespace liblnn {
  struct layer_def {
    layer_def() : dispatch_size{ 1, 1, 1 }, batch_count( 1 ), clear_input_grad( false ) {}
    layer_def &set_dispatch_size( uint32_t x, uint32_t y, uint32_t z ) {
      dispatch_size[ 0 ] = x;
      dispatch_size[ 1 ] = y;
      dispatch_size[ 2 ] = z;
      return *this;
    }
    LIBLNN_SET_SMALL_VALUE( batch_count )
    LIBLNN_SET_LARGE_VALUE( dispatch_size )
    LIBLNN_SET_LARGE_VALUE( module )
    LIBLNN_SET_LARGE_VALUE( input_value )
    LIBLNN_SET_LARGE_VALUE( output_value )
    LIBLNN_SET_LARGE_VALUE( weight )
    LIBLNN_SET_LARGE_VALUE( input_grad )
    LIBLNN_SET_LARGE_VALUE( output_grad )
    LIBLNN_SET_LARGE_VALUE( teacher_value )
    LIBLNN_SET_LARGE_VALUE( pipeline )
    LIBLNN_SET_LARGE_VALUE( descriptor_set )
    LIBLNN_SET_LARGE_VALUE( pipeline_layout )
    LIBLNN_SET_LARGE_VALUE( descriptor_set_layout )
    LIBLNN_SET_SMALL_VALUE( clear_input_grad )
    std::array< uint32_t, 3 > dispatch_size;
    uint32_t batch_count;
    std::shared_ptr< vk::ShaderModule > module;
    buffer_view< float > input_value;
    buffer_view< float > output_value;
    buffer_view< glm::vec4 > weight;
    buffer_view< float > input_grad;
    buffer_view< float > output_grad;
    buffer_view< float > teacher_value;
    std::shared_ptr< vk::Pipeline > pipeline;
    std::shared_ptr< vk::DescriptorSet > descriptor_set;
    std::shared_ptr< vk::PipelineLayout > pipeline_layout;
    std::shared_ptr< vk::DescriptorSetLayout > descriptor_set_layout;
    bool clear_input_grad;
  };
}
#endif

