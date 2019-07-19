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

#include <liblnn/network.h>
#include <memory>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <boost/scope_exit.hpp>
#include <liblnn/exceptions.h>
#include <liblnn/layer.h>
#include <liblnn/pipeline.h>
#include <liblnn/command_buffer.h>
#include <liblnn/print.h>
#include <liblnn/evaluate.h>
namespace liblnn {

  simple::simple(
    const std::shared_ptr< vk::CommandPool > &command_pool_,
    const std::shared_ptr< vk::Device > &device_,
    const std::shared_ptr< vk::Queue > &queue_,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
    const device_props &props_,
    const std::shared_ptr< VmaAllocator > &allocator_,
    const std::shared_ptr< data_source > &tin_,
    const std::shared_ptr< data_source > &ein_,
    const liblnn::modules &mods,
    size_t hidden_width_,
    size_t batch_size_,
    bool debug_
  ) : network( command_pool_, device_, queue_, descriptor_pool_, pipeline_cache_, props_, allocator_, tin_, ein_, mods, batch_size_, debug_ ), image_width( tin_->get_image_width() ), image_height( tin_->get_image_height() ), image_channels( tin_->get_image_channel() ), hidden_width( hidden_width_ ), output_width( tin_->get_label_width() ) {
    auto buf_type = debug ? VMA_MEMORY_USAGE_GPU_TO_CPU : VMA_MEMORY_USAGE_GPU_ONLY;
    const unsigned int image_size = train_input->get_image_width() * train_input->get_image_height() * train_input->get_image_channel();
    hidden_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_size * hidden_width * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( hidden_weight, image_size );
    output_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( hidden_width * output_width * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( output_weight, hidden_width );
    hidden_affine_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( hidden_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "hidden_affine_output" ), hidden_affine_output ) );
    hidden_activation_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( hidden_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "hidden_activation_output" ), hidden_activation_output ) );
    output_affine_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( output_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "output_affine_output" ), output_affine_output ) );
    output_activation_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( output_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "output_activation_output" ), output_activation_output ) );
    output_activation_output_eval.reset( new liblnn::buffer< float >(
      allocator, VMA_MEMORY_USAGE_GPU_TO_CPU,
      vk::BufferCreateInfo()
        .setSize( output_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    softmax_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( output_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "softmax_grad" ), softmax_grad ) );
    output_activation_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( output_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "output_activation_grad" ), output_activation_grad ) );
    output_affine_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( hidden_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "output_affine_grad" ), output_affine_grad ) );
    hidden_activation_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( hidden_width * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "hidden_activation_grad" ), hidden_activation_grad ) );
    hidden_affine_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "hidden_affine_grad" ), hidden_affine_grad ) );
    error_out.reset( new liblnn::buffer< float >(
      allocator, VMA_MEMORY_USAGE_GPU_TO_CPU,
      vk::BufferCreateInfo()
        .setSize( batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "error_out" ), error_out ) );
    hidden_affine1.reset( new layer( create_affine_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, batch_images[ 0 ], hidden_affine_output, hidden_weight, batch_size
    ) ) );
    hidden_affine2.reset( new layer( create_affine_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, batch_images[ 1 ], hidden_affine_output, hidden_weight, batch_size
    ) ) );
    hidden_affine3.reset( new layer( create_affine_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, batch_images[ 2 ], hidden_affine_output, hidden_weight, batch_size
    ) ) );
    hidden_activation.reset( new layer( create_relu_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, hidden_affine_output, hidden_activation_output
    ) ) );
    output_affine.reset( new layer( create_affine_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, hidden_activation_output, output_affine_output, output_weight, batch_size
    ) ) );
    output_activation.reset( new layer( create_tanh_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, output_affine_output, output_activation_output
    ) ) );
    output_activation3.reset( new layer( create_tanh_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, output_affine_output, output_activation_output_eval
    ) ) );
    error1.reset( new layer( create_softmax_combined_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, output_activation_output, error_out, softmax_grad, batch_labels[ 0 ]
    ) ) );
    error2.reset( new layer( create_softmax_combined_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, output_activation_output, error_out, softmax_grad, batch_labels[ 1 ]
    ) ) );
    output_activation_backward.reset( new layer( create_tanh_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, output_affine_output, output_activation_output, output_activation_grad, softmax_grad
    ) ) );
    output_affine_backward.reset( new layer( create_affine_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, hidden_activation_output, output_affine_output, output_weight, output_affine_grad, output_activation_grad, batch_size
    ) ) );
    hidden_activation_backward.reset( new layer( create_relu_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, hidden_affine_output, hidden_activation_output, hidden_activation_grad, output_affine_grad
    ) ) );
    hidden_affine_backward1.reset( new layer( create_affine_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, batch_images[ 0 ], hidden_affine_output, hidden_weight, hidden_affine_grad, hidden_activation_grad, batch_size
    ) ) );
    hidden_affine_backward2.reset( new layer( create_affine_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, batch_images[ 1 ], hidden_affine_output, hidden_weight, hidden_affine_grad, hidden_activation_grad, batch_size
    ) ) );
    {
      auto &command_buffer = (*command_buffers)[ 0 ];
      command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
      (*hidden_affine1)( command_buffer );
      (*hidden_activation)( command_buffer );
      (*output_affine)( command_buffer );
      (*output_activation)( command_buffer );
      (*error1)( command_buffer );
      (*output_activation_backward)( command_buffer );
      (*output_affine_backward)( command_buffer );
      (*hidden_activation_backward)( command_buffer );
      (*hidden_affine_backward1)( command_buffer );
      command_buffer.end();
    }
    {
      auto &command_buffer = (*command_buffers)[ 1 ];
      command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
      (*hidden_affine2)( command_buffer );
      (*hidden_activation)( command_buffer );
      (*output_affine)( command_buffer );
      (*output_activation)( command_buffer );
      (*error2)( command_buffer );
      (*output_activation_backward)( command_buffer );
      (*output_affine_backward)( command_buffer );
      (*hidden_activation_backward)( command_buffer );
      (*hidden_affine_backward2)( command_buffer );
      command_buffer.end();
    }
    {
      auto &command_buffer = (*command_buffers)[ 2 ];
      command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
      (*hidden_affine3)( command_buffer );
      (*hidden_activation)( command_buffer );
      (*output_affine)( command_buffer );
      (*output_activation3)( command_buffer );
      command_buffer.end();
    }
    fill( false, false );
    queue->waitIdle();
  }
}
