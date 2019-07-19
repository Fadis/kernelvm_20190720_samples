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

  conv6::conv6(
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
    size_t c1_channels_,
    size_t hidden_width_,
    size_t batch_size_,
    bool debug_
  ) : network( command_pool_, device_, queue_, descriptor_pool_, pipeline_cache_, props_, allocator_, tin_, ein_, mods, batch_size_, debug_ ), image_width( tin_->get_image_width() ), image_height( tin_->get_image_height() ), image_channels( tin_->get_image_channel() ), c1_width( tin_->get_image_width() / 2 ), c1_height( tin_->get_image_height() / 2 ), c1_channels( c1_channels_ ), hidden_width( hidden_width_ ), output_width( tin_->get_label_width() ) {
    auto buf_type = debug ? VMA_MEMORY_USAGE_GPU_TO_CPU : VMA_MEMORY_USAGE_GPU_ONLY;
    c1_conv1_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( 3 * 3 * image_channels * c1_channels * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( c1_conv1_weight, image_width * image_height * image_channels );
    c1_conv2_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( 3 * 3 * c1_channels * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( c1_conv2_weight, image_width * image_height * c1_channels );
    c1_conv3_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( 3 * 3 * c1_channels * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( c1_conv3_weight, image_width * image_height * c1_channels );
    hidden_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_width * c1_height * c1_channels * hidden_width * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( hidden_weight, c1_width * c1_height * c1_channels );
    output_weight.reset( new liblnn::buffer< glm::vec4 >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( hidden_width * output_width * sizeof( glm::vec4 ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferSrc|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    weights.emplace_back( output_weight, hidden_width );
    c1_conv1_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_conv1_output" ), c1_conv1_output ) );
    c1_activation1_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_conv1_output->size() * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_activation1_output" ), c1_activation1_output ) );
    c1_conv2_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_conv2_output" ), c1_conv2_output ) );
    c1_activation2_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_conv2_output->size() * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_activation2_output" ), c1_activation2_output ) );
    c1_conv3_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_conv3_output" ), c1_conv3_output ) );
    c1_activation3_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_conv3_output->size() * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_activation3_output" ), c1_activation3_output ) );
    c1_mp_output.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_width * c1_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
    buffers.insert( std::make_pair( std::string( "c1_mp_output" ), c1_mp_output ) );
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
        .setSize( c1_width * c1_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "hidden_affine_grad" ), hidden_affine_grad ) );
    c1_mp_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_mp_grad" ), c1_mp_grad ) );
    c1_activation3_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_mp_grad->size() * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_activation3_grad" ), c1_activation3_grad ) );
    c1_conv3_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_conv3_grad" ), c1_conv3_grad ) );
    c1_activation2_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_conv3_grad->size() * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_activation2_grad" ), c1_activation2_grad ) );
    c1_conv2_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * c1_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_conv2_grad" ), c1_conv2_grad ) );
    c1_activation1_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( c1_conv2_grad->size() * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_activation1_grad" ), c1_activation1_grad ) );
    c1_conv1_grad.reset( new liblnn::buffer< float >(
      allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_width * image_height * image_channels * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "c1_conv1_grad" ), c1_conv1_grad ) );
    error_out.reset( new liblnn::buffer< float >(
      allocator, VMA_MEMORY_USAGE_GPU_TO_CPU,
      vk::BufferCreateInfo()
        .setSize( batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
    ) );
    buffers.insert( std::make_pair( std::string( "error_out" ), error_out ) );

    c1_conv1_1.reset( new layer( create_conv_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 0 ], c1_conv1_output, c1_conv1_weight,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv1_2.reset( new layer( create_conv_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 1 ], c1_conv1_output, c1_conv1_weight,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv1_3.reset( new layer( create_conv_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 2 ], c1_conv1_output, c1_conv1_weight,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_activation1.reset( new layer( create_relu_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, c1_conv1_output, c1_activation1_output
    ) ) );
    c1_conv2.reset( new layer( create_conv_straight_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation1_output, c1_conv2_output, c1_conv2_weight,
      image_width, image_height, batch_size, 3, 3, c1_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_activation2.reset( new layer( create_relu_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, c1_conv2_output, c1_activation2_output
    ) ) );
    c1_conv3.reset( new layer( create_conv_straight_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation1_output, c1_conv3_output, c1_conv3_weight,
      image_width, image_height, batch_size, 3, 3, c1_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_activation3.reset( new layer( create_relu_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, c1_conv3_output, c1_activation3_output
    ) ) );
    c1_mp.reset( new layer( create_max_pooling_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, c1_activation3_output, c1_mp_output,
      c1_width, c1_height, c1_channels, batch_size, 2, 2, 2, 2
    ) ) );
    hidden_affine.reset( new layer( create_affine_forward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props, c1_mp_output, hidden_affine_output, hidden_weight, batch_size
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
    hidden_affine_backward.reset( new layer( create_affine_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_mp_output, hidden_affine_output, hidden_weight,
      hidden_affine_grad, hidden_activation_grad,
      batch_size
    ) ) );
    c1_mp_backward.reset( new layer( create_max_pooling_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation3_output, c1_mp_output, c1_mp_grad, hidden_affine_grad,
      c1_width, c1_height, c1_channels, batch_size, 2, 2, 2, 2
    ) ) );
    c1_activation3_backward.reset( new layer( create_relu_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_conv3_output, c1_activation3_output,
      c1_activation3_grad, c1_mp_grad
    ) ) );
    c1_conv3_bp_backward.reset( new layer( create_conv2_straight_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation2_output, c1_conv3_output, c1_conv3_weight, c1_conv3_grad, c1_activation3_grad,
      image_width, image_height, batch_size, 3, 3, c1_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv3_update_backward.reset( new layer( create_conv_straight_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation2_output, c1_conv3_output, c1_conv3_weight, c1_activation3_grad,
      image_width, image_height, batch_size, 3, 3, c1_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_activation2_backward.reset( new layer( create_relu_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_conv2_output, c1_activation2_output,
      c1_activation2_grad, c1_conv3_grad
    ) ) );
    c1_conv2_bp_backward.reset( new layer( create_conv2_straight_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation1_output, c1_conv2_output, c1_conv2_weight, c1_conv2_grad, c1_activation2_grad,
      image_width, image_height, batch_size, 3, 3, c1_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv2_update_backward.reset( new layer( create_conv_straight_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_activation1_output, c1_conv2_output, c1_conv2_weight, c1_activation2_grad,
      image_width, image_height, batch_size, 3, 3, c1_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_activation1_backward.reset( new layer( create_relu_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      c1_conv1_output, c1_activation1_output,
      c1_activation1_grad, c1_conv2_grad
    ) ) );
    c1_conv1_bp_backward_1.reset( new layer( create_conv2_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 0 ], c1_conv1_output, c1_conv1_weight, c1_conv1_grad, c1_activation1_grad,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv1_update_backward_1.reset( new layer( create_conv_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 0 ], c1_conv1_output, c1_conv1_weight, c1_activation1_grad,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv1_bp_backward_2.reset( new layer( create_conv2_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 1 ], c1_conv1_output, c1_conv1_weight, c1_conv1_grad, c1_activation1_grad,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    c1_conv1_update_backward_2.reset( new layer( create_conv_backward_pipeline(
      device, mods, descriptor_pool, pipeline_cache, props,
      batch_images[ 1 ], c1_conv1_output, c1_conv1_weight, c1_activation1_grad,
      image_width, image_height, c1_channels, batch_size, 3, 3, image_channels, 1, 1, 1, 1, 1
    ) ) );
    {
      auto &command_buffer = (*command_buffers)[ 0 ];
      command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
      (*c1_conv1_1)( command_buffer );
      (*c1_activation1)( command_buffer );
      (*c1_conv2)( command_buffer );
      (*c1_activation2)( command_buffer );
      (*c1_conv3)( command_buffer );
      (*c1_activation3)( command_buffer );
      (*c1_mp)( command_buffer );
      (*hidden_affine)( command_buffer );
      (*hidden_activation)( command_buffer );
      (*output_affine)( command_buffer );
      (*output_activation)( command_buffer );
      (*error1)( command_buffer );
      (*output_activation_backward)( command_buffer );
      (*output_affine_backward)( command_buffer );
      (*hidden_activation_backward)( command_buffer );
      (*hidden_affine_backward)( command_buffer );
      (*c1_mp_backward)( command_buffer );
      (*c1_activation3_backward)( command_buffer );
      (*c1_conv3_bp_backward)( command_buffer );
      (*c1_conv3_update_backward)( command_buffer );
      (*c1_activation2_backward)( command_buffer );
      (*c1_conv2_bp_backward)( command_buffer );
      (*c1_conv2_update_backward)( command_buffer );
      (*c1_activation1_backward)( command_buffer );
      (*c1_conv1_bp_backward_1)( command_buffer );
      (*c1_conv1_update_backward_1)( command_buffer );
      command_buffer.end();
    }
    {
      auto &command_buffer = (*command_buffers)[ 1 ];
      command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
      (*c1_conv1_2)( command_buffer );
      (*c1_activation1)( command_buffer );
      (*c1_conv2)( command_buffer );
      (*c1_activation2)( command_buffer );
      (*c1_conv3)( command_buffer );
      (*c1_activation3)( command_buffer );
      (*c1_mp)( command_buffer );
      (*hidden_affine)( command_buffer );
      (*hidden_activation)( command_buffer );
      (*output_affine)( command_buffer );
      (*output_activation)( command_buffer );
      (*error2)( command_buffer );
      (*output_activation_backward)( command_buffer );
      (*output_affine_backward)( command_buffer );
      (*hidden_activation_backward)( command_buffer );
      (*hidden_affine_backward)( command_buffer );
      (*c1_mp_backward)( command_buffer );
      (*c1_activation3_backward)( command_buffer );
      (*c1_conv3_bp_backward)( command_buffer );
      (*c1_conv3_update_backward)( command_buffer );
      (*c1_activation2_backward)( command_buffer );
      (*c1_conv2_bp_backward)( command_buffer );
      (*c1_conv2_update_backward)( command_buffer );
      (*c1_activation1_backward)( command_buffer );
      (*c1_conv1_bp_backward_2)( command_buffer );
      (*c1_conv1_update_backward_2)( command_buffer );
      command_buffer.end();
    }
    {
      auto &command_buffer = (*command_buffers)[ 2 ];
      command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
      (*c1_conv1_3)( command_buffer );
      (*c1_activation1)( command_buffer );
      (*c1_conv2)( command_buffer );
      (*c1_activation2)( command_buffer );
      (*c1_conv3)( command_buffer );
      (*c1_activation3)( command_buffer );
      (*c1_mp)( command_buffer );
      (*hidden_affine)( command_buffer );
      (*hidden_activation)( command_buffer );
      (*output_affine)( command_buffer );
      (*output_activation3)( command_buffer );
      command_buffer.end();
    }
    fill( false, false );
    queue->waitIdle();
  }
}
