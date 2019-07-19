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
#include <array>
#include <vector>
#include <algorithm>
#include <liblnn/buffer.h>
#include <liblnn/input_cache.h>
#include <liblnn/exceptions.h>
#include <liblnn/command_buffer.h>

namespace liblnn {
  input_cache::input_cache(
    std::shared_ptr< VmaAllocator > allocator,
    std::shared_ptr< data_source > source_,
    uint32_t count
  ) : source( source_ ), cache_count( count ), current_image( count ) {
    images.reset( new liblnn::buffer< float >(
      allocator, VMA_MEMORY_USAGE_CPU_TO_GPU,
      vk::BufferCreateInfo()
        .setSize( source->get_image_width() * source->get_image_height() * source->get_image_channel() * cache_count * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eTransferSrc )
      ) );
    labels.reset( new liblnn::buffer< float >(
      allocator, VMA_MEMORY_USAGE_CPU_TO_GPU,
      vk::BufferCreateInfo()
        .setSize( source->get_label_width() * cache_count * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eTransferSrc )
      ) );
  }
  void input_cache::operator()(
    vk::CommandBuffer &command_buffer,
    const std::shared_ptr< vk::Device > &device,
    const std::shared_ptr< vk::Queue > &queue,
    const device_props &props,
    const buffer_view< float > &image_buffer,
    const buffer_view< float > &label_buffer
  ) {
    if( current_image == cache_count ) {
      (*source)( command_buffer, device, queue, props, images, labels );
      current_image = 0;
    }
    size_t image_size = get_image_width() * get_image_height() * get_image_channel();
    if( image_buffer.size() % image_size ) throw invalid_data_length();
    size_t batch_size = image_buffer.size() / image_size;
    if( ( cache_count - current_image ) % batch_size ) throw invalid_data_length();
    size_t label_size = get_label_width();
    if( label_buffer.size() != batch_size * label_size ) throw invalid_data_length();
    std::array< vk::BufferCopy, 1 > image_region{
      vk::BufferCopy().setSize( batch_size * image_size * sizeof( float ) ).setSrcOffset( current_image * image_size * sizeof( float ) )
    };
    std::array< vk::BufferCopy, 1 > label_region{
      vk::BufferCopy().setSize( batch_size * label_size * sizeof( float ) ).setSrcOffset( current_image * label_size * sizeof( float ) )
    };
    command_buffer.copyBuffer( images->get(), image_buffer.get(), image_region );
    command_buffer.copyBuffer( labels->get(), label_buffer.get(), label_region );
    std::vector< vk::BufferMemoryBarrier > barrier;
    barrier.emplace_back(
      vk::BufferMemoryBarrier()
        .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
        .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
        .setBuffer( image_buffer.get() )
        .setOffset( image_buffer.offset() * sizeof( float ) )
        .setSize( image_buffer.size() * sizeof( float ) )
    );
    barrier.emplace_back(
      vk::BufferMemoryBarrier()
        .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
        .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
        .setBuffer( label_buffer.get() )
        .setOffset( label_buffer.offset() * sizeof( float ) )
        .setSize( label_buffer.size() * sizeof( float ) )
    );
    command_buffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::DependencyFlagBits::eDeviceGroup,
      std::vector< vk::MemoryBarrier >{},
      barrier,
      std::vector< vk::ImageMemoryBarrier >{}
    );
    current_image += batch_size;
  }
}
