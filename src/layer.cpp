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

#include <vector>
#include <array>
#include <glm/vec4.hpp>
#include <liblnn/layer.h>
namespace liblnn {
  void layer::operator()( vk::CommandBuffer &command_buffer ) const {
    std::vector< uint32_t > ds_offset{};
    command_buffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, *def.pipeline_layout, 0, *def.descriptor_set, ds_offset );
    command_buffer.bindPipeline( vk::PipelineBindPoint::eCompute, *def.pipeline );
    std::vector< uint32_t > constants{ 0 };
    std::vector< vk::BufferMemoryBarrier > barrier;
    if( def.input_value )
      barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.input_value.get() )
          .setOffset( def.input_value.offset() * sizeof( float ) )
          .setSize( def.input_value.size() * sizeof( float ) )
      );
    if( def.output_value )
      barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.output_value.get() )
          .setOffset( def.output_value.offset() * sizeof( float ) )
          .setSize( def.output_value.size() * sizeof( float ) )
      );
    if( def.weight )
      barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.weight.get() )
          .setOffset( def.weight.offset() * sizeof( glm::vec4 ) )
          .setSize( def.weight.size() * sizeof( glm::vec4 ) )
      );
    if( def.input_grad )
      barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.input_grad.get() )
          .setOffset( def.input_grad.offset() * sizeof( float ) )
          .setSize( def.input_grad.size() * sizeof( float ) )
      );
    if( def.output_grad )
      barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.output_grad.get() )
          .setOffset( def.output_grad.offset() * sizeof( float ) )
          .setSize( def.output_grad.size() * sizeof( float ) )
      );
    if( def.teacher_value )
      barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.teacher_value.get() )
          .setOffset( def.teacher_value.offset() * sizeof( float ) )
          .setSize( def.teacher_value.size() * sizeof( float ) )
      );
    std::array< uint32_t, 1 > pcs{ def.batch_count };
    if( def.clear_input_grad && def.input_grad ) {
      std::vector< vk::BufferMemoryBarrier > fill_barrier;
      fill_barrier.emplace_back(
        vk::BufferMemoryBarrier()
          .setSrcAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setDstAccessMask( vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite )
          .setBuffer( def.input_grad.get() )
          .setOffset( def.input_grad.offset() * sizeof( float ) )
          .setSize( def.input_grad.size() * sizeof( float ) )
      );
      command_buffer.fillBuffer( def.input_grad.get(), 0, def.input_grad.size() * sizeof( float ), 0 );
      command_buffer.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::DependencyFlagBits::eDeviceGroup,
        std::vector< vk::MemoryBarrier >{},
        fill_barrier,
        std::vector< vk::ImageMemoryBarrier >{}
      );
    }
    command_buffer.pushConstants< uint32_t >( *def.pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, pcs );
    command_buffer.dispatch( def.dispatch_size[ 0 ], def.dispatch_size[ 1 ], def.dispatch_size[ 2 ] );
    command_buffer.pipelineBarrier(
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::DependencyFlagBits::eDeviceGroup,
      std::vector< vk::MemoryBarrier >{},
      barrier,
      std::vector< vk::ImageMemoryBarrier >{}
    );
  }
}


