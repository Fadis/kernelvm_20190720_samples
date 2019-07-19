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
#include <memory>
#include <vulkan/vulkan.hpp>
#include <liblnn/command_buffer.h>

namespace liblnn {
  std::shared_ptr< std::vector< vk::CommandBuffer > >
  get_command_buffers(
    const std::shared_ptr< vk::Device > &device,
    const std::shared_ptr< vk::CommandPool > &command_pool,
    size_t count
  ) {
    auto command_buffers = device->allocateCommandBuffers(
      vk::CommandBufferAllocateInfo()
        .setCommandPool( *command_pool )
        .setLevel( vk::CommandBufferLevel::ePrimary )
        .setCommandBufferCount( count )
    );
    return std::shared_ptr< std::vector< vk::CommandBuffer > >(
      new std::vector< vk::CommandBuffer >( std::move( command_buffers ) ),
      [device,command_pool]( std::vector< vk::CommandBuffer > *p ) {
        if( p ) {
          device->freeCommandBuffers( *command_pool, *p );
          delete p;
        }
      }
    );
  }
}

