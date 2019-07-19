#ifndef LIBLNN_INCLUDE_BUFFER_H
#define LIBLNN_INCLUDE_BUFFER_H
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
#include <vk_mem_alloc.h>
namespace liblnn {
  template< typename T >
  class buffer {
  public:
    buffer(
      const std::shared_ptr< VmaAllocator > &allocator_,
      VmaMemoryUsage usage,
      const VkBufferCreateInfo &create_info
    ) : allocator( allocator_ ), length( create_info.size / sizeof( T ) ) {
      VmaAllocationCreateInfo alloc_info = {};
      alloc_info.flags = 0;
      alloc_info.usage = usage;
      alloc_info.requiredFlags = 0;
      alloc_info.preferredFlags = 0;
      alloc_info.memoryTypeBits = 0;
      alloc_info.pool = VK_NULL_HANDLE;
      alloc_info.pUserData = nullptr;
      const auto result = vmaCreateBuffer( *allocator, &create_info, &alloc_info, &raw, &alloc, nullptr );
      if( result != VK_SUCCESS ) vk::throwResultException( vk::Result( result ), "バッファを作成できない" );
    }
    buffer( const buffer& ) = delete;
    buffer &operator=( const buffer& ) = delete;
    ~buffer() {
      vmaDestroyBuffer( *allocator, raw, alloc );
    }
    std::shared_ptr< T > map() {
      void* mapped_memory;
      const auto result = vmaMapMemory( *allocator, alloc, &mapped_memory );
      if( result != VK_SUCCESS ) vk::throwResultException( vk::Result( result ), "バッファをマップできない" );
      return std::shared_ptr< T >(
        reinterpret_cast< T* >( mapped_memory ),
        [allocator=allocator,alloc=alloc]( T *p ) {
          if( p ) vmaUnmapMemory( *allocator, alloc );
        }
      );
    }
    size_t size() const { return length; }
    VkBuffer &get() { return raw; }
  private:
    std::shared_ptr< VmaAllocator > allocator;
    size_t length;
    VmaAllocation alloc;
    VkBuffer raw;
  };
}
#endif

