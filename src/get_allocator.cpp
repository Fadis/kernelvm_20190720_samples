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

#include <liblnn/allocator.h>
namespace liblnn {
  std::shared_ptr< VmaAllocator >
  get_allocator(
    const vk::PhysicalDevice &physical_device,
    const std::shared_ptr< vk::Device > &device
  ) {
    VmaAllocatorCreateInfo allocator_info = {};
    allocator_info.flags = 0;
    allocator_info.physicalDevice = physical_device;
    allocator_info.device = *device;
    allocator_info.preferredLargeHeapBlockSize = 0;
    allocator_info.pAllocationCallbacks = nullptr;
    allocator_info.pDeviceMemoryCallbacks = nullptr;
    allocator_info.frameInUseCount = 0;
    allocator_info.pHeapSizeLimit = nullptr;
    allocator_info.pVulkanFunctions = nullptr;
    allocator_info.pRecordSettings = nullptr;
    VmaAllocator allocator;
    {
      const auto result = vmaCreateAllocator( &allocator_info, &allocator );
      if( result != VK_SUCCESS ) vk::throwResultException( vk::Result( result ), "アロケータを作成できない" );
    }
    return std::shared_ptr< VmaAllocator >(
      new VmaAllocator( std::move( allocator ) ),
      []( VmaAllocator *p ){
        vmaDestroyAllocator( *p );
        delete p;
      }
    );
  }
}

