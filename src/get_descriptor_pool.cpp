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

#include <liblnn/descriptor_pool.h>
namespace liblnn {
  std::shared_ptr< vk::DescriptorPool > get_descriptior_pool(
    const std::shared_ptr< vk::Device > &device,
    const std::vector< vk::DescriptorPoolSize > &size,
    size_t max
  ) {
    auto descriptor_pool = device->createDescriptorPool(
      vk::DescriptorPoolCreateInfo()
        .setPoolSizeCount( size.size() )
        .setPPoolSizes( size.data() )
        .setMaxSets( max )
        .setFlags( vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet )
    );
    return std::shared_ptr< vk::DescriptorPool >(
      new vk::DescriptorPool( std::move( descriptor_pool ) ),
      [device]( vk::DescriptorPool *p ) {
        if( p ) {
          device->destroyDescriptorPool( *p );
          delete p;
        }
      }
    );
  }
}

