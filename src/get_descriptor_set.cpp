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
#include <liblnn/descriptor_set.h>
namespace liblnn {
  std::tuple< std::shared_ptr< vk::DescriptorSet >, std::shared_ptr< vk::DescriptorSetLayout > >
  get_descriptor_set(
    const std::shared_ptr< vk::Device > &device,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::vector< vk::DescriptorSetLayoutBinding > &descriptor_set_layout_bindings
  ) {
    std::vector< vk::DescriptorSetLayout > descriptor_set_layout;
    descriptor_set_layout.emplace_back( device->createDescriptorSetLayout(
      vk::DescriptorSetLayoutCreateInfo()
        .setBindingCount( descriptor_set_layout_bindings.size() )
        .setPBindings( descriptor_set_layout_bindings.data() ),
      nullptr
    ) );
    auto descriptor_set = device->allocateDescriptorSets(
      vk::DescriptorSetAllocateInfo()
        .setDescriptorPool( *descriptor_pool )
        .setDescriptorSetCount( descriptor_set_layout.size() )
        .setPSetLayouts( descriptor_set_layout.data() )
    );
    return std::make_tuple(
      std::shared_ptr< vk::DescriptorSet >(
        new vk::DescriptorSet( std::move( descriptor_set.front() ) ), [device,descriptor_pool]( vk::DescriptorSet *p ) {
          if( p ) {
            device->freeDescriptorSets( *descriptor_pool, 1, p );
            delete p;
          }
        }
      ),
      std::shared_ptr< vk::DescriptorSetLayout >(
        new vk::DescriptorSetLayout( std::move( descriptor_set_layout.front() ) ),
        [device]( vk::DescriptorSetLayout *p ) {
          if( p ) {
            device->destroyDescriptorSetLayout( *p );
            delete p;
          }
        }
      )
    );
  }
}

