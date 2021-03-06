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

#include <liblnn/pipeline_layout.h>
namespace liblnn {
  std::shared_ptr< vk::PipelineLayout >
  get_pipeline_layout(
    const std::shared_ptr< vk::Device > &device,
    const std::shared_ptr< vk::DescriptorSetLayout > &descriptor_set_layout,
    const std::vector< vk::PushConstantRange > &push_constant_range
  ) {
    auto pipeline_layout = device->createPipelineLayout(
      vk::PipelineLayoutCreateInfo()
        .setSetLayoutCount( 1 )
        .setPSetLayouts( descriptor_set_layout.get() )
        .setPushConstantRangeCount( push_constant_range.size() )
        .setPPushConstantRanges( push_constant_range.data() )
    );
    return std::shared_ptr< vk::PipelineLayout >(
      new vk::PipelineLayout( pipeline_layout ),
      [device]( vk::PipelineLayout *p ) {
        if( p ) {
          device->destroyPipelineLayout( *p );
          delete p;
        }
      }
    );
  }
}

