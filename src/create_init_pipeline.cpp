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

#include <array>
#include <vector>
#include <utility>
#include <boost/math/common_factor_rt.hpp>
#include <glm/vec4.hpp>
#include <liblnn/layer_def.h>
#include <liblnn/descriptor_set.h>
#include <liblnn/pipeline_layout.h>
#include <liblnn/exceptions.h>
#include <liblnn/pipeline.h>
namespace liblnn {
  layer create_init_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< glm::vec4 > &weight,
    uint32_t input_size
  ) {
    const std::vector< vk::DescriptorSetLayoutBinding > descriptor_set_layout_bindings{
      vk::DescriptorSetLayoutBinding()
        .setDescriptorType( vk::DescriptorType::eStorageBuffer )
        .setDescriptorCount( 1 )
        .setBinding( 2 )
        .setStageFlags( vk::ShaderStageFlagBits::eCompute )
        .setPImmutableSamplers( nullptr )
    };
    const uint32_t size = weight.size();
    uint32_t width = ( size > props.props.limits.maxComputeWorkGroupCount[ 0 ] ) ? boost::math::gcd( size, props.props.limits.maxComputeWorkGroupCount[ 0 ] ) : size;
    uint32_t local_group_size = boost::math::gcd( width, props.subgroup_props.subgroupSize );
    uint32_t height = size / width;
    if( height > props.props.limits.maxComputeWorkGroupCount[ 1 ] ) throw too_large_data();
    auto [descriptor_set,descriptor_set_layout] = get_descriptor_set( device, descriptor_pool, descriptor_set_layout_bindings );
    std::vector< vk::PushConstantRange > push_constant_range{
      vk::PushConstantRange()
       .setStageFlags( vk::ShaderStageFlagBits::eCompute )
       .setOffset( 0 )
       .setSize( 8 )
    };
    auto pipeline_layout = get_pipeline_layout( device, descriptor_set_layout, push_constant_range );
    std::array< uint32_t, 3 > spec_data{ local_group_size, 1, input_size };
    std::array< vk::SpecializationMapEntry, 3 > spec_ent {
      vk::SpecializationMapEntry()
        .setConstantID( 1 )
        .setOffset( 0 )
        .setSize( 4 ),
      vk::SpecializationMapEntry()
        .setConstantID( 2 )
        .setOffset( 4 )
        .setSize( 4 ),
      vk::SpecializationMapEntry()
        .setConstantID( 3 )
        .setOffset( 8 )
        .setSize( 4 )
    };
    auto spec = vk::SpecializationInfo()
      .setMapEntryCount( spec_ent.size() )
      .setPMapEntries( spec_ent.data() )
      .setDataSize( spec_data.size() )
      .setPData( spec_data.data() );
    auto pipelines = device->createComputePipelines(
      *pipeline_cache,
      std::vector< vk::ComputePipelineCreateInfo >{
        vk::ComputePipelineCreateInfo()
    .setStage(
      vk::PipelineShaderStageCreateInfo()
        .setStage( vk::ShaderStageFlagBits::eCompute )
        .setModule( *mods.init )
        .setPName( "main" )
    .setPSpecializationInfo( &spec )
    )
    .setLayout( *pipeline_layout )
      }
    );
    std::shared_ptr< vk::Pipeline > pipeline(
      new vk::Pipeline( std::move( pipelines[ 0 ] ) ),
      [device,pipeline_cache,module=mods.init,pipeline_layout]( vk::Pipeline *p ) {
        if( p ) device->destroyPipeline( *p );
        delete p;
      }
    );

    auto weight_dbi = vk::DescriptorBufferInfo()
      .setBuffer( weight.get() )
      .setOffset( weight.offset() * sizeof( glm::vec4 ) )
      .setRange( weight.size() * sizeof( glm::vec4 ) );
    device->updateDescriptorSets(
      std::vector< vk::WriteDescriptorSet >{
         vk::WriteDescriptorSet()
           .setDstSet( *descriptor_set )
           .setDstBinding( 2 )
           .setDescriptorType( vk::DescriptorType::eStorageBuffer )
           .setDescriptorCount( 1 )
           .setPBufferInfo( &weight_dbi )
      },
      nullptr
    );
    return layer( layer_def()
      .set_weight( weight )
      .set_descriptor_set( descriptor_set )
      .set_pipeline( pipeline )
      .set_descriptor_set_layout( descriptor_set_layout )
      .set_pipeline_layout( pipeline_layout )
      .set_dispatch_size( width / local_group_size, height, 1 ) );
  }
}

