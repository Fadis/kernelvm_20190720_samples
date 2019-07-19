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
#include <glm/vec3.hpp>
#include <liblnn/layer_def.h>
#include <liblnn/descriptor_set.h>
#include <liblnn/pipeline_layout.h>
#include <liblnn/exceptions.h>
#include <liblnn/pipeline.h>

namespace liblnn {
  layer create_softmax_combined_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &teacher_value
  ) {
    const std::vector< vk::DescriptorSetLayoutBinding > descriptor_set_layout_bindings{
      vk::DescriptorSetLayoutBinding()
        .setDescriptorType( vk::DescriptorType::eStorageBuffer )
        .setDescriptorCount( 1 )
        .setBinding( 0 )
        .setStageFlags( vk::ShaderStageFlagBits::eCompute )
        .setPImmutableSamplers( nullptr ),
      vk::DescriptorSetLayoutBinding()
        .setDescriptorType( vk::DescriptorType::eStorageBuffer )
        .setDescriptorCount( 1 )
        .setBinding( 1 )
        .setStageFlags( vk::ShaderStageFlagBits::eCompute )
        .setPImmutableSamplers( nullptr ),
      vk::DescriptorSetLayoutBinding()
        .setDescriptorType( vk::DescriptorType::eStorageBuffer )
        .setDescriptorCount( 1 )
        .setBinding( 3 )
        .setStageFlags( vk::ShaderStageFlagBits::eCompute )
        .setPImmutableSamplers( nullptr ),
      vk::DescriptorSetLayoutBinding()
        .setDescriptorType( vk::DescriptorType::eStorageBuffer )
        .setDescriptorCount( 1 )
        .setBinding( 5 )
        .setStageFlags( vk::ShaderStageFlagBits::eCompute )
        .setPImmutableSamplers( nullptr )
    };
    const uint32_t batch_size = output_value.size();
    const uint32_t width = input_value.size() / batch_size;
    if( input_value.size() % batch_size ) throw invalid_data_length();
    if( input_value.size() != teacher_value.size() ) throw invalid_data_length();
    if( input_value.size() != input_grad.size() ) throw invalid_data_length();
    if( width > props.props.limits.maxComputeWorkGroupSize[ 0 ] ) throw too_large_data();
    if( batch_size > props.props.limits.maxComputeWorkGroupCount[ 2 ] ) throw too_large_data();
    auto [descriptor_set,descriptor_set_layout] = get_descriptor_set( device, descriptor_pool, descriptor_set_layout_bindings );
    std::vector< vk::PushConstantRange > push_constant_range{
      vk::PushConstantRange()
       .setStageFlags( vk::ShaderStageFlagBits::eCompute )
       .setOffset( 0 )
       .setSize( 8 )
    };
    auto pipeline_layout = get_pipeline_layout( device, descriptor_set_layout, push_constant_range );
    auto aligned_width = ( width / props.subgroup_props.subgroupSize + ( ( width % props.subgroup_props.subgroupSize ) ? 1 : 0 ) ) * props.subgroup_props.subgroupSize;
    std::array< uint32_t, 4 > spec_data{ aligned_width, 1, width, aligned_width / props.subgroup_props.subgroupSize };
    std::array< vk::SpecializationMapEntry, 4 > spec_ent{
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
        .setSize( 4 ),
      vk::SpecializationMapEntry()
        .setConstantID( 4 )
        .setOffset( 12 )
        .setSize( 4 )
    };
    auto spec = vk::SpecializationInfo()
      .setMapEntryCount( spec_ent.size() )
      .setPMapEntries( spec_ent.data() )
      .setDataSize( spec_data.size() * sizeof( uint32_t ) )
      .setPData( spec_data.data() );
    auto pipelines = device->createComputePipelines(
      *pipeline_cache,
      std::vector< vk::ComputePipelineCreateInfo >{
        vk::ComputePipelineCreateInfo()
          .setStage(
            vk::PipelineShaderStageCreateInfo()
              .setStage( vk::ShaderStageFlagBits::eCompute )
              .setModule( *mods.softmax_combined )
              .setPName( "main" )
              .setPSpecializationInfo( &spec )
          )
          .setLayout( *pipeline_layout )
      }
    );
    std::shared_ptr< vk::Pipeline > pipeline(
      new vk::Pipeline( std::move( pipelines[ 0 ] ) ),
      [device,pipeline_cache,module=mods.softmax_combined,pipeline_layout]( vk::Pipeline *p ) {
        if( p ) device->destroyPipeline( *p );
        delete p;
      }
    );

    auto input_value_dbi = vk::DescriptorBufferInfo()
      .setBuffer( input_value.get() )
      .setOffset( input_value.offset() * sizeof( float ) )
      .setRange( input_value.size() * sizeof( float ) );
    auto output_value_dbi = vk::DescriptorBufferInfo()
      .setBuffer( output_value.get() )
      .setOffset( output_value.offset() * sizeof( float ) )
      .setRange( output_value.size() * sizeof( float ) );
    auto input_grad_dbi = vk::DescriptorBufferInfo()
      .setBuffer( input_grad.get() )
      .setOffset( input_grad.offset() * sizeof( float ) )
      .setRange( input_grad.size() * sizeof( float ) );
    auto teacher_value_dbi = vk::DescriptorBufferInfo()
      .setBuffer( teacher_value.get() )
      .setOffset( teacher_value.offset() * sizeof( float ) )
      .setRange( teacher_value.size() * sizeof( float ) );
    device->updateDescriptorSets(
      std::vector< vk::WriteDescriptorSet >{
         vk::WriteDescriptorSet()
           .setDstSet( *descriptor_set )
           .setDstBinding( 0 )
           .setDescriptorType( vk::DescriptorType::eStorageBuffer )
           .setDescriptorCount( 1 )
           .setPBufferInfo( &input_value_dbi ),
         vk::WriteDescriptorSet()
           .setDstSet( *descriptor_set )
           .setDstBinding( 1 )
           .setDescriptorType( vk::DescriptorType::eStorageBuffer )
           .setDescriptorCount( 1 )
           .setPBufferInfo( &output_value_dbi ),
         vk::WriteDescriptorSet()
           .setDstSet( *descriptor_set )
           .setDstBinding( 3 )
           .setDescriptorType( vk::DescriptorType::eStorageBuffer )
           .setDescriptorCount( 1 )
           .setPBufferInfo( &input_grad_dbi ),
         vk::WriteDescriptorSet()
           .setDstSet( *descriptor_set )
           .setDstBinding( 5 )
           .setDescriptorType( vk::DescriptorType::eStorageBuffer )
           .setDescriptorCount( 1 )
           .setPBufferInfo( &teacher_value_dbi ),
      },
      nullptr
    );
    return layer( layer_def()
      .set_input_value( input_value )
      .set_output_value( output_value )
      .set_input_grad( input_grad )
      .set_teacher_value( teacher_value )
      .set_descriptor_set( descriptor_set )
      .set_pipeline( pipeline )
      .set_descriptor_set_layout( descriptor_set_layout )
      .set_pipeline_layout( pipeline_layout )
      .set_dispatch_size( 1, 1, batch_size ) );
  }
}

