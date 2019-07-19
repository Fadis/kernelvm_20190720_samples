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

#include <liblnn/network.h>
#include <memory>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <boost/scope_exit.hpp>
#include <boost/crc.hpp>
#include <liblnn/exceptions.h>
#include <liblnn/layer.h>
#include <liblnn/pipeline.h>
#include <liblnn/command_buffer.h>
#include <liblnn/print.h>
#include <liblnn/evaluate.h>
namespace liblnn {
  network::network(
    const std::shared_ptr< vk::CommandPool > &command_pool_,
    const std::shared_ptr< vk::Device > &device_,
    const std::shared_ptr< vk::Queue > &queue_,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
    const device_props &props_,
    const std::shared_ptr< VmaAllocator > &allocator_,
    const std::shared_ptr< data_source > &tin_,
    const std::shared_ptr< data_source > &ein_,
    const liblnn::modules &mods_,
    size_t batch_size_,
    bool debug_
  ) : command_pool( command_pool_ ), device( device_ ), queue( queue_ ), descriptor_pool( descriptor_pool_ ), pipeline_cache( pipeline_cache_ ), props( props_ ), allocator( allocator_ ), train_input( tin_ ), eval_input( ein_ ), mods( mods_ ), batch_size( batch_size_ ), debug( debug_ ), swap_index( 0 ) {
    if( train_input->get_image_width() != eval_input->get_image_width() ) throw invalid_data_length();
    if( train_input->get_image_height() != eval_input->get_image_height() ) throw invalid_data_length();
    if( train_input->get_image_channel() != eval_input->get_image_channel() ) throw invalid_data_length();
    if( train_input->get_label_width() != eval_input->get_label_width() ) throw invalid_data_length();
    const auto buf_type = debug ? VMA_MEMORY_USAGE_GPU_TO_CPU : VMA_MEMORY_USAGE_GPU_ONLY;
    const unsigned int image_size = train_input->get_image_width() * train_input->get_image_height() * train_input->get_image_channel();
    const unsigned int label_size = train_input->get_label_width();
    command_buffers = liblnn::get_command_buffers( device, command_pool, 5 );
    batch_images[ 0 ].reset( new liblnn::buffer< float >( allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_size * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    batch_images[ 1 ].reset( new liblnn::buffer< float >( allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_size * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    batch_images[ 2 ].reset( new liblnn::buffer< float >( allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( image_size * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    batch_labels[ 0 ].reset( new liblnn::buffer< float >( allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( label_size * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    batch_labels[ 1 ].reset( new liblnn::buffer< float >( allocator, buf_type,
      vk::BufferCreateInfo()
        .setSize( label_size * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
    batch_labels[ 2 ].reset( new liblnn::buffer< float >( allocator, VMA_MEMORY_USAGE_GPU_TO_CPU,
      vk::BufferCreateInfo()
        .setSize( label_size * batch_size * sizeof( float ) )
        .setUsage( vk::BufferUsageFlagBits::eStorageBuffer|vk::BufferUsageFlagBits::eTransferDst )
    ) );
  }

  void network::dump(
    const std::string &filename
  ) {
    queue->waitIdle();
    auto command_buffers = liblnn::get_command_buffers( device, command_pool, 1 );
    auto &command_buffer = (*command_buffers)[ 0 ];
    std::vector< std::shared_ptr< liblnn::buffer< glm::vec4 > > > temporary_buffers;
    for( const auto &weight: weights ) {
      std::shared_ptr< liblnn::buffer< glm::vec4 > > temp( new liblnn::buffer< glm::vec4 >(
        allocator, VMA_MEMORY_USAGE_GPU_TO_CPU,
        vk::BufferCreateInfo()
          .setSize( weight.first->size() * sizeof( glm::vec4 ) )
          .setUsage( vk::BufferUsageFlagBits::eTransferDst )
      ) );
      temporary_buffers.emplace_back( std::move( temp ) );
    }
    std::vector< std::array< vk::BufferCopy, 1 > > regions;
    for( const auto &weight: weights ) {
      regions.emplace_back( std::array< vk::BufferCopy, 1 >{
        vk::BufferCopy().setSize( weight.first->size() * sizeof( glm::vec4 ) )
      } );
    }
    command_buffer.reset( vk::CommandBufferResetFlagBits::eReleaseResources );
    command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
    for( size_t index = 0u; index != weights.size(); ++index ) {
      command_buffer.copyBuffer( weights[ index ].first->get(), temporary_buffers[ index ]->get(), regions[ index ] );
    }
    command_buffer.end();
    queue->submit(
      vk::SubmitInfo()
        .setCommandBufferCount( 1 )
        .setPCommandBuffers( &command_buffer ),
      vk::Fence()
    );
    queue->waitIdle();
    int fd = open( filename.c_str(), O_WRONLY|O_CREAT, 0644 );
    if( fd == -1 ) throw unable_to_load_file();
    BOOST_SCOPE_EXIT( &fd ) {
      close( fd );
    } BOOST_SCOPE_EXIT_END
    boost::crc_32_type crc32;
    for( const auto &buf: temporary_buffers ) {
      auto head = buf->map();
      const long int length = buf->size() * sizeof( glm::vec4 );
      auto result = write( fd, reinterpret_cast< void* >( head.get() ), length );
      if( result != length ) throw unable_to_load_file();
      crc32.process_bytes( head.get(), length );
    }
    uint32_t checksum = crc32.checksum();
    auto result = write( fd, reinterpret_cast< void* >( &checksum ), sizeof( checksum ) );
    if( result != sizeof( checksum ) ) throw unable_to_load_file();
  }
  void network::restore(
    const std::string &filename
  ) {
    auto command_buffers = liblnn::get_command_buffers( device, command_pool, 1 );
    auto &command_buffer = (*command_buffers)[ 0 ];
    std::vector< std::shared_ptr< liblnn::buffer< glm::vec4 > > > temporary_buffers;
    for( const auto &weight: weights ) {
      std::shared_ptr< liblnn::buffer< glm::vec4 > > temp( new liblnn::buffer< glm::vec4 >(
        allocator, VMA_MEMORY_USAGE_CPU_TO_GPU,
        vk::BufferCreateInfo()
          .setSize( weight.first->size() * sizeof( glm::vec4 ) )
          .setUsage( vk::BufferUsageFlagBits::eTransferSrc )
      ) );
      temporary_buffers.emplace_back( std::move( temp ) );
    }
    int fd = open( filename.c_str(), O_RDONLY );
    if( fd == -1 ) throw unable_to_load_file();
    BOOST_SCOPE_EXIT( &fd ) {
      close( fd );
    } BOOST_SCOPE_EXIT_END
    boost::crc_32_type crc32;
    for( const auto &buf: temporary_buffers ) {
      auto head = buf->map();
      const long int length = buf->size() * sizeof( glm::vec4 );
      auto result = read( fd, reinterpret_cast< void* >( head.get() ), length );
      if( result != length ) throw unable_to_load_file();
      crc32.process_bytes( head.get(), length );
    }
    {
      const uint32_t expected_checksum = crc32.checksum();
      uint32_t checksum = 0;
      auto result = read( fd, reinterpret_cast< void* >( &checksum ), sizeof( checksum ) );
      if( result != sizeof( checksum ) ) throw unable_to_load_file();
      if( expected_checksum != checksum ) throw corrupted_file();
    }
    std::vector< std::array< vk::BufferCopy, 1 > > regions;
    for( const auto &weight: weights ) {
      regions.emplace_back( std::array< vk::BufferCopy, 1 >{
        vk::BufferCopy().setSize( weight.first->size() * sizeof( glm::vec4 ) )
      } );
    }
    command_buffer.reset( vk::CommandBufferResetFlagBits::eReleaseResources );
    command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
    for( size_t index = 0u; index != weights.size(); ++index ) {
      command_buffer.copyBuffer( temporary_buffers[ index ]->get(), weights[ index ].first->get(), regions[ index ] );
    }
    command_buffer.end();
    queue->submit(
      vk::SubmitInfo()
        .setCommandBufferCount( 1 )
        .setPCommandBuffers( &command_buffer ),
      vk::Fence()
    );
    queue->waitIdle();
  }
  void network::init() {
    auto command_buffers = liblnn::get_command_buffers( device, command_pool, 1 );
    auto &command_buffer = (*command_buffers)[ 0 ];
    std::vector< std::shared_ptr< layer > > layers;
    for( const auto &weight: weights ) {
      layers.emplace_back( new layer( create_init_pipeline(
        device, mods, descriptor_pool, pipeline_cache, props, weight.first, weight.second
      ) ) );
    }
    command_buffer.reset( vk::CommandBufferResetFlagBits::eReleaseResources );
    command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eSimultaneousUse ) );
    for( const auto &layer: layers )
      (*layer)( command_buffer );
    command_buffer.end();
    queue->waitIdle();
    queue->submit(
      vk::SubmitInfo()
        .setCommandBufferCount( 1 )
        .setPCommandBuffers( &command_buffer ),
      vk::Fence()
    );
    queue->waitIdle();
  }
  void network::fill( bool fill_to_eval, bool use_eval ) {
    auto &command_buffer = command_buffers->at( 3 );
    command_buffer.reset( vk::CommandBufferResetFlagBits::eReleaseResources );
    command_buffer.begin( vk::CommandBufferBeginInfo().setFlags( vk::CommandBufferUsageFlagBits::eOneTimeSubmit ) );
    auto input = use_eval ? eval_input : train_input;
    auto target = fill_to_eval ? 2 : ( swap_index + 1 ) % 2;
    (*input)( command_buffer, device, queue, props, batch_images[ target ], batch_labels[ target ] );
    command_buffer.end();
    queue->submit(
      vk::SubmitInfo()
        .setCommandBufferCount( 1 )
        .setPCommandBuffers( &command_buffer ),
      vk::Fence()
    );
  }
  void network::check() {
    std::for_each( buffers.begin(), buffers.end(), []( const auto &v ) { detect_nan( *v.second, v.first ); } );
  }
  void network::exec() {
    ++swap_index;
    swap_index %= 2;
    queue->submit(
      vk::SubmitInfo()
        .setCommandBufferCount( 1 )
        .setPCommandBuffers( command_buffers->data() + swap_index ),
      vk::Fence()
    );
    fill( false, false );
    queue->waitIdle();
    if( debug ) {
      std::cout << "==============" << std::endl;
      check();
      print( *error_out, batch_size );
      print( *output_activation_output, batch_size );
      print_image( *batch_images[ swap_index ], train_input->get_image_width(), batch_size );
      print_label( *batch_labels[ swap_index ], batch_size );
    }
  }
  void network::evaluate() {
    float train = 0.0;
    for( size_t i = 0; i != 10; ++i ) {
      fill( true, true );
      queue->submit(
        vk::SubmitInfo()
          .setCommandBufferCount( 1 )
          .setPCommandBuffers( command_buffers->data() + 2 ),
        vk::Fence()
      );
      queue->waitIdle();
      train += liblnn::evaluate( output_activation_output_eval, batch_labels[ 2 ], batch_size );
    }
    float eval = 0.0;
    for( size_t i = 0; i != 10; ++i ) {
      fill( true, false );
      queue->submit(
        vk::SubmitInfo()
          .setCommandBufferCount( 1 )
          .setPCommandBuffers( command_buffers->data() + 2 ),
        vk::Fence()
      );
      queue->waitIdle();
      eval += liblnn::evaluate( output_activation_output_eval, batch_labels[ 2 ], batch_size );
    }
    std::cout << train / 10.f << "\t" << eval / 10.f << std::endl;
  }
}

