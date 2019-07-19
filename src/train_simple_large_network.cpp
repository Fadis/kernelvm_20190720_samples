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

#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <boost/math/common_factor_rt.hpp>
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <glm/vec4.hpp>
#include <liblnn/config.h>
#include <liblnn/instance.h>
#include <liblnn/device.h>
#include <liblnn/shader.h>
#include <liblnn/command_buffer.h>
#include <liblnn/modules.h>
#include <liblnn/device_props.h>
#include <liblnn/layer_def.h>
#include <liblnn/layer.h>
#include <liblnn/pipeline_cache.h>
#include <liblnn/descriptor_pool.h>
#include <liblnn/descriptor_set.h>
#include <liblnn/pipeline_layout.h>
#include <liblnn/allocator.h>
#include <liblnn/buffer.h>
#include <liblnn/pipeline.h>
#include <liblnn/load_mnist.h>
#include <liblnn/data_source.h>
#include <liblnn/input_cache.h>
#include <liblnn/network.h>

void dump( const liblnn::buffer_view< float > &v, size_t batch_size ) {
  auto head = v.map();
  int i = 0;
  std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( float v ) { std::cout << v << "\t"; if( !( ++i % 16 ) ) std::cout << std::endl; } );
  std::cout << std::endl;
}
void dump( const liblnn::buffer_view< glm::vec4 > &v, size_t batch_size ) {
  auto head = v.map();
  int i = 0;
  std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( glm::vec4 v ) { std::cout << v.x << ',' << v.y << ',' << v.z << ',' << v.w << "\t"; if( !( ++i % 16 ) ) std::cout << std::endl; } );
  std::cout << std::endl;
}


int main( int argc, const char *argv[] ) {
  auto config = liblnn::parse_configs( argc, argv );
  auto [instance,physical_device] = liblnn::get_instance(
    config,
    {},{},
    {},{}
  );
  const auto props = liblnn::get_device_props( physical_device );
  auto [device,queue,command_pool] = liblnn::get_device(
    config, physical_device, {}, {}
  );
  std::vector< vk::DescriptorPoolSize > descriptor_pool_size{
    vk::DescriptorPoolSize().setType( vk::DescriptorType::eStorageBuffer ).setDescriptorCount( 2 )
  };
  auto descriptor_pool = liblnn::get_descriptior_pool( device, descriptor_pool_size, 100 );
  auto pipeline_cache = liblnn::get_pipeline_cache( device );

  liblnn::modules mods( device );

  auto allocator = liblnn::get_allocator( physical_device, device );
  const size_t hidden_width = 4096;
  const size_t batch_size = 256;
  std::shared_ptr< liblnn::mnist > tin_( new liblnn::mnist(
    config.train_data,
    config.train_label
  ) );
  std::shared_ptr< liblnn::mnist > ein_( new liblnn::mnist(
    config.eval_data,
    config.eval_label
  ) );
  std::shared_ptr< liblnn::input_cache > tin( new liblnn::input_cache( allocator, tin_, batch_size * 100 ) );
  std::shared_ptr< liblnn::input_cache > ein( new liblnn::input_cache( allocator, ein_, batch_size * 10 ) );
  liblnn::simple network(
    command_pool,
    device,
    queue,
    descriptor_pool,
    pipeline_cache,
    props,
    allocator,
    tin,
    ein,
    mods,
    hidden_width,
    batch_size,
    config.debug_mode
  );
  if( std::filesystem::exists( std::filesystem::path( config.dump_file ) ) ) {
    std::cout << "restart from " << config.dump_file << std::endl;
    network.restore( config.dump_file );
  }
  else
    network.init();
  for( size_t i = 0; i != 60000 * 1000; i += batch_size ) {
    network.exec();
    if( ( i + batch_size ) % 60000 < batch_size ) {
      network.evaluate();
      std::cout << "saved: " << i << std::endl;
      network.dump( config.dump_file );
      std::cout << "done." << std::endl;
    }
  }
  network.dump( config.dump_file );
  std::cout << "ok" << std::endl;
}

