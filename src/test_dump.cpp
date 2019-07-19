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
  const size_t hidden_width = 1024;
  const size_t batch_size = 256;
  std::shared_ptr< liblnn::data_source > mnist_( new liblnn::mnist(
    "../../mnist/train-images-idx3-ubyte",
    "../../mnist/train-labels-idx1-ubyte"
  ) );
  std::shared_ptr< liblnn::data_source > mnist( new liblnn::input_cache( allocator, mnist_, batch_size * 100 ) );
  liblnn::conv8 network(
    command_pool,
    device,
    queue,
    descriptor_pool,
    pipeline_cache,
    props,
    allocator,
    mnist,
    mods,
    8,
    16,
    hidden_width,
    batch_size,
    false
  );
  if( std::filesystem::exists( std::filesystem::path( config.dump_file ) ) )
    network.restore( config.dump_file );
  else
    network.init();
  for( size_t i = 0; i != 60000 * 100 / batch_size; ++i ) {
    network.exec();
    if( ( i * batch_size ) % 60000 == 0 ) {
      std::cout << "dump: " << i << std::endl;
      network.dump( config.dump_file );
      std::cout << "done." << std::endl;
    }
  }
  network.dump( config.dump_file );
  std::cout << "ok" << std::endl;
}

