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
#include <cstdlib>
#include <memory>
#include <vector>
#include <liblnn/config.h>
#include <liblnn/device.h>
#include <liblnn/exceptions.h>

namespace liblnn {
  std::tuple<
    std::shared_ptr< vk::Device >,
    std::shared_ptr< vk::Queue >,
    std::shared_ptr< vk::CommandPool >
  > get_device(
    const configs_t&,
    const vk::PhysicalDevice &physical_device,
    const std::vector< const char* > &dext,
    const std::vector< const char* > &dlayers
  ) {
    const auto queue_props = physical_device.getQueueFamilyProperties();
    uint32_t queue_index =std::distance( queue_props.begin(), std::find_if( queue_props.begin(), queue_props.end(), []( const auto &v ) { return bool( v.queueFlags & vk::QueueFlagBits::eCompute ) && bool( v.queueFlags & vk::QueueFlagBits::eTransfer ); } ) );
    if( queue_index == queue_props.size() ) throw required_queue_is_not_available();
    const float priority = 0.0f;
    std::vector< vk::DeviceQueueCreateInfo > queues{
    };
    const auto queue_create_info =
      vk::DeviceQueueCreateInfo()
        .setQueueFamilyIndex( queue_index ).setQueueCount( 1 ).setPQueuePriorities( &priority );
    const auto features = physical_device.getFeatures();
    auto device = physical_device.createDevice(
      vk::DeviceCreateInfo()
        .setQueueCreateInfoCount( 1 )
        .setPQueueCreateInfos( &queue_create_info )
        .setEnabledExtensionCount( dext.size() )
        .setPpEnabledExtensionNames( dext.data() )
        .setEnabledLayerCount( dlayers.size() )
        .setPpEnabledLayerNames( dlayers.data() )
        .setPEnabledFeatures( &features )
    );
    std::shared_ptr< vk::Device > d(
      new vk::Device( std::move( device ) ),
      []( const auto &p ) {
        if( p ) {
          p->destroy();
          delete p;
        }
      }
    );
    auto queue = device.getQueue( queue_index, 0 );
    auto command_pool = device.createCommandPool(
      vk::CommandPoolCreateInfo().setQueueFamilyIndex( queue_index ).setFlags( vk::CommandPoolCreateFlagBits::eResetCommandBuffer )
    );
    std::shared_ptr< vk::Queue > q( new vk::Queue( std::move( queue ) ), [d]( const auto& ) {} );
    std::shared_ptr< vk::CommandPool > p(
      new vk::CommandPool( std::move( command_pool ) ),
      [d]( const vk::CommandPool *p ) {
        if( p ) {
          d->destroyCommandPool( *p );
          delete p;
        }
      }
    );
    return std::make_tuple( d, q, p );
  }
}


