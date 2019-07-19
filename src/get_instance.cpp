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
#include <vector>
#include <algorithm>
#include <unistd.h>
#include <vulkan/vulkan.hpp>
#include <liblnn/config.h>
#include <liblnn/instance.h>
#include <liblnn/exceptions.h>

namespace liblnn {
  std::tuple< instance_ptr_t, vk::PhysicalDevice >
  get_instance(
    const configs_t &config,
    const std::vector< const char* > &ext_,
    const std::vector< const char* > &layers_,
    const std::vector< const char* > &dext,
    const std::vector< const char* > &dlayers
  ) {
    auto ext = ext_;
    auto layers = layers_;
    if( config.validation ) layers.emplace_back( "VK_LAYER_LUNARG_standard_validation" );
    const auto app_info = vk::ApplicationInfo(
      config.prog_name.c_str(),
      VK_MAKE_VERSION(
        config.prog_version.major,
        config.prog_version.middle,
        config.prog_version.minor
      ),
      config.engine_name.c_str(),
      VK_MAKE_VERSION(
        config.prog_version.major,
        config.prog_version.middle,
        config.prog_version.minor
      ),
      VK_API_VERSION_1_1
    );
    instance_ptr_t instance(
      new vk::Instance(
        vk::createInstance(
          vk::InstanceCreateInfo()
            .setPApplicationInfo( &app_info )
            .setEnabledExtensionCount( ext.size() )
            .setPpEnabledExtensionNames( ext.data() )
            .setEnabledLayerCount( layers.size() )
            .setPpEnabledLayerNames( layers.data() )
	)
      )
    );
    auto devices = instance->enumeratePhysicalDevices();
    if( devices.empty() ) throw device_is_not_available();
    devices.erase( std::remove_if( devices.begin(), devices.end(), [&]( const auto &d ) -> bool {
      auto avail_dext = d.enumerateDeviceExtensionProperties();
      for( const char *w: dext )
        if( std::find_if( avail_dext.begin(), avail_dext.end(), [&]( const auto &v ) { return !strcmp( v.extensionName,
w ); } ) == avail_dext.end() ) return true;
      const auto avail_dlayers = d.enumerateDeviceLayerProperties();
      for( const char *w: dlayers )
        if( std::find_if( avail_dlayers.begin(), avail_dlayers.end(), [&]( const auto &v ) { return !strcmp( v.layerName, w ); } ) == avail_dlayers.end() ) return true;
      return false;
    } ), devices.end() );
    if( devices.empty() ) throw required_extensions_or_layers_are_not_available();
    if( config.list ) {
      std::cout << "利用可能なデバイス" << std::endl;
      for( unsigned int index = 0u; index != devices.size(); ++index ) {
        const auto prop = devices[ index ].getProperties();
        std::cout << index << ": " << prop.deviceName << std::endl;
      }
      exit( 0 ); 
    }
    if( config.device_index >= devices.size() ) throw device_index_is_out_of_range();
    return std::make_tuple( std::move( instance ), std::move( devices[ config.device_index ] ) );
  }
}

