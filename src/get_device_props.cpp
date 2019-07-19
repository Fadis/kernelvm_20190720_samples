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

#include <liblnn/device_props.h>
namespace liblnn {
  device_props get_device_props(
    const vk::PhysicalDevice &physical_device
  ) {
    auto subgroup_props = vk::PhysicalDeviceSubgroupProperties();
    auto props2 = vk::PhysicalDeviceProperties2();
    props2.pNext = &subgroup_props;
    physical_device.getProperties2( &props2 );
    auto props = vk::PhysicalDeviceProperties();
    physical_device.getProperties( &props );
    return device_props()
      .set_props( props )
      .set_props2( props2 )
      .set_subgroup_props( subgroup_props );
  }
}

