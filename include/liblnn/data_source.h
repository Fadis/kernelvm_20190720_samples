#ifndef LIBLNN_INCLUDE_LOAD_DATA_SOURCE_H
#define LIBLNN_INCLUDE_LOAD_DATA_SOURCE_H
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

#include <cstdint>
#include <memory>
#include <vulkan/vulkan.hpp>
#include <liblnn/device_props.h>
#include <liblnn/buffer_view.h>
namespace liblnn {
  class data_source {
  public:
    virtual ~data_source() {}
    virtual void operator()(
      vk::CommandBuffer&,
      const std::shared_ptr< vk::Device >&,
      const std::shared_ptr< vk::Queue >&,
      const device_props&,
      const buffer_view< float > &image_buffer,
      const buffer_view< float > &label_buffer
    ) = 0;
    virtual uint32_t get_image_count() const = 0;
    virtual uint32_t get_image_width() const = 0;
    virtual uint32_t get_image_height() const = 0;
    virtual uint32_t get_image_channel() const = 0;
    virtual uint32_t get_label_width() const = 0;
  };
}

#endif

