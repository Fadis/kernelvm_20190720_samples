#ifndef LIBLNN_INCLUDE_INPUT_CACHE_H
#define LIBLNN_INCLUDE_INPUT_CACHE_H
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
#include <vulkan/vulkan.hpp>
#include <vk_mem_alloc.h>
#include <liblnn/data_source.h>
namespace liblnn {
  class input_cache : public data_source {
  public:
    input_cache(
      std::shared_ptr< VmaAllocator > allocator,
      std::shared_ptr< data_source > source_,
      uint32_t
    );
    input_cache( const input_cache& ) = delete;
    input_cache( input_cache&& ) = delete;
    input_cache &operator=( const input_cache& ) = delete;
    input_cache &operator=( input_cache&& ) = delete;
    virtual uint32_t get_image_count() const { return source->get_image_count(); }
    virtual uint32_t get_image_width() const { return source->get_image_width(); }
    virtual uint32_t get_image_height() const { return source->get_image_height(); }
    virtual uint32_t get_image_channel() const { return source->get_image_channel(); }
    virtual uint32_t get_label_width() const { return source->get_label_width(); }
    void operator()(
      vk::CommandBuffer&,
      const std::shared_ptr< vk::Device >&,
      const std::shared_ptr< vk::Queue >&,
      const device_props&,
      const buffer_view< float > &image_buffer,
      const buffer_view< float > &label_buffer
    );
  private:
    std::shared_ptr< data_source > source;
    std::shared_ptr< buffer< float > > images;
    std::shared_ptr< buffer< float > > labels;
    uint32_t cache_count;
    uint32_t current_image;
  };
}
#endif

