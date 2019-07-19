#ifndef LIBLNN_INCLUDE_LOAD_MNIST_H
#define LIBLNN_INCLUDE_LOAD_MNIST_H
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

#include <random>
#include <tuple>
#include <memory>
#include <string>
#include <cstdint>
#include <liblnn/buffer.h>
#include <liblnn/buffer_view.h>
#include <liblnn/data_source.h>
#include <vk_mem_alloc.h>
namespace liblnn {
  class mnist : public data_source {
  public:
    mnist(
      const std::string &image_filename,
      const std::string &label_filename
    );
    mnist( const mnist& ) = delete;
    mnist( mnist&& ) = delete;
    mnist &operator=( const mnist& ) = delete;
    mnist &operator=( mnist&& ) = delete;
    virtual uint32_t get_image_count() const { return image_count; }
    virtual uint32_t get_image_width() const { return image_width; }
    virtual uint32_t get_image_height() const { return image_height; }
    virtual uint32_t get_image_channel() const { return 1; }
    virtual uint32_t get_label_width() const { return label_width; }
    virtual void operator()(
      vk::CommandBuffer&,
      const std::shared_ptr< vk::Device >&,
      const std::shared_ptr< vk::Queue >&,
      const device_props&,
      const buffer_view< float > &image_buffer,
      const buffer_view< float > &label_buffer
    );
  private:
    std::shared_ptr< unsigned char > images;
    std::shared_ptr< unsigned char > labels;
    uint32_t image_count;
    uint32_t image_width;
    uint32_t image_height;
    uint32_t label_width;
    unsigned char *image_head;
    unsigned char *label_head;
    uint32_t current_image;
    std::mt19937 gen;
  };

  std::tuple< std::shared_ptr< buffer< float > >, uint32_t >
  load_mnist_images(
    const std::shared_ptr< VmaAllocator > &allocator,
    const std::string &filename
  );
  std::tuple< std::shared_ptr< buffer< float > >, uint32_t >
  load_mnist_labels(
    const std::shared_ptr< VmaAllocator > &allocator,
    const std::string &filename
  );
}
#endif

