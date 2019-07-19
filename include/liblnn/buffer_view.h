#ifndef LIBLNN_INCLUDE_BUFFER_VIEW_H
#define LIBLNN_INCLUDE_BUFFER_VIEW_H
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

#include <memory>
#include <stdexcept>
#include <liblnn/buffer.h>
namespace liblnn {
  template< typename T >
  class buffer_view {
  public:
    buffer_view() : offset_( 0 ), length( 0 ) {}
    buffer_view(
      const std::shared_ptr< buffer< T > > &buf_,
      uint32_t offset__,
      uint32_t length_
    ) : buf( buf_ ), offset_( offset__ ), length( length_ ) {
      if( buf->size() < offset_ ) throw std::out_of_range( "buffer_view: Offset must smaller than buffer length." );
      if( buf->size() - offset_ < length ) throw std::out_of_range( "buffer_view: Length must smaller than buffer length - offset." );
    }
    buffer_view(
      const std::shared_ptr< buffer< T > > &buf_
    ) : buf( buf_ ), offset_( 0 ), length( buf_->size() ) {
      if( buf->size() < offset_ ) throw std::out_of_range( "buffer_view: Offset must smaller than buffer length." );
      if( buf->size() - offset_ < length ) throw std::out_of_range( "buffer_view: Length must smaller than buffer length - offset." );
    }
    std::shared_ptr< T > map() const {
      if( !buf ) throw std::out_of_range( "buffer_view: Uninitialized buffer_view." );
      auto mapped = buf->map();
      return std::shared_ptr< T >( std::next( mapped.get(), offset_ ), [mapped](T*){} );
    }
    size_t size() const { return length; }
    size_t offset() const { return offset_; }
    VkBuffer &get() const {
      if( !buf ) throw std::out_of_range( "buffer_view: Uninitialized buffer_view." );
      return buf->get();
    }
    operator bool() const {
      return bool( buf );
    }
  private:
    std::shared_ptr< buffer< T > > buf;
    size_t offset_;
    size_t length;
  };
}
#endif

