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
#include <tuple>
#include <vector>
#include <algorithm>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <boost/scope_exit.hpp>
#include <boost/spirit/include/qi.hpp>
#include <liblnn/buffer.h>
#include <liblnn/load_mnist.h>
#include <liblnn/exceptions.h>
namespace liblnn {
  mnist::mnist(
    const std::string &image_filename,
    const std::string &label_filename
  ) : current_image( 0 ), gen( std::random_device()() ) {
    {
      int fd = open( image_filename.c_str(), O_RDONLY );
      if( fd == -1 ) throw unable_to_load_file();
      size_t file_size = 0;
      {
        struct stat stat_;
        if( fstat( fd, &stat_ ) == -1 ) {
          close( fd );
          throw unable_to_load_file();
        }
        file_size = stat_.st_size;
      }
      void *addr = mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0 );
      if( !addr ) {
        close( fd );
        throw unable_to_load_file();
      }
      images.reset(
        reinterpret_cast< unsigned char* >( addr ),
        [file_size,fd]( unsigned char *p ) {
          munmap( reinterpret_cast< void* >( p ), file_size );
          close( fd );
        }
      );
      {
        auto hbegin = images.get();
        const auto hend = std::next( hbegin, 16 );
        uint32_t magic;
        boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, magic );
        if( magic != 0x803 ) throw corrupted_file();
        boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, image_count );
        boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, image_width );
        boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, image_height );
        image_head = hend;
        if( file_size < ( 16 + image_count * image_width * image_height ) ) throw corrupted_file();
      }
    }
    {
      int fd = open( label_filename.c_str(), O_RDONLY );
      if( fd == -1 ) throw unable_to_load_file();
      size_t file_size = 0;
      {
        struct stat stat_;
        if( fstat( fd, &stat_ ) == -1 ) {
          close( fd );
          throw unable_to_load_file();
        }
        file_size = stat_.st_size;
      }
      void *addr = mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0 );
      if( !addr ) {
        close( fd );
        throw unable_to_load_file();
      }
      labels.reset(
        reinterpret_cast< unsigned char* >( addr ),
        [file_size,fd]( unsigned char *p ) {
          munmap( reinterpret_cast< void* >( p ), file_size );
          close( fd );
        }
      );
      {
        auto hbegin = labels.get();
        const auto hend = std::next( hbegin, 8 );
        uint32_t magic;
        boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, magic );
        if( magic != 0x801 ) throw corrupted_file();
	uint32_t label_count;
        boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, label_count );
	if( label_count != image_count ) throw invalid_data_length();
        label_head = hend;
        if( file_size < ( 8 + label_count ) ) throw corrupted_file();
	label_width = 10;
      }
    }
  }
  void mnist::operator()(
    vk::CommandBuffer&,
    const std::shared_ptr< vk::Device >&,
    const std::shared_ptr< vk::Queue >&,
    const device_props&,
    const buffer_view< float > &image_buffer,
    const buffer_view< float > &label_buffer
  ) {
    if( image_buffer.size() % ( image_width * image_height ) ) throw invalid_data_length();
    uint32_t batch_size = image_buffer.size() / ( image_width * image_height );
    if( label_buffer.size() != batch_size * 10 ) throw invalid_data_length();
    std::uniform_int_distribution< uint32_t > dist( 0u, image_count - 1u );
    std::vector< uint32_t > selected_index;
    selected_index.reserve( batch_size );
    {
      auto mapped = image_buffer.map();
      auto obegin = mapped.get();
      for( uint32_t local_image_index = 0; local_image_index != batch_size; ++local_image_index ) {
        selected_index.push_back( dist( gen ) );
        uint32_t global_image_index = selected_index.back(); // ( current_image + local_image_index ) % image_count;
	auto ibegin = image_head + global_image_index * image_width * image_height;
	auto iend = image_head + ( global_image_index + 1 ) * image_width * image_height;
        std::transform( ibegin, iend, obegin, []( unsigned char v ) { return v / 255.f; } );
	obegin += image_width * image_height;
      }
    }
    {
      auto mapped = label_buffer.map();
      auto obegin = mapped.get();
      std::fill( obegin, std::next( obegin, label_buffer.size() ), 0.f );
      for( uint32_t local_label_index = 0; local_label_index != batch_size; ++local_label_index ) {
        uint32_t global_label_index = selected_index[ local_label_index ]; // ( current_image + local_label_index ) % image_count;
	auto ibegin = label_head + global_label_index;
	obegin[ int( *ibegin ) ] = 1.f;
	obegin += 10;
      }
    }
    current_image += batch_size;
    current_image %= image_count;
  }

  std::tuple< std::shared_ptr< buffer< float > >, uint32_t >
  load_mnist_images(
    const std::shared_ptr< VmaAllocator > &allocator,
    const std::string &filename
  ) {
    int fd = open( filename.c_str(), O_RDONLY );
    if( fd == -1 ) throw unable_to_load_file();
    BOOST_SCOPE_EXIT( &fd ) {
      close( fd );
    } BOOST_SCOPE_EXIT_END
    struct stat stat_;
    if( fstat( fd, &stat_ ) == -1 ) throw unable_to_load_file();
    const size_t file_size = stat_.st_size;
    void *addr = mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0 );
    if( !addr ) throw unable_to_load_file();
    {
      BOOST_SCOPE_EXIT( &addr, &file_size ) {
        munmap( addr, file_size );
      } BOOST_SCOPE_EXIT_END
      auto hbegin = reinterpret_cast< const unsigned char* >( addr );
      const auto hend = std::next( hbegin, 16 );
      uint32_t magic;
      boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, magic );
      if( magic != 0x803 ) throw corrupted_file();
      uint32_t data_count;
      boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, data_count );
      uint32_t width = 0;
      boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, width );
      if( width != 0x1c ) throw corrupted_file();
      uint32_t height = 0;
      boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, height );
      if( height != 0x1c ) throw corrupted_file();
      auto dbegin = hend;
      size_t output_size = 32 * 32 * data_count;
      std::shared_ptr< liblnn::buffer< float > > output( new liblnn::buffer< float >( allocator, VMA_MEMORY_USAGE_CPU_TO_GPU,
        vk::BufferCreateInfo()
          .setSize( output_size * sizeof( float ) )
          .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
      {
        auto mapped = output->map();
        auto obegin = mapped.get();
	auto oend = std::next( obegin, output_size );
	std::fill( obegin, oend, 0.f );
	for( uint32_t data_index = 0; data_index != data_count; ++data_index ) {
	  for( uint32_t y = 0; y != height; ++y ) {
	    for( uint32_t x = 0; x != width; ++x ) {
	      obegin[ data_index * 32 * 32 + ( y + 2 ) * width + x + 2 ] = dbegin[ data_index * width * height + y * width + x ] / 255.f;
	    }
	  }
	}
      }
      return std::make_tuple( output, data_count ); 
    }
  }
  std::tuple< std::shared_ptr< buffer< float > >, uint32_t >
  load_mnist_labels(
    const std::shared_ptr< VmaAllocator > &allocator,
    const std::string &filename
  ) {
    int fd = open( filename.c_str(), O_RDONLY );
    if( fd == -1 ) throw unable_to_load_file();
    BOOST_SCOPE_EXIT( &fd ) {
      close( fd );
    } BOOST_SCOPE_EXIT_END
    struct stat stat_;
    if( fstat( fd, &stat_ ) == -1 ) throw unable_to_load_file();
    const size_t file_size = stat_.st_size;
    void *addr = mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0 );
    if( !addr ) throw unable_to_load_file();
    {
      BOOST_SCOPE_EXIT( &addr, &file_size ) {
        munmap( addr, file_size );
      } BOOST_SCOPE_EXIT_END
      auto hbegin = reinterpret_cast< const unsigned char* >( addr );
      const auto hend = std::next( hbegin, 16 );
      uint32_t magic;
      boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, magic );
      if( magic != 0x801 ) throw corrupted_file();
      uint32_t data_count;
      boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, data_count );
      auto dbegin = hend;
      size_t output_size = 10 * data_count;
      std::shared_ptr< liblnn::buffer< float > > output( new liblnn::buffer< float >( allocator, VMA_MEMORY_USAGE_CPU_TO_GPU,
        vk::BufferCreateInfo()
          .setSize( output_size * sizeof( float ) )
          .setUsage( vk::BufferUsageFlagBits::eStorageBuffer )
      ) );
      {
        auto mapped = output->map();
        auto obegin = mapped.get();
	auto oend = std::next( obegin, output_size );
	std::fill( obegin, oend, 0.f );
	for( uint32_t data_index = 0; data_index != data_count; ++data_index ) {
	  obegin[ data_index * 10 + dbegin[ data_index ] ] = 1.f;
	}
      }
      return std::make_tuple( output, data_count ); 
    }
  }
}
