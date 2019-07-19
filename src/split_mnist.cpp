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
#include <memory>
#include <tuple>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <boost/scope_exit.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/karma.hpp>
#include <boost/program_options.hpp>
#include <OpenImageIO/imageio.h>
#include "liblnn/exceptions.h"
int main( int argc, const char *argv[] ) {
  namespace po = boost::program_options;
  po::options_description desc( "Options" );
  std::string input_filename;
  std::string output_dir;
  desc.add_options()
    ( "help,h", "show this message" )
    ( "input,i", po::value< std::string >(&input_filename)->default_value( "train-images-idx3-ubyte" ), "window size" )
    ( "output,o", po::value< std::string >(&output_dir)->default_value( "out" ), "use specific device" );
  po::variables_map vm;
  po::store( po::parse_command_line( argc, argv, desc ), vm );
  po::notify( vm );
  if( vm.count( "help" ) ) {
    std::cout << desc << std::endl;
    exit( 0 );
  }
  int fd = open( input_filename.c_str(), O_RDONLY );
  if( fd == -1 ) throw liblnn::unable_to_load_file();
  size_t file_size = 0;
  {
    struct stat stat_;
    if( fstat( fd, &stat_ ) == -1 ) {
      close( fd );
      throw liblnn::unable_to_load_file();
    }
    file_size = stat_.st_size;
  }
  void *addr = mmap( nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0 );
  if( !addr ) {
    close( fd );
    throw liblnn::unable_to_load_file();
  }
  std::shared_ptr< unsigned char > images(
    reinterpret_cast< unsigned char* >( addr ),
    [file_size,fd]( unsigned char *p ) {
      munmap( reinterpret_cast< void* >( p ), file_size );
      close( fd );
    }
  );
  unsigned char *image_head = nullptr;
  unsigned int image_width = 0;
  unsigned int image_height = 0;
  unsigned int image_count = 0;
  {
    auto hbegin = images.get();
    const auto hend = std::next( hbegin, 16 );
    uint32_t magic;
    boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, magic );
    if( magic != 0x803 ) throw liblnn::corrupted_file();
    boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, image_count );
    boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, image_width );
    boost::spirit::qi::parse( hbegin, hend, boost::spirit::qi::big_dword, image_height );
    image_head = hend;
    if( file_size < ( 16 + image_count * image_width * image_height ) ) throw liblnn::corrupted_file();
  }
  auto out_path = std::filesystem::path( output_dir );
  if( !std::filesystem::exists( out_path ) ) {
    std::filesystem::create_directory( out_path );
  }
  using namespace OIIO;
  ImageSpec spec (image_width, image_height, 1, TypeDesc::UINT8 );
  for( unsigned int image_index = 0; image_index != image_count; ++image_index ) {
    std::string output_filename;
    boost::spirit::karma::generate( std::back_inserter( output_filename ), boost::spirit::karma::uint_ << ".png", image_index );
    std::string output_file_path = ( out_path / output_filename );
    std::unique_ptr<ImageOutput> out( ImageOutput::create( output_file_path.c_str() ) );
    if( !out ) throw liblnn::corrupted_file();
    out->open( output_file_path.c_str(), spec );
    out->write_image( TypeDesc::UINT8, image_head + image_width * image_height * image_index );
    out->close();
  }
}
