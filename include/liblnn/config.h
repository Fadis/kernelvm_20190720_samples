#ifndef LIBLNN_INCLUDE_COMMON_CONFIG_H
#define LIBLNN_INCLUDE_COMMON_CONFIG_H
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

#include <string>
#include <liblnn/setter.h>

namespace liblnn {
  struct version_t {
    version_t() : major( 0 ), middle( 0 ), minor( 0 ) {}
    version_t( int a, int b, int c ) : major( a ), middle( b ), minor( c ) {}
    LIBLNN_SET_SMALL_VALUE( major )
    LIBLNN_SET_SMALL_VALUE( middle )
    LIBLNN_SET_SMALL_VALUE( minor )
    int major;
    int middle;
    int minor;
  };

  struct configs_t {
    configs_t() :
      engine_name( "liblnn" ), engine_version( 0, 0, 1 ),
      prog_name( "" ), prog_version( 0, 0, 1 ),
      list( false ),
      device_index( 0 ),
      validation( false ),
      batch_size( 0 ),
      hidden_width( 0 ),
      c1_channels( 0 ),
      c2_channels( 0 ),
      debug_mode( false ) {}
    LIBLNN_SET_LARGE_VALUE( engine_name )
    LIBLNN_SET_LARGE_VALUE( engine_version )
    LIBLNN_SET_LARGE_VALUE( prog_name )
    LIBLNN_SET_LARGE_VALUE( prog_version )
    LIBLNN_SET_SMALL_VALUE( list )
    LIBLNN_SET_SMALL_VALUE( device_index )
    LIBLNN_SET_SMALL_VALUE( validation )
    LIBLNN_SET_LARGE_VALUE( dump_file )
    LIBLNN_SET_LARGE_VALUE( train_data )
    LIBLNN_SET_LARGE_VALUE( train_label )
    LIBLNN_SET_LARGE_VALUE( eval_data )
    LIBLNN_SET_LARGE_VALUE( eval_label )
    LIBLNN_SET_LARGE_VALUE( batch_size )
    LIBLNN_SET_SMALL_VALUE( hidden_width )
    LIBLNN_SET_SMALL_VALUE( c1_channels )
    LIBLNN_SET_SMALL_VALUE( c2_channels )
    LIBLNN_SET_SMALL_VALUE( debug_mode )
    std::string engine_name;
    version_t engine_version;
    std::string prog_name; 
    version_t prog_version;
    bool list;
    unsigned int device_index;
    bool validation;
    std::string dump_file;
    std::string train_data;
    std::string train_label;
    std::string eval_data;
    std::string eval_label;
    unsigned int batch_size;
    unsigned int hidden_width;
    unsigned int c1_channels;
    unsigned int c2_channels;
    bool debug_mode;
  };
  configs_t parse_configs( int argc, const char *argv[] );
}

#endif
