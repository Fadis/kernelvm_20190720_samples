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

#include <cmath>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <glm/vec4.hpp>
#include <liblnn/print.h>
namespace liblnn {
  void print( liblnn::buffer< float > &v, size_t batch_size ) {
    auto head = v.map();
    int i = 0;
    std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( float v ) { std::cout << v << "\t"; if( !( ++i % 4 ) ) std::cout << std::endl; } );
    std::cout << std::endl;
  }
  void print( liblnn::buffer< glm::vec4 > &v, size_t batch_size ) {
    auto head = v.map();
    int i = 0;
    std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( glm::vec4 v ) { std::cout << v.x << ',' << v.y << ',' << v.z << ',' << v.w << "\t"; if( !( ++i % 4 ) ) std::cout << std::endl; } );
    std::cout << std::endl;
  }
  void print_image( liblnn::buffer< float > &v, size_t width, size_t batch_size ) {
    auto head = v.map();
    int i = 0;
    std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( float v ) {
     if( v <= 0.125f ) std::cout << ' ';
     else if( v <= 0.25f ) std::cout << '.';
     else if( v <= 0.25f ) std::cout << ':';
     else if( v <= 0.375f ) std::cout << '=';
     else if( v <= 0.5f ) std::cout << 'u';
     else if( v <= 0.625f ) std::cout << 'h';
     else if( v <= 0.75f ) std::cout << 'K';
     else if( v <= 0.875f ) std::cout << '@';
     else std::cout << '#';
     if( !( ++i % width ) ) std::cout << std::endl;
    } );
    std::cout << std::endl;
  }
  void print_label( liblnn::buffer< float > &v, size_t batch_size ) {
    auto head = v.map();
    int i = 0;
    std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( float v ) {
     if( v == 1.0f ) std::cout << i;
     ++i;
    } );
    std::cout << std::endl;
  }
  void print_eval( liblnn::buffer< float > &v, size_t batch_size ) {
    auto head = v.map();
    int i = 0;
    float max = std::numeric_limits< float >::min();
    int max_index = 0;
    std::for_each( head.get(), std::next( head.get(), v.size() / batch_size ), [&]( float v ) {
     if( v > max ) {
       max_index = i;
       max = v;
     }
     ++i;
    } );
    std::cout << max_index << std::endl;
  }
  void detect_nan( liblnn::buffer< float > &v, const std::string &name ) {
    auto head = v.map();
    auto end = std::next( head.get(), v.size() );
    const auto detected = std::find_if( head.get(), end, []( float v ) { return std::isnan( v ); } );
    if( detected != end )
      std::cout <<  "NaN at " << name << " " << std::distance( head.get(), detected ) << std::endl;
  }
}

