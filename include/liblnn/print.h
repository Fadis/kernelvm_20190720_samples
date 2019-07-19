#ifndef LIBLNN_INCLUDE_PRINT_H
#define LIBLNN_INCLUDE_PRINT_H
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

#include <cstddef>
#include <liblnn/buffer.h>
namespace liblnn {
  void print( liblnn::buffer< float > &v, size_t batch_size );
  void print( liblnn::buffer< glm::vec4 > &v, size_t batch_size );
  void print_image( liblnn::buffer< float > &v, size_t width, size_t batch_size );
  void print_label( liblnn::buffer< float > &v, size_t batch_size );
  void print_eval( liblnn::buffer< float > &v, size_t batch_size );
  void detect_nan( liblnn::buffer< float > &v, const std::string &name );
}
#endif

