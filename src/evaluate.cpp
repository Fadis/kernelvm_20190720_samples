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
#include <vector>
#include <algorithm>
#include <liblnn/evaluate.h>

namespace liblnn {
  float evaluate(
    std::shared_ptr< liblnn::buffer< float > > &output,
    std::shared_ptr< liblnn::buffer< float > > &expected,
    size_t batch_size
  ) {
    std::vector< size_t > output_index;
    {
      const auto data = output->map();
      const size_t length = output->size() / batch_size;
      for( size_t index = 0; index != batch_size; ++index ) {
        const auto begin = std::next( data.get(), index * length );
        const auto end = std::next( begin, length );
        const auto max_elem = std::max_element( begin, end );
	output_index.push_back( std::distance( begin, max_elem ) );
      }
    }
    std::vector< size_t > expected_index;
    {
      const auto data = expected->map();
      const size_t length = expected->size() / batch_size;
      for( size_t index = 0; index != batch_size; ++index ) {
        const auto begin = std::next( data.get(), index * length );
        const auto end = std::next( begin, length );
        const auto max_elem = std::max_element( begin, end );
	expected_index.push_back( std::distance( begin, max_elem ) );
      }
    }
    size_t positive = 0;
    size_t negative = 0;
    const size_t count = std::min( output_index.size(), expected_index.size() );
    for( size_t i = 0; i != count; ++i ) {
      if( output_index[ i ] == expected_index[ i ] )
        ++positive;
      else
        ++negative;
    }
    const float rate = float( positive ) / float( positive + negative );
    return rate;
  }
}
