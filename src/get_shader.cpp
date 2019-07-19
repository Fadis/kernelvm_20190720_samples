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

#include <vector>
#include <iterator>
#include <fstream>
#include <liblnn/shader.h>

namespace liblnn {
  std::shared_ptr< vk::ShaderModule >
  get_shader(
    const std::shared_ptr< vk::Device > &device,
    const std::string &filename
  ) {
    std::fstream file( filename, std::ios::in|std::ios::binary );
    const std::vector< char > bin( ( std::istreambuf_iterator< char >( file ) ), std::istreambuf_iterator<char>() ); 
    auto module = device->createShaderModule(
      vk::ShaderModuleCreateInfo().setCodeSize( bin.size() ).setPCode( reinterpret_cast< const uint32_t* >( bin.data() ) )
    );
    return std::shared_ptr< vk::ShaderModule >(
      new vk::ShaderModule( std::move( module ) ),
      [device]( vk::ShaderModule *p ) {
        if( p ) {
          device->destroyShaderModule( *p );
          delete p;
        }
      }
    );
  }
}

