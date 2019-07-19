#ifndef LIBLNN_INCLUDE_EXCEPTIONS_H
#define LIBLNN_INCLUDE_EXCEPTIONS_H
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

#include <stdexcept>

namespace liblnn {
  struct vulkan_is_not_available : public std::runtime_error {
    vulkan_is_not_available() : std::runtime_error( "vulkan_is_not_available" ) {}
  };
  struct device_is_not_available : public std::runtime_error {
    device_is_not_available() : std::runtime_error( "device_is_not_available" ) {}
  };
  struct required_extensions_or_layers_are_not_available : public std::runtime_error {
    required_extensions_or_layers_are_not_available() : std::runtime_error( "required_extensions_or_layers_are_not_available" ) {}
  };
  struct device_index_is_out_of_range : public std::runtime_error {
    device_index_is_out_of_range() : std::runtime_error( "device_index_is_out_of_range" ) {}
  };
  struct required_queue_is_not_available : public std::runtime_error {
    required_queue_is_not_available() : std::runtime_error( "required_queue_is_not_available" ) {}
  };
  struct too_large_data : public std::runtime_error {
    too_large_data() : std::runtime_error( "too_large_data" ) {}
  };
  struct invalid_data_length : public std::runtime_error {
    invalid_data_length() : std::runtime_error( "invalid_data_length" ) {}
  };
  struct unable_to_load_file : public std::runtime_error {
    unable_to_load_file() : std::runtime_error( "unable_to_load_file" ) {}
  };
  struct corrupted_file : public std::runtime_error {
    corrupted_file() : std::runtime_error( "corrupted_file" ) {}
  };
}

#endif
