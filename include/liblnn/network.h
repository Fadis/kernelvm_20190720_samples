#ifndef LIBLNN_INCLUDE_NETWORK_H
#define LIBLNN_INCLUDE_NETWORK_H
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
#include <memory>
#include <array>
#include <string>
#include <functional>
#include <vector>
#include <boost/container/flat_map.hpp>
#include <vk_mem_alloc.h>
#include <vulkan/vulkan.hpp>
#include <glm/vec4.hpp>
#include <liblnn/buffer.h>
#include <liblnn/device_props.h>
#include <liblnn/modules.h>
#include <liblnn/data_source.h>
#include <liblnn/layer.h>
namespace liblnn {
  class network {
  public:
    network(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t batch_size_,
      bool debug_
    );
    virtual ~network() {}
    void exec();
    void evaluate();
    void dump( const std::string &filename );
    void restore( const std::string &filename );
    void init();
  protected:
    void fill( bool, bool );
    void check();
    std::vector< std::pair< std::shared_ptr< liblnn::buffer< glm::vec4 > >, uint32_t > > weights;
    std::shared_ptr< vk::CommandPool > command_pool;
    std::shared_ptr< vk::Device > device;
    std::shared_ptr< vk::Queue > queue;
    std::shared_ptr< vk::DescriptorPool > descriptor_pool;
    std::shared_ptr< vk::PipelineCache > pipeline_cache;
    device_props props;
    std::shared_ptr< VmaAllocator > allocator;
    std::shared_ptr< data_source > train_input;
    std::shared_ptr< data_source > eval_input;
    liblnn::modules mods;
    size_t batch_size;
    bool debug;
    size_t swap_index;
    std::shared_ptr< std::vector< vk::CommandBuffer > > command_buffers;
    std::array< std::shared_ptr< liblnn::buffer< float > >, 3 > batch_images;
    std::array< std::shared_ptr< liblnn::buffer< float > >, 3 > batch_labels;
    boost::container::flat_map< std::string, std::shared_ptr< liblnn::buffer< float > > > buffers;
    std::shared_ptr< liblnn::buffer< float > > output_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_activation_output_eval;
    std::shared_ptr< liblnn::buffer< float > > error_out;
  };
  class simple : public network {
  public:
    simple(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > hidden_affine1;
    std::shared_ptr< layer > hidden_affine2;
    std::shared_ptr< layer > hidden_affine3;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward1;
    std::shared_ptr< layer > hidden_affine_backward2;
  };
  class conv3 : public network {
  public:
    conv3(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > init_c1_conv1_weight;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
  class conv4 : public network {
  public:
    conv4(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > init_c1_conv1_weight;
    std::shared_ptr< layer > init_c1_conv2_weight;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > c1_conv2;
    std::shared_ptr< layer > c1_activation2;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c1_activation2_backward;
    std::shared_ptr< layer > c1_conv2_bp_backward;
    std::shared_ptr< layer > c1_conv2_update_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
  class conv4x : public network {
  public:
    conv4x(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > init_c1_conv1_weight;
    std::shared_ptr< layer > init_c1_conv2_weight;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > c1_conv2;
    std::shared_ptr< layer > c1_activation2;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c1_activation2_backward;
    std::shared_ptr< layer > c1_conv2_bp_backward;
    std::shared_ptr< layer > c1_conv2_update_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
  class conv5 : public network {
  public:
    conv5(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_width;
    size_t c1_height;
    size_t c1_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > init_c1_conv1_weight;
    std::shared_ptr< layer > init_c1_conv2_weight;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > c1_conv2;
    std::shared_ptr< layer > c1_activation2;
    std::shared_ptr< layer > c1_mp;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c1_mp_backward;
    std::shared_ptr< layer > c1_activation2_backward;
    std::shared_ptr< layer > c1_conv2_bp_backward;
    std::shared_ptr< layer > c1_conv2_update_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
  class conv6 : public network {
  public:
    conv6(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_width;
    size_t c1_height;
    size_t c1_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv3_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv3_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation3_output;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation3_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv3_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > init_c1_conv1_weight;
    std::shared_ptr< layer > init_c1_conv2_weight;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > c1_conv2;
    std::shared_ptr< layer > c1_activation2;
    std::shared_ptr< layer > c1_conv3;
    std::shared_ptr< layer > c1_activation3;
    std::shared_ptr< layer > c1_mp;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c1_mp_backward;
    std::shared_ptr< layer > c1_activation3_backward;
    std::shared_ptr< layer > c1_conv3_bp_backward;
    std::shared_ptr< layer > c1_conv3_update_backward;
    std::shared_ptr< layer > c1_activation2_backward;
    std::shared_ptr< layer > c1_conv2_bp_backward;
    std::shared_ptr< layer > c1_conv2_update_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
  class conv8 : public network {
  public:
    conv8(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t c1_channels_,
      size_t c2_channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_width;
    size_t c1_height;
    size_t c1_channels;
    size_t c2_width;
    size_t c2_height;
    size_t c2_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c2_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c2_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_output;
    std::shared_ptr< liblnn::buffer< float > > c2_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c2_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c2_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c2_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > c2_mp_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_mp_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_conv1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > init_c1_conv1_weight;
    std::shared_ptr< layer > init_c1_conv2_weight;
    std::shared_ptr< layer > init_c2_conv1_weight;
    std::shared_ptr< layer > init_c2_conv2_weight;
    std::shared_ptr< layer > init_hidden_weight;
    std::shared_ptr< layer > init_output_weight;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > c1_conv2;
    std::shared_ptr< layer > c1_activation2;
    std::shared_ptr< layer > c1_mp;
    std::shared_ptr< layer > c2_conv1;
    std::shared_ptr< layer > c2_activation1;
    std::shared_ptr< layer > c2_conv2;
    std::shared_ptr< layer > c2_activation2;
    std::shared_ptr< layer > c2_mp;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c2_mp_backward;
    std::shared_ptr< layer > c2_activation2_backward;
    std::shared_ptr< layer > c2_conv2_bp_backward;
    std::shared_ptr< layer > c2_conv2_update_backward;
    std::shared_ptr< layer > c2_activation1_backward;
    std::shared_ptr< layer > c2_conv1_bp_backward;
    std::shared_ptr< layer > c2_conv1_update_backward;
    std::shared_ptr< layer > c1_mp_backward;
    std::shared_ptr< layer > c1_activation2_backward;
    std::shared_ptr< layer > c1_conv2_bp_backward;
    std::shared_ptr< layer > c1_conv2_update_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
  class conv10 : public network {
  public:
    conv10(
      const std::shared_ptr< vk::CommandPool > &command_pool_,
      const std::shared_ptr< vk::Device > &device_,
      const std::shared_ptr< vk::Queue > &queue_,
      const std::shared_ptr< vk::DescriptorPool > &descriptor_pool_,
      const std::shared_ptr< vk::PipelineCache > &pipeline_cache_,
      const device_props &props_,
      const std::shared_ptr< VmaAllocator > &allocator_,
      const std::shared_ptr< data_source > &tin_,
      const std::shared_ptr< data_source > &ein_,
      const liblnn::modules &mods,
      size_t c1_channels_,
      size_t c2_channels_,
      size_t hidden_width_,
      size_t batch_size_,
      bool debug_
    );
  private:
    size_t image_width;
    size_t image_height;
    size_t image_channels;
    size_t c1_width;
    size_t c1_height;
    size_t c1_channels;
    size_t c2_width;
    size_t c2_height;
    size_t c2_channels;
    size_t hidden_width;
    size_t output_width;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c1_conv3_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c2_conv1_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c2_conv2_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > c2_conv3_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > hidden_weight;
    std::shared_ptr< liblnn::buffer< glm::vec4 > > output_weight;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > c1_conv3_output;
    std::shared_ptr< liblnn::buffer< float > > c1_activation3_output;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_output;
    std::shared_ptr< liblnn::buffer< float > > c2_conv1_output;
    std::shared_ptr< liblnn::buffer< float > > c2_activation1_output;
    std::shared_ptr< liblnn::buffer< float > > c2_conv2_output;
    std::shared_ptr< liblnn::buffer< float > > c2_activation2_output;
    std::shared_ptr< liblnn::buffer< float > > c2_conv3_output;
    std::shared_ptr< liblnn::buffer< float > > c2_activation3_output;
    std::shared_ptr< liblnn::buffer< float > > c2_mp_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_output;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_output;
    std::shared_ptr< liblnn::buffer< float > > output_affine_output;
    std::shared_ptr< liblnn::buffer< float > > softmax_grad;
    std::shared_ptr< liblnn::buffer< float > > output_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > output_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_activation_grad;
    std::shared_ptr< liblnn::buffer< float > > hidden_affine_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_mp_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_activation3_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_conv3_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c2_conv1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_mp_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation3_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv3_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv2_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_activation1_grad;
    std::shared_ptr< liblnn::buffer< float > > c1_conv1_grad;
    std::shared_ptr< layer > c1_conv1_1;
    std::shared_ptr< layer > c1_conv1_2;
    std::shared_ptr< layer > c1_conv1_3;
    std::shared_ptr< layer > c1_activation1;
    std::shared_ptr< layer > c1_conv2;
    std::shared_ptr< layer > c1_activation2;
    std::shared_ptr< layer > c1_conv3;
    std::shared_ptr< layer > c1_activation3;
    std::shared_ptr< layer > c1_mp;
    std::shared_ptr< layer > c2_conv1;
    std::shared_ptr< layer > c2_activation1;
    std::shared_ptr< layer > c2_conv2;
    std::shared_ptr< layer > c2_activation2;
    std::shared_ptr< layer > c2_conv3;
    std::shared_ptr< layer > c2_activation3;
    std::shared_ptr< layer > c2_mp;
    std::shared_ptr< layer > hidden_affine;
    std::shared_ptr< layer > hidden_activation;
    std::shared_ptr< layer > output_affine;
    std::shared_ptr< layer > output_activation;
    std::shared_ptr< layer > output_activation3;
    std::shared_ptr< layer > error1;
    std::shared_ptr< layer > error2;
    std::shared_ptr< layer > output_activation_backward;
    std::shared_ptr< layer > output_affine_backward;
    std::shared_ptr< layer > hidden_activation_backward;
    std::shared_ptr< layer > hidden_affine_backward;
    std::shared_ptr< layer > c2_mp_backward;
    std::shared_ptr< layer > c2_activation3_backward;
    std::shared_ptr< layer > c2_conv3_bp_backward;
    std::shared_ptr< layer > c2_conv3_update_backward;
    std::shared_ptr< layer > c2_activation2_backward;
    std::shared_ptr< layer > c2_conv2_bp_backward;
    std::shared_ptr< layer > c2_conv2_update_backward;
    std::shared_ptr< layer > c2_activation1_backward;
    std::shared_ptr< layer > c2_conv1_bp_backward;
    std::shared_ptr< layer > c2_conv1_update_backward;
    std::shared_ptr< layer > c1_mp_backward;
    std::shared_ptr< layer > c1_activation3_backward;
    std::shared_ptr< layer > c1_conv3_bp_backward;
    std::shared_ptr< layer > c1_conv3_update_backward;
    std::shared_ptr< layer > c1_activation2_backward;
    std::shared_ptr< layer > c1_conv2_bp_backward;
    std::shared_ptr< layer > c1_conv2_update_backward;
    std::shared_ptr< layer > c1_activation1_backward;
    std::shared_ptr< layer > c1_conv1_bp_backward_2;
    std::shared_ptr< layer > c1_conv1_update_backward_2;
    std::shared_ptr< layer > c1_conv1_bp_backward_1;
    std::shared_ptr< layer > c1_conv1_update_backward_1;
  };
}
#endif

