#ifndef LIBLNN_INCLUDE_PIPELINE_H
#define LIBLNN_INCLUDE_PIPELINE_H
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

#include <array>
#include <vector>
#include <utility>
#include <vulkan/vulkan.hpp>
#include <glm/vec4.hpp>
#include <liblnn/layer.h>
#include <liblnn/modules.h>
#include <liblnn/device_props.h>
#include <liblnn/buffer.h>
#include <liblnn/buffer_view.h>
namespace liblnn {
  layer create_init_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< glm::vec4 > &weight,
    uint32_t input_size
  );
  layer create_affine_forward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    size_t batch_size
  );
  layer create_affine_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &output_grad,
    size_t batch_size
  );
  layer create_relu_forward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value
  );
  layer create_relu_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &output_grad
  );
  layer create_tanh_forward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value
  );
  layer create_tanh_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &output_grad
  );
  layer create_softmax_combined_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &teacher_value
  );
  layer create_max_pooling_forward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t channels,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_xstride,
    uint32_t filter_ystride
  );
  layer create_max_pooling_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &output_grad,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t channels,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_xstride,
    uint32_t filter_ystride
  );
  layer create_conv_forward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t output_channels,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_channels,
    uint32_t filter_xstride,
    uint32_t filter_ystride,
    uint32_t filter_zstride,
    uint32_t input_xmargin,
    uint32_t input_ymargin
  );
  layer create_conv_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    const buffer_view< float > &output_grad,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t output_channels,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_channels,
    uint32_t filter_xstride,
    uint32_t filter_ystride,
    uint32_t filter_zstride,
    uint32_t input_xmargin,
    uint32_t input_ymargin
  );
  layer create_conv2_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &output_grad,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t output_channels,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_channels,
    uint32_t filter_xstride,
    uint32_t filter_ystride,
    uint32_t filter_zstride,
    uint32_t input_xmargin,
    uint32_t input_ymargin
  );
  layer create_conv_straight_forward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_channels,
    uint32_t filter_xstride,
    uint32_t filter_ystride,
    uint32_t filter_zstride,
    uint32_t input_xmargin,
    uint32_t input_ymargin
  );
  layer create_conv_straight_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    const buffer_view< float > &output_grad,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_channels,
    uint32_t filter_xstride,
    uint32_t filter_ystride,
    uint32_t filter_zstride,
    uint32_t input_xmargin,
    uint32_t input_ymargin
  );
  layer create_conv2_straight_backward_pipeline(
    const std::shared_ptr< vk::Device > &device,
    const modules &mods,
    const std::shared_ptr< vk::DescriptorPool > &descriptor_pool,
    const std::shared_ptr< vk::PipelineCache > &pipeline_cache,
    const device_props &props,
    const buffer_view< float > &input_value,
    const buffer_view< float > &output_value,
    const buffer_view< glm::vec4 > &weight,
    const buffer_view< float > &input_grad,
    const buffer_view< float > &output_grad,
    uint32_t output_width,
    uint32_t output_height,
    uint32_t batch_size,
    uint32_t filter_width,
    uint32_t filter_height,
    uint32_t filter_channels,
    uint32_t filter_xstride,
    uint32_t filter_ystride,
    uint32_t filter_zstride,
    uint32_t input_xmargin,
    uint32_t input_ymargin
  );
}
#endif
