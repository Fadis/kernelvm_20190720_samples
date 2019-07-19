add_library( lnn SHARED config.cpp get_instance.cpp get_device.cpp
	get_shader.cpp get_command_buffer.cpp get_device_props.cpp modules.cpp
	get_descriptor_pool.cpp get_pipeline_cache.cpp get_descriptor_set.cpp
	get_pipeline_layout.cpp get_allocator.cpp create_init_pipeline.cpp
	layer.cpp create_affine_forward_pipeline.cpp
	create_relu_forward_pipeline.cpp create_softmax_combined_pipeline.cpp
	create_affine_backward_pipeline.cpp load_mnist.cpp
	create_relu_backward_pipeline.cpp simple_network.cpp input_cache.cpp
	create_max_pooling_forward_pipeline.cpp
	create_max_pooling_backward_pipeline.cpp
	create_conv_forward_pipeline.cpp create_conv_backward_pipeline.cpp
	create_conv2_backward_pipeline.cpp
	create_conv_straight_forward_pipeline.cpp
	create_conv_straight_backward_pipeline.cpp
	create_conv2_straight_backward_pipeline.cpp print.cpp conv_network.cpp
	create_tanh_forward_pipeline.cpp create_tanh_backward_pipeline.cpp
	evaluate.cpp conv3_network.cpp conv4_network.cpp conv4x_network.cpp
	conv5_network.cpp conv6_network.cpp conv10_network.cpp network.cpp vma.cpp )
target_link_libraries( lnn ${Boost_PROGRAM_OPTIONS_LIBRARIES}
	${Boost_SYSTEM_LIBRARIES} ${OIIO_LIBRARIES} stdc++fs )
add_executable( train_simple_network train_simple_network.cpp )
target_link_libraries( train_simple_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_simple_large_network train_simple_large_network.cpp )
target_link_libraries( train_simple_large_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv_network train_conv_network.cpp )
target_link_libraries( train_conv_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv3_network train_conv3_network.cpp )
target_link_libraries( train_conv3_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv4_network train_conv4_network.cpp )
target_link_libraries( train_conv4_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv4x_network train_conv4x_network.cpp )
target_link_libraries( train_conv4x_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv5_network train_conv5_network.cpp )
target_link_libraries( train_conv5_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv6_network train_conv6_network.cpp )
target_link_libraries( train_conv6_network lnn ${Vulkan_LIBRARIES} )
add_executable( train_conv10_network train_conv10_network.cpp )
target_link_libraries( train_conv10_network lnn ${Vulkan_LIBRARIES} )
add_executable( split_mnist split_mnist.cpp )
target_link_libraries( split_mnist
	${Boost_PROGRAM_OPTIONS_LIBRARIES} ${Boost_SYSTEM_LIBRARIES}
	${OIIO_LIBRARIES} stdc++fs )
