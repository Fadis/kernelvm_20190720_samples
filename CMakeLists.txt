cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
PROJECT(vulkan_samples_2019)
include(CheckIncludeFiles)
include(FindPkgConfig)
include(CheckCXXSourceCompiles)

# build type
#set(CMAKE_BUILD_TYPE Debug)

# compile option (Debug) 
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++17 -g -Wall -Wextra -Werror -pthread")
set(CMAKE_C_FLAGS_DEBUG "-std=c11 -g -Wall -Wextra -Werror -pthread")

# compile option (Release)
set(CMAKE_CXX_FLAGS_RELEASE "-march=native -std=c++17 -W -Wall -Wextra -Werror -O3 -pthread")
set(CMAKE_C_FLAGS_RELEASE "-march=native -std=c11 -W -Wall -Wextra -Werror -O3 -pthread")

#if you don't want the full compiler output, remove the following line
set(CMAKE_VERBOSE_MAKEFILE OFF)

find_package(Threads REQUIRED)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost 1.65.0 COMPONENTS filesystem program_options system REQUIRED )
find_package(Vulkan REQUIRED)
find_package(OpenImageIO REQUIRED)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/include
	${CMAKE_CURRENT_BINARY_DIR}/include ${Boost_INCLUDE_DIRS}
	${Vulkan_INCLUDE_DIRS} ${OIIO_INCLUDE_DIR} )
link_directories( ${Boost_LIBRARY_DIRS} ${SPDK_LIBRARY_DIRS}
	${OIIO_LIBRARY_DIR} )
subdirs( include src )

