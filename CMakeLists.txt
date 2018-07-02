# Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

cmake_minimum_required(VERSION 2.8)
project(xtensor)

find_package(xtensor REQUIRED)
find_package(xtl REQUIRED)

find_package(xtensor-io REQUIRED)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "/home/wolfv/Programs/xtensor-examples/modules")

find_package(OIIO REQUIRED)
message(STATUS "Found OpenImageIO: ${OIIO_LIBRARIES}")
include_directories(${OIIO_INCLUDE_DIRS})

find_package(xtensor-blas REQUIRED)
find_package(BLAS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/mnist/include/)
include_directories(${xtensor-blas_INCLUDE_DIRS})

add_executable(mnist_nn src/mnist.cpp)
add_executable(raytracer src/raytracer.cpp)
add_executable(kmeans src/kmeans.cpp)

target_link_libraries(mnist_nn xtensor xtl ${BLAS_LIBRARIES})
target_link_libraries(raytracer xtensor xtl ${BLAS_LIBRARIES} ${OIIO_LIBRARIES})
target_link_libraries(kmeans xtensor xtl)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/mnist)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz ${CMAKE_CURRENT_BINARY_DIR}/mnist/train-images-idx3-ubyte.gz)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz ${CMAKE_CURRENT_BINARY_DIR}/mnist/train-labels-idx1-ubyte.gz)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz ${CMAKE_CURRENT_BINARY_DIR}/mnist/t10k-images-idx3-ubyte.gz)
file(DOWNLOAD http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz ${CMAKE_CURRENT_BINARY_DIR}/mnist/t10k-labels-idx1-ubyte.gz)
execute_process(COMMAND gunzip ${CMAKE_CURRENT_BINARY_DIR}/mnist/*.gz)