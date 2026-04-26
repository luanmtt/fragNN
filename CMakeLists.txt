cmake_minimum_required(VERSION 3.24)
project(FragNN LANGUAGES CXX CUDA)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
 
find_package(CUDAToolkit REQUIRED)
 
add_executable(FragNN
    main.cpp
    src/dataloader.cpp
    src/nn.cpp
    cuda/kernel.cu
)
 
target_include_directories(FragNN PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/cuda
)
 
target_link_libraries(FragNN PRIVATE
    CUDA::cudart
)
 
set_target_properties(FragNN PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES         native
    LINKER_LANGUAGE            CUDA)
 
