cmake_minimum_required(VERSION 3.22)

project(onnx-fhe-runtime CXX)

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

if(APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set(CMAKE_INSTALL_RPATH "\$ORIGIN")
endif()

add_compile_definitions(ORT_API_MANUAL_INIT)

include_directories(${OPENMP_INCLUDES})
include_directories(${OpenFHE_INCLUDE})
include_directories(${OpenFHE_INCLUDE}/openfhe)
include_directories(${OpenFHE_INCLUDE}/openfhe/cereal)
include_directories(${OpenFHE_INCLUDE}/openfhe/core)
include_directories(${OpenFHE_INCLUDE}/openfhe/pke)
include_directories(${OpenFHE_INCLUDE}/openfhe/binfhe)
include_directories(${OpenFHE_INCLUDE}/cereal)
include_directories(${OpenFHE_INCLUDE}/core)
include_directories(${OpenFHE_INCLUDE}/pke)
include_directories(${OpenFHE_INCLUDE}/binfhe)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

link_directories(${OpenFHE_LIBDIR})
link_directories(${OPENMP_LIBRARIES})
link_libraries(OPENFHEbinfhe OPENFHEpke OPENFHEcore)

set(SOURCES
    tools/rndstr.cpp
    registry/registry.cpp
    operators/ckks/add.cpp
    operators/ckks/matmul.cpp
    operators/ckks/serialize.cpp
    operators/ckks/square.cpp
    init.cpp
)

add_library(onnx-fhe-runtime SHARED ${SOURCES})

if (OpenMP_CXX_FOUND)
    message(STATUS "FOUND OpenMP: ${OpenMP_CXX_LIBRARIES}")
    message(STATUS "FOUND OpenMP Dir: ${OpenMP_libomp_LIBRARY}")

    #workaround for macos github actions
    set(lib_openmp_path "${OpenMP_libomp_LIBRARY}")
    cmake_path(GET lib_openmp_path PARENT_PATH OMP_LINK_DIR)

    target_link_directories(onnx-fhe-runtime PRIVATE ${OMP_LINK_DIR})
endif()

target_include_directories(
    onnx-fhe-runtime SYSTEM PUBLIC 
    ${ONNXRUNTIME_INCLUDE_DIR} 
)

if(NOT SYSTEM_OPENFHE)
    message(STATUS "dep")
    add_dependencies(onnx-fhe-runtime openfhe_external)
endif()

install(TARGETS onnx-fhe-runtime)
