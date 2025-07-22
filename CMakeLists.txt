cmake_minimum_required(VERSION 3.22)

project(onnx-fhe-runtime CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)

set(OPENFHE_INSTALL_DIR ${CMAKE_BINARY_DIR}/libs/install)
set(OPENFHE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/libs/openfhe-development)
set(ONNXRUNTIME_TOP_DIR ${CMAKE_SOURCE_DIR}/libs/onnxruntime)
set(SYSTEM_OPENFHE ${USE_SYSTEM_OPENFHE} CACHE BOOL "Use system system installation of OpenFHE")

if(NOT SYSTEM_OPENFHE)
    include(ExternalProject)

    if(APPLE)
        set(OPEN_FHE_RPATH "@loader_path")
    elseif(UNIX)
        set(OPEN_FHE_RPATH "\$ORIGIN")
    endif()

    ExternalProject_Add(openfhe_external
        SOURCE_DIR ${OPENFHE_SOURCE_DIR}
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_INSTALL_PREFIX=${OPENFHE_INSTALL_DIR}
            -DLIBINSTALL=${OPEN_FHE_RPATH}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DOPENFHE_BUILD_TESTS=OFF
            -DOPENFHE_BUILD_UNITTESTS=OFF
            -DOPENFHE_BUILD_EXAMPLES=OFF
            -DOPENFHE_BUILD_BENCHMARK=OFF
        UPDATE_COMMAND ""
        BUILD_BYPRODUCTS ${OPENFHE_INSTALL_DIR}/lib/OpenFHE/OpenFHEConfig.cmake 
    )

    set(OpenFHE_INCLUDE
        ${OPENFHE_INSTALL_DIR}/include
    )

    set(OpenFHE_LIBDIR
        ${OPENFHE_INSTALL_DIR}/lib
    )
else()
    find_package(OpenFHE CONFIG REQUIRED)
    if (OpenFHE_FOUND)
        message(STATUS "FOUND PACKAGE OpenFHE")
    else()
        message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
    endif ()
endif()

message(STATUS "OPENFHE_INSTALL_DIR: ${OPENFHE_INSTALL_DIR}")
message(STATUS "OpenFHE_INCLUDE: ${OpenFHE_INCLUDE}")
message(STATUS "OpenFHE_LIBDIR: ${OpenFHE_LIBDIR}")

find_path(ONNXRUNTIME_INCLUDE_DIR
	NAMES
		onnxruntime_c_api.h
		onnxruntime_cxx_api.h
		onnxruntime_cxx_inline.h
	HINTS ${ONNXRUNTIME_TOP_DIR}
	PATH_SUFFIXES include/onnxruntime/core/session
)

if (${ONNXRUNTIME_INCLUDE_DIR} STREQUAL "ONNXRUNTIME_INCLUDE_DIR-NOTFOUND")
    message(ERROR " ONNXRuntime not found. set ONNXRUNTIME_TOP_DIR correctly")
else ()
    message(STATUS "ONNXRUNTIME_INCLUDE_DIR: ${ONNXRUNTIME_INCLUDE_DIR}")
endif ()

add_subdirectory(src)

if(NOT SYSTEM_OPENFHE)
    install(DIRECTORY ${OPENFHE_INSTALL_DIR}/lib/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        FILES_MATCHING PATTERN "*.so*" PATTERN "*.a" PATTERN "*.dylib" PATTERN "*.lib" PATTERN "*.dll")
endif()