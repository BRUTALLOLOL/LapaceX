cmake_minimum_required(VERSION 3.18)
project(LaplaceX VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /W3)
  add_compile_definitions(_USE_MATH_DEFINES)
else()
  add_compile_options(-O3 -Wall -Wextra -Wpedantic)
  if(NOT CMAKE_CROSSCOMPILING)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
      add_compile_options(-march=armv8-a)
    else()
      add_compile_options(-march=native)
    endif()
  endif()
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_laplacex
    cpp/src/core.cpp
    cpp/src/critical.cpp
    cpp/src/utils.cpp
    cpp/src/bindings.cpp
)

target_include_directories(_laplacex PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

if(MSVC)
  target_compile_options(_laplacex PRIVATE /MP)
endif()

install(TARGETS _laplacex DESTINATION laplacex)
