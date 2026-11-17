# ─────────────────────────────────────────────────────────────────────────────
# echo-agent · C++ Cognitive Engine bindings (pybind11)
#
# Builds `cpp_memory`: a pybind11 module exposing a high-speed vector index
# (cosine top-k) and a directed cause-and-effect graph with BFS reachability,
# used by the memory plugin when the native extension is available.
#
# Build & install into the active venv:
#   cmake -S . -B build && cmake --build build --target install
# ─────────────────────────────────────────────────────────────────────────────
cmake_minimum_required(VERSION 3.15)
project(echo_agent_memory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Locate the Python interpreter (prefer the active venv's python).
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Derive pybind11's cmake dir from the installed python package.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
  RESULT_VARIABLE _pybind11_status
  OUTPUT_VARIABLE _pybind11_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(_pybind11_status EQUAL 0)
  set(pybind11_DIR "${_pybind11_dir}")
endif()
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(cpp_memory src/cpp_memory/index.cpp)

target_compile_options(cpp_memory PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O3 -march=native -funroll-loops>
  $<$<CXX_COMPILER_ID:MSVC>:/O2>
)

# Install next to the python_core package so `import cpp_memory` works.
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import sysconfig,sys;sys.stdout.write(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE _site_dir OUTPUT_STRIP_TRAILING_WHITESPACE
)
install(TARGETS cpp_memory LIBRARY DESTINATION "${_site_dir}")
