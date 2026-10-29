cmake_minimum_required(VERSION 3.19...3.28)
project(InferSSD LANGUAGES C CXX)

# Keep product build commands and output paths stable while the bundled engine
# retains its own source-relative layout and can also be built directly.
option(LLAMA_BUILD_COMMON "Build shared command-line support" ON)
option(LLAMA_BUILD_TOOLS "Build engine tools" ON)
option(LLAMA_BUILD_SERVER "Build the inference server" ON)
option(LLAMA_BUILD_TESTS "Build engine contract tests" ON)
option(LLAMA_BUILD_EXAMPLES "Build upstream examples" OFF)
option(LLAMA_TOOLS_INSTALL "Install the bundled engine tools" OFF)
option(LLAMA_TESTS_INSTALL "Install engine contract tests" OFF)
include(CTest)
add_subdirectory(engine)
