# The per-OS platform layer: one interface, one implementation per OS. Every
# InferSSD consumer reaches the OS only through this library.
add_library(ggml-ssd-platform STATIC)
target_compile_features(ggml-ssd-platform PUBLIC cxx_std_17)
target_include_directories(ggml-ssd-platform PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(ggml-ssd-platform PROPERTIES POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(ggml-ssd-platform PUBLIC Threads::Threads)
if (MSVC)
    target_compile_options(ggml-ssd-platform PRIVATE /W4)
else()
    target_compile_options(ggml-ssd-platform PRIVATE -Wall -Wextra -Wpedantic -Wshadow)
endif()

# Thread placement (CPU topology, pinning, roles) and arena memory primitives.
target_sources(ggml-ssd-platform PRIVATE threads.cpp threads.h memory.cpp memory.h)

# Files: direct and buffered opens, positional reads and file identity. Windows
# reads a handle's no-buffering mode back through ntdll.
target_sources(ggml-ssd-platform PRIVATE file.cpp file.h)
if (WIN32)
    target_link_libraries(ggml-ssd-platform PRIVATE ntdll)
endif()
