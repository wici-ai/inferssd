# Stateless POSIX reader shared by the Mac M1 source and portable fixtures.
# It owns no runtime/cache globals and has no backend or llama dependency.
add_library(ggml-ssd-pread STATIC pread-nocache.cpp pread-nocache.h)
set_target_properties(ggml-ssd-pread PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(ggml-ssd-pread PUBLIC cxx_std_17)
target_include_directories(ggml-ssd-pread PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

# The CPU direct-syscall ring retains its independent scheduling and queue.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(ggml-ssd-linux-ring STATIC linux-ring.cpp linux-ring.h)
    target_compile_features(ggml-ssd-linux-ring PUBLIC cxx_std_17)
    target_include_directories(ggml-ssd-linux-ring PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set_target_properties(ggml-ssd-linux-ring PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ggml-ssd INTERFACE ggml-ssd-linux-ring)
endif()

include("${CMAKE_CURRENT_SOURCE_DIR}/object-uring.cmake")
