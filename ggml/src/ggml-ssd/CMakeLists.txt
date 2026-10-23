# SSDash core modules depend only on ggml-base and platform facilities.
add_library(ggml-ssd INTERFACE)
target_link_libraries(ggml-ssd INTERFACE ggml-base)
target_include_directories(ggml-ssd INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

if (LLAMA_SSD_STORE)
    add_subdirectory(store)
endif()
if (UNIX)
    add_subdirectory(io)
endif()
