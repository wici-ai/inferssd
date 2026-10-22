# Private prototype: one DSO, never a static archive copied into two consumers.
find_package(Threads REQUIRED)
add_library(ggml-ssd-c0-store SHARED store.cpp parser.cpp parser.h store.h catalog.h legacy-index.h legacy-config.h)
target_compile_features(ggml-ssd-c0-store PRIVATE cxx_std_17)
target_link_libraries(ggml-ssd-c0-store PRIVATE Threads::Threads)
set_target_properties(ggml-ssd-c0-store PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
target_include_directories(ggml-ssd-c0-store PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
if (GGML_CUDA)
    target_compile_definitions(ggml-ssd-c0-store PRIVATE GGML_C0_REQUIRE_CUDA_PROVIDER)
endif()
install(TARGETS ggml-ssd-c0-store LIBRARY)
