# Context-owned immutable source indexes and explicit mature parser.
# No process-global catalog, environment loader or provider registration is linked.
if (NOT GGML_CPU)
    message(FATAL_ERROR "LLAMA_SSD_STORE requires GGML_CPU=ON for the CPU sink executor")
endif()
if (GGML_BACKEND_DL OR GGML_CPU_ALL_VARIANTS)
    message(FATAL_ERROR "LLAMA_SSD_STORE requires GGML_BACKEND_DL=OFF and GGML_CPU_ALL_VARIANTS=OFF; the currently supported store build profile uses direct backend linkage")
endif()
include(GNUInstallDirs)
find_package(Threads REQUIRED)
add_library(ggml-ssd-c0-store SHARED parser.cpp index.cpp parser.h index.h source.h catalog.h legacy-index.h)
target_compile_features(ggml-ssd-c0-store PRIVATE cxx_std_17)
target_link_libraries(ggml-ssd-c0-store PRIVATE Threads::Threads)
set_target_properties(ggml-ssd-c0-store PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
target_include_directories(ggml-ssd-c0-store PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS ggml-ssd-c0-store LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
