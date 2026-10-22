# Private prototype: one DSO, never a static archive copied into two consumers.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(C0_EXPERIMENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../experiments/c0-store")
file(GLOB C0_ORACLE_FILES CONFIGURE_DEPENDS "${C0_EXPERIMENT_DIR}/oracle/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${C0_EXPERIMENT_DIR}/extract.py" ${C0_ORACLE_FILES})
execute_process(COMMAND "${Python3_EXECUTABLE}" "${C0_EXPERIMENT_DIR}/extract.py"
    "${CMAKE_CURRENT_BINARY_DIR}/generated" COMMAND_ERROR_IS_FATAL ANY)
add_library(ggml-ssd-c0-store SHARED store.cpp store.h catalog.h legacy-index.h legacy-config.h)
target_compile_features(ggml-ssd-c0-store PRIVATE cxx_std_17)
set_target_properties(ggml-ssd-c0-store PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
target_include_directories(ggml-ssd-c0-store PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated")
if (GGML_CUDA)
    target_compile_definitions(ggml-ssd-c0-store PRIVATE GGML_C0_REQUIRE_CUDA_PROVIDER)
endif()
install(TARGETS ggml-ssd-c0-store LIBRARY)
