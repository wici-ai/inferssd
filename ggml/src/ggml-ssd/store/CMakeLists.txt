# Private feasibility target. The default build never enters this directory.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(C0_EXTRACT_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../experiments/c0-store/extract.py")
set(C0_LEGACY_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../../ggml-cuda/moe_stream_batch_common.cu")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${C0_EXTRACT_SCRIPT}" "${C0_LEGACY_SOURCE}")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${C0_EXTRACT_SCRIPT}" "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND_ERROR_IS_FATAL ANY)
add_library(ggml-ssd-c0-store STATIC store.cpp store.h legacy-index.h legacy-config.h)
target_compile_features(ggml-ssd-c0-store PRIVATE cxx_std_17)
set_target_properties(ggml-ssd-c0-store PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(ggml-ssd-c0-store PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}"
    PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
