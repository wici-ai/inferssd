find_package(Threads REQUIRED)

# The standalone nvcc builds below compile for one architecture. Arch-specific "a" targets
# (120a for the RTX 50 series, 121a for GB10) do not run on any other GPU, so follow the first entry
# of CMAKE_CUDA_ARCHITECTURES unless PARITY_CUDA_ARCH is set in the environment.
set(KIMI_PARITY_ARCH_ENV "")
if (DEFINED ENV{PARITY_CUDA_ARCH})
    set(KIMI_PARITY_ARCH_ENV "PARITY_CUDA_ARCH=$ENV{PARITY_CUDA_ARCH}")
elseif (CMAKE_CUDA_ARCHITECTURES)
    list(GET CMAKE_CUDA_ARCHITECTURES 0 KIMI_PARITY_ARCH)
    string(REGEX REPLACE "-(real|virtual)$" "" KIMI_PARITY_ARCH "${KIMI_PARITY_ARCH}")
    if (KIMI_PARITY_ARCH MATCHES "^[0-9]+[af]?$")
        set(KIMI_PARITY_ARCH_ENV "PARITY_CUDA_ARCH=sm_${KIMI_PARITY_ARCH}")
    endif()
endif()

# MXFP4 x Q8_0 component for DeepSeek-V4.1 experts: a separate DSO with the same
# strict rounding flags, separate from the backend.
get_filename_component(KIMI_PARITY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(KIMI_MXFP4_PARITY_OUTPUT "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/libkimi-mxfp4-parity.so")
add_custom_command(OUTPUT "${KIMI_MXFP4_PARITY_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E env "PARITY_NVCC=${CMAKE_CUDA_COMPILER}" ${KIMI_PARITY_ARCH_ENV}
        bash "${CMAKE_CURRENT_SOURCE_DIR}/build_mxfp4_component.sh"
        "${KIMI_PARITY_SOURCE_DIR}" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
    DEPENDS kimi-mxfp4-parity.cu kimi-mxfp4-chain.cu kimi-mxfp4-batch.cu kimi-mxfp4-batch.h ggml-kimi-mxfp4-parity.h ggml-kimi-mxfp4-chain.h ../../ggml-common.h build_mxfp4_component.sh
    VERBATIM)
add_custom_target(kimi-mxfp4-parity DEPENDS "${KIMI_MXFP4_PARITY_OUTPUT}")

add_library(kimi-vram-tier SHARED vram-tier.cpp mxfp4-compute-adapter.cpp q2-compute-adapter.cu
    group-compute-adapter.cu)
target_include_directories(kimi-vram-tier PRIVATE ../../../include ../.. ../../ggml-cpu)
target_compile_features(kimi-vram-tier PRIVATE cxx_std_17)
set_target_properties(kimi-vram-tier PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_compile_options(kimi-vram-tier PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
target_link_libraries(kimi-vram-tier PRIVATE ggml-cpu ggml-base CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
set_target_properties(kimi-vram-tier PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
