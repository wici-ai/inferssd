find_package(Threads REQUIRED)

# The tier is part of the backend built from ggml-cuda's sources: ggml-cuda, or ggml-hip, which compiles the same
# sources as HIP through vendors/hip.h. The backend's CMakeLists sets SSD_GPU_BACKEND (its target), SSD_GPU_LANGUAGE
# (CUDA or HIP) and SSD_GPU_RUNTIME (the runtime's target) before it adds this directory.
# The sources that include the GPU runtime (gpu-runtime.h) compile in the backend's language.
set(SSD_GPU_SOURCES vram-tier.cpp transport.cpp generic-compute-adapter.cu group-compute-adapter.cu)
add_library(ssd-vram-tier OBJECT ${SSD_GPU_SOURCES} provider.cpp)
target_sources(${SSD_GPU_BACKEND} PRIVATE group-matmul.cu)
if (SSD_GPU_LANGUAGE STREQUAL "HIP")
    set_source_files_properties(${SSD_GPU_SOURCES} PROPERTIES LANGUAGE HIP)
    set_source_files_properties(group-matmul.cu TARGET_DIRECTORY ${SSD_GPU_BACKEND} PROPERTIES LANGUAGE HIP)
endif()
# The host side keeps exact floating point whatever the language compiles it (HIP contracts by default).
set_source_files_properties(vram-tier.cpp transport.cpp provider.cpp PROPERTIES
    COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
set_target_properties(ssd-vram-tier PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_include_directories(ssd-vram-tier PRIVATE ../../../include ../..)
target_compile_features(ssd-vram-tier PRIVATE cxx_std_17)
target_link_libraries(ssd-vram-tier PRIVATE ggml-base ggml-ssd ${SSD_GPU_RUNTIME} Threads::Threads)
target_link_libraries(${SSD_GPU_BACKEND} PRIVATE ssd-vram-tier ggml-ssd ${SSD_GPU_RUNTIME})
target_compile_definitions(${SSD_GPU_BACKEND} PRIVATE GGML_SSD_CUDA_EXECUTOR)

# What the tier's tests (engine/tests) build against: the backend, its runtime, and its language (SSD_GPU_LANGUAGE), in
# which a test source that includes the GPU runtime compiles, for the backend's GPU architectures
# (SSD_GPU_ARCHITECTURES under HIP).
add_library(ssd-vram-tier-gpu INTERFACE)
target_link_libraries(ssd-vram-tier-gpu INTERFACE ${SSD_GPU_BACKEND} ${SSD_GPU_RUNTIME})
set_property(TARGET ssd-vram-tier-gpu PROPERTY SSD_GPU_LANGUAGE ${SSD_GPU_LANGUAGE})
if (SSD_GPU_LANGUAGE STREQUAL "HIP")
    target_compile_definitions(ssd-vram-tier-gpu INTERFACE GGML_USE_HIP)
    get_target_property(ssd_gpu_architectures ${SSD_GPU_BACKEND} HIP_ARCHITECTURES)
    set_property(TARGET ssd-vram-tier-gpu PROPERTY SSD_GPU_ARCHITECTURES "${ssd_gpu_architectures}")
endif()
