# These probe programs are independently addressable resources. Building them
# does not run measurements or start the framework bootstrap process.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(inferssd-ioreplay ioreplay.c)
    target_link_libraries(inferssd-ioreplay PRIVATE m)
    install(TARGETS inferssd-ioreplay RUNTIME)
    if(GGML_CUDA)
        add_executable(inferssd-memprobe memprobe.cu)
        target_link_libraries(inferssd-memprobe PRIVATE CUDA::cudart)
        install(TARGETS inferssd-memprobe RUNTIME)
    endif()
endif()

# Random-read bandwidth and latency through the engine's own read ring, on
# every OS: io_uring on Linux, IoRing or overlapped reads on Windows, the
# reader pool elsewhere (io/read-ring.h).
add_executable(inferssd-ssdprobe ssdprobe.cpp)
if(TARGET ggml-ssd-linux-ring)
    target_link_libraries(inferssd-ssdprobe PRIVATE ggml-ssd-linux-ring)
elseif(TARGET ggml-ssd-win-ring)
    # shell32: CommandLineToArgvW, for UTF-8 paths on the command line.
    target_link_libraries(inferssd-ssdprobe PRIVATE ggml-ssd-win-ring shell32)
else()
    target_link_libraries(inferssd-ssdprobe PRIVATE ggml-ssd-pool-ring)
endif()
target_link_libraries(inferssd-ssdprobe PRIVATE ggml-ssd-platform)
install(TARGETS inferssd-ssdprobe RUNTIME)

# CPU throughput per weight format and DRAM read bandwidth. It calls ggml's CPU backend directly, so it is not built
# when backends are loaded dynamically.
if(NOT GGML_BACKEND_DL)
    add_executable(inferssd-cpuprobe cpuprobe.cpp)
    target_link_libraries(inferssd-cpuprobe PRIVATE ggml Threads::Threads)
    target_compile_features(inferssd-cpuprobe PRIVATE cxx_std_17)
    install(TARGETS inferssd-cpuprobe RUNTIME)
endif()
