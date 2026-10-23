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
