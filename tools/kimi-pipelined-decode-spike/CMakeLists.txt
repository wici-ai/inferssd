set(TARGET llama-kimi-pipelined-decode-spike)
add_executable(${TARGET}
    kimi-pipelined-decode-spike.cpp
    ../../examples/gguf-hash/deps/sha256/sha256.c)
target_include_directories(${TARGET} PRIVATE
    ../../examples/gguf-hash/deps
    ../../examples/gguf-hash/deps/sha256)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# This target is a private feasibility probe and is intentionally not installed.
