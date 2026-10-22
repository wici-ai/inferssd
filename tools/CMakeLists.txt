# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

if (EMSCRIPTEN)
else()
    add_subdirectory(batched-bench)
    add_subdirectory(gguf-split)
    add_subdirectory(imatrix)
    add_subdirectory(kimi-verify-bench)
    add_subdirectory(kimi-pipelined-decode-spike)
    add_subdirectory(llama-bench)
    add_subdirectory(completion)
    add_subdirectory(perplexity)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND GGML_CUDA)
        add_subdirectory(q4-colibri-direct-reader)
    endif()
    add_subdirectory(quantize)
    if (LLAMA_BUILD_SERVER)
        add_subdirectory(cli)
        add_subdirectory(server)
    endif()
    add_subdirectory(tokenize)
    add_subdirectory(parser)
    add_subdirectory(tts)
    add_subdirectory(mtmd)
    if (GGML_RPC)
        add_subdirectory(rpc)
    endif()
    if (NOT GGML_BACKEND_DL)
        # these examples use the backends directly and cannot be built with dynamic loading
        add_subdirectory(cvector-generator)
        add_subdirectory(export-lora)
    endif()
    add_subdirectory(fit-params)
    add_subdirectory(results)
endif()
