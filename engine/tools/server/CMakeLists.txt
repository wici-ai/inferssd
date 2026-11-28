include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by inferssd-server and CLI

set(TARGET server-context)

add_library(${TARGET} STATIC
    server-chat.cpp
    server-chat.h
    server-task.cpp
    server-task.h
    inferssd/prompt-cache.cpp
    inferssd/prompt-cache.h
    inferssd/app-versions.cpp
    inferssd/app-versions.h
    inferssd/code-mode.cpp
    inferssd/code-mode.h
    inferssd/checkpoint.h
    inferssd/decode-probe.cpp
    inferssd/decode-probe.h
    inferssd/identifiers.cpp
    inferssd/identifiers.h
    inferssd/runtime.h
    inferssd/usage.cpp
    inferssd/usage.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-ssd-reservation.h
    server-tools.cpp
    server-tools.h
    ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps/sha256/sha256.c
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/examples/gguf-hash/deps)
target_link_libraries(${TARGET} PUBLIC llama-common mtmd ggml-ssd-platform ${CMAKE_THREAD_LIBS_INIT})
if (GGML_CUDA)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_SERVER_CUDA_MOE)
endif()


# inferssd-server executable

set(TARGET inferssd-server)

set(TARGET_SRCS
    server.cpp
    server-http.cpp
    server-http.h
    server-models.cpp
    server-models.h
)

option(LLAMA_BUILD_WEBUI "Build the embedded Web UI" ON)

if (LLAMA_BUILD_WEBUI)
    set(PUBLIC_ASSETS
        index.html
        bundle.js
        bundle.css
        loading.html
    )

    foreach(asset ${PUBLIC_ASSETS})
        set(input "${CMAKE_CURRENT_SOURCE_DIR}/public/${asset}")
        set(output "${CMAKE_CURRENT_BINARY_DIR}/${asset}.hpp")
        list(APPEND TARGET_SRCS ${output})
        add_custom_command(
            DEPENDS "${input}"
            OUTPUT "${output}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${input}" "-DOUTPUT=${output}" -P "${PROJECT_SOURCE_DIR}/scripts/xxd.cmake"
        )
        set_source_files_properties(${output} PROPERTIES GENERATED TRUE)
    endforeach()
    add_definitions(-DLLAMA_BUILD_WEBUI)
else()
endif()

add_executable(${TARGET} ${TARGET_SRCS})
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC llama-common cpp-httplib ${CMAKE_THREAD_LIBS_INIT})

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# The server reads a model in a folder whose name is not ASCII (the model's facts, which need no backend). On Windows
# its command line comes in the process code page, which the application manifest (cmake/win-app.cmake) makes
# UTF-8, the encoding the model reader takes: the models are under the user's profile folder, whose name can be in
# any language.
if (LLAMA_BUILD_TESTS AND TARGET inferssd-ci-tests)
    set(unicode_model_dir "${CMAKE_CURRENT_BINARY_DIR}/modèle Ω")
    file(COPY "${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf" DESTINATION "${unicode_model_dir}")
    add_test(NAME server_unicode_model_path
             COMMAND ${TARGET} --ssd-model-facts "${unicode_model_dir}/ggml-vocab-qwen2.gguf")
    set_tests_properties(server_unicode_model_path PROPERTIES PASS_REGULAR_EXPRESSION "\"architecture\":\"qwen2\"")
    inferssd_ci_test(${TARGET} server_unicode_model_path)
endif()
