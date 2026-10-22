set(TARGET llama-kimi-verify-bench)
add_executable(${TARGET} kimi-verify-bench.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(ROLLBACK_TARGET llama-kimi-rollback-check)
add_executable(${ROLLBACK_TARGET} kimi-rollback-check.cpp)
target_link_libraries(${ROLLBACK_TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${ROLLBACK_TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} ${ROLLBACK_TARGET} RUNTIME)
endif()
