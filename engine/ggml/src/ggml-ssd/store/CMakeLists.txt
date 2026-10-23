cmake_minimum_required(VERSION 3.19)
# This component can be consumed without configuring llama or an executor.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(inferssd_store LANGUAGES CXX)
    include(CTest)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
# Context-owned immutable source indexes and explicit mature parser.
# No process-global catalog, environment loader or provider registration is linked.
include(GNUInstallDirs)
find_package(Threads REQUIRED)
add_library(ggml-ssd-c0-store SHARED parser.cpp index.cpp parser.h index.h source.h catalog.h legacy-index.h)
target_compile_features(ggml-ssd-c0-store PRIVATE cxx_std_17)
target_link_libraries(ggml-ssd-c0-store PRIVATE Threads::Threads)
set_target_properties(ggml-ssd-c0-store PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
target_include_directories(ggml-ssd-c0-store PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
install(TARGETS ggml-ssd-c0-store LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND BUILD_TESTING)
    get_filename_component(SSD_ENGINE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
    add_executable(test-ssd-store-parser-failures
        "${SSD_ENGINE_ROOT}/tests/ssd/test-store-parser-failures.cpp" parser.cpp)
    target_compile_features(test-ssd-store-parser-failures PRIVATE cxx_std_17)
    target_include_directories(test-ssd-store-parser-failures PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ssd_store_parser_failures COMMAND test-ssd-store-parser-failures)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_executable(test-ssd-source-index "${SSD_ENGINE_ROOT}/tests/ssd/test-source-index.cpp")
        target_compile_features(test-ssd-source-index PRIVATE cxx_std_17)
        target_include_directories(test-ssd-source-index PRIVATE "${SSD_ENGINE_ROOT}/ggml/src")
        target_link_libraries(test-ssd-source-index PRIVATE ggml-ssd-c0-store)
        add_test(NAME ssd_source_index COMMAND test-ssd-source-index)
    endif()
endif()
