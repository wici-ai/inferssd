llama_add_compile_flags()
find_package(Threads REQUIRED)

function(llama_build source)
    set(TEST_SOURCES ${source} ${ARGN})

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_link_libraries(${TEST_TARGET} PRIVATE llama llama-common)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS} get-model.cpp)

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE llama-common)

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Hosted CI (.github/workflows/ci.yml) builds inferssd-ci-tests and runs `ctest -L ci` on Linux, macOS and Windows.
# inferssd_ci_test(<executable target> <test>...) adds tests to it on every OS that registers them; call it next to
# their add_test. Tests that need a GPU, a model, the network or hardware a hosted runner lacks stay out.
add_custom_target(inferssd-ci-tests)
function(inferssd_ci_test target)
    add_dependencies(inferssd-ci-tests ${target})
    set_property(TEST ${ARGN} APPEND PROPERTY LABELS ci)
endfunction()

add_executable(test-q2_0 test-q2_0.cpp)
target_include_directories(test-q2_0 PRIVATE ../ggml/src)
target_link_libraries(test-q2_0 PRIVATE ggml-cpu ggml-base)
add_test(NAME q2_0 COMMAND test-q2_0)
set_property(TEST q2_0 PROPERTY LABELS unit)
inferssd_ci_test(test-q2_0 q2_0)

# The queries answered before the parser loads backends (common_params_answer_query): the real parser, compiled here with
# its backend loader turned into a tripwire, so the test needs no GPU. The parser's --license reads LICENSES, which
# llama-common does not export from its DLL on Windows, so the test compiles its own copy.
add_executable(test-ssd-backend-free-query ssd/test-backend-free-query.cpp ../common/arg.cpp ${CMAKE_BINARY_DIR}/license.cpp)
target_link_libraries(test-ssd-backend-free-query PRIVATE llama-common)
if (MSVC)
    target_compile_options(test-ssd-backend-free-query PRIVATE
        /FI${CMAKE_CURRENT_SOURCE_DIR}/ssd/backend-free-query-loader.h)
else()
    target_compile_options(test-ssd-backend-free-query PRIVATE
        -include ${CMAKE_CURRENT_SOURCE_DIR}/ssd/backend-free-query-loader.h)
endif()
set(query_absent_model ${CMAKE_CURRENT_BINARY_DIR}/absent.gguf)
add_test(NAME ssd_query_schema COMMAND test-ssd-backend-free-query --ssd-schema)
add_test(NAME ssd_query_schema_underscore COMMAND test-ssd-backend-free-query --ssd_schema)
add_test(NAME ssd_query_model_facts COMMAND test-ssd-backend-free-query --ssd-model-facts ${query_absent_model})
add_test(NAME ssd_query_version COMMAND test-ssd-backend-free-query --version)
set_tests_properties(ssd_query_schema ssd_query_schema_underscore PROPERTIES PASS_REGULAR_EXPRESSION "InferSSD settings v1")
set_tests_properties(ssd_query_model_facts PROPERTIES PASS_REGULAR_EXPRESSION "absent[.]gguf: ")
set_tests_properties(ssd_query_version PROPERTIES PASS_REGULAR_EXPRESSION "built with ")
set_tests_properties(ssd_query_schema ssd_query_schema_underscore ssd_query_model_facts ssd_query_version PROPERTIES
    FAIL_REGULAR_EXPRESSION "attempted to load;unexpected backend")
# with other arguments, the same options reach the parser, which loads the backends first
add_test(NAME ssd_query_option_value COMMAND test-ssd-backend-free-query --expect-backend --model --ssd-schema)
add_test(NAME ssd_query_mixed_args COMMAND test-ssd-backend-free-query --expect-backend --ssd-schema --help)
add_test(NAME ssd_query_facts_mixed COMMAND test-ssd-backend-free-query --expect-backend --ssd-model-facts absent.gguf -v)
inferssd_ci_test(test-ssd-backend-free-query
    ssd_query_schema ssd_query_schema_underscore ssd_query_model_facts ssd_query_version
    ssd_query_option_value ssd_query_mixed_args ssd_query_facts_mixed)

add_executable(test-moe-common-dynamic-tier ssd/test-moe-common-dynamic-tier.cpp)
target_link_libraries(test-moe-common-dynamic-tier PRIVATE Threads::Threads)
add_test(NAME moe_common_dynamic_tier COMMAND test-moe-common-dynamic-tier)
set_property(TEST moe_common_dynamic_tier PROPERTY LABELS main)
inferssd_ci_test(test-moe-common-dynamic-tier moe_common_dynamic_tier)


add_executable(test-ssd-model-adapter ssd/test-model-adapter.cpp ../src/ssd/model-adapter.cpp)
target_compile_features(test-ssd-model-adapter PRIVATE cxx_std_17)
target_include_directories(test-ssd-model-adapter PRIVATE ../src ../include)
target_link_libraries(test-ssd-model-adapter PRIVATE ggml-base)
add_test(NAME ssd_model_adapter COMMAND test-ssd-model-adapter)
set_property(TEST ssd_model_adapter PROPERTY LABELS main)
inferssd_ci_test(test-ssd-model-adapter ssd_model_adapter)

add_executable(test-sched-inplace ssd/test-sched-inplace.cpp)
target_compile_features(test-sched-inplace PRIVATE cxx_std_17)
target_include_directories(test-sched-inplace PRIVATE ../ggml/src)
target_link_libraries(test-sched-inplace PRIVATE ggml)
add_test(NAME sched_inplace COMMAND test-sched-inplace)
set_property(TEST sched_inplace PROPERTY LABELS main)
inferssd_ci_test(test-sched-inplace sched_inplace)

add_executable(test-sched-unplaceable ssd/test-sched-unplaceable.cpp)
target_compile_features(test-sched-unplaceable PRIVATE cxx_std_17)
target_include_directories(test-sched-unplaceable PRIVATE ../ggml/src)
target_link_libraries(test-sched-unplaceable PRIVATE ggml)
add_test(NAME sched_unplaceable COMMAND test-sched-unplaceable)
set_property(TEST sched_unplaceable PROPERTY LABELS main)
inferssd_ci_test(test-sched-unplaceable sched_unplaceable)

add_executable(test-sched-cache-placement ssd/test-sched-cache-placement.cpp)
target_compile_features(test-sched-cache-placement PRIVATE cxx_std_17)
target_include_directories(test-sched-cache-placement PRIVATE ../ggml/src)
target_link_libraries(test-sched-cache-placement PRIVATE ggml)
add_test(NAME sched_cache_placement COMMAND test-sched-cache-placement)
set_property(TEST sched_cache_placement PROPERTY LABELS main)
inferssd_ci_test(test-sched-cache-placement sched_cache_placement)

add_executable(test-node-name ssd/test-node-name.cpp)
target_compile_features(test-node-name PRIVATE cxx_std_17)
target_include_directories(test-node-name PRIVATE ../src)
target_link_libraries(test-node-name PRIVATE ggml)
add_test(NAME node_name COMMAND test-node-name)
set_property(TEST node_name PROPERTY LABELS main)
inferssd_ci_test(test-node-name node_name)

add_executable(test-gallocr-shared-size ssd/test-gallocr-shared-size.cpp)
target_compile_features(test-gallocr-shared-size PRIVATE cxx_std_17)
target_include_directories(test-gallocr-shared-size PRIVATE ../ggml/src)
target_link_libraries(test-gallocr-shared-size PRIVATE ggml)
add_test(NAME gallocr_shared_size COMMAND test-gallocr-shared-size)
set_property(TEST gallocr_shared_size PROPERTY LABELS main)
inferssd_ci_test(test-gallocr-shared-size gallocr_shared_size)

add_executable(test-ssd-runtime-policy ssd/test-ssd-runtime-policy.cpp)
target_compile_features(test-ssd-runtime-policy PRIVATE cxx_std_17)
target_compile_options(test-ssd-runtime-policy PRIVATE -ffp-contract=off)
target_include_directories(test-ssd-runtime-policy PRIVATE ../ggml/src)
add_test(NAME ssd_runtime_policy COMMAND test-ssd-runtime-policy)
set_property(TEST ssd_runtime_policy PROPERTY LABELS main)
inferssd_ci_test(test-ssd-runtime-policy ssd_runtime_policy)

add_executable(test-ssd-prefill-cost ssd/test-ssd-prefill-cost.cpp)
target_compile_features(test-ssd-prefill-cost PRIVATE cxx_std_17)
target_include_directories(test-ssd-prefill-cost PRIVATE ../ggml/src/ggml-cpu)
add_test(NAME ssd_prefill_cost COMMAND test-ssd-prefill-cost)
set_property(TEST ssd_prefill_cost PROPERTY LABELS main)
inferssd_ci_test(test-ssd-prefill-cost ssd_prefill_cost)

add_executable(test-ssd-host-gather ssd/test-ssd-host-gather.cpp)
target_compile_features(test-ssd-host-gather PRIVATE cxx_std_17)
target_include_directories(test-ssd-host-gather PRIVATE ../src ../ggml/src)
target_link_libraries(test-ssd-host-gather PRIVATE ggml-base ggml-ssd-platform)
add_test(NAME ssd_host_gather COMMAND test-ssd-host-gather)
set_property(TEST ssd_host_gather PROPERTY LABELS main)
inferssd_ci_test(test-ssd-host-gather ssd_host_gather)

add_executable(test-ssd-host-lock ssd/test-ssd-host-lock.cpp)
target_compile_features(test-ssd-host-lock PRIVATE cxx_std_17)
target_include_directories(test-ssd-host-lock PRIVATE ../ggml/src/ggml-cpu)
target_link_libraries(test-ssd-host-lock PRIVATE Threads::Threads)
add_test(NAME ssd_host_lock COMMAND test-ssd-host-lock)
set_property(TEST ssd_host_lock PROPERTY LABELS main)
inferssd_ci_test(test-ssd-host-lock ssd_host_lock)

add_executable(test-graph-lane ssd/test-graph-lane.cpp)
target_compile_features(test-graph-lane PRIVATE cxx_std_17)
target_include_directories(test-graph-lane PRIVATE ../ggml/src)
target_link_libraries(test-graph-lane PRIVATE ggml ggml-ssd)
add_test(NAME graph_lane COMMAND test-graph-lane)
# Hosted Linux and Windows runners have no GPU for the lane; the test passes there without running it.
if (APPLE)
    inferssd_ci_test(test-graph-lane graph_lane)
endif()

add_executable(test-ssd-gpu-contract ssd/test-ssd-gpu-contract.cpp)
target_compile_features(test-ssd-gpu-contract PRIVATE cxx_std_17)
target_include_directories(test-ssd-gpu-contract PRIVATE ../ggml/src ../ggml/include)
add_test(NAME ssd_gpu_contract COMMAND test-ssd-gpu-contract)
inferssd_ci_test(test-ssd-gpu-contract ssd_gpu_contract)

add_executable(test-ssd-cuda-reserve ssd/test-ssd-cuda-reserve.cpp)
target_compile_features(test-ssd-cuda-reserve PRIVATE cxx_std_17)
target_include_directories(test-ssd-cuda-reserve PRIVATE ../ggml/src)
target_link_libraries(test-ssd-cuda-reserve PRIVATE Threads::Threads)
add_test(NAME ssd_cuda_reserve COMMAND test-ssd-cuda-reserve)
inferssd_ci_test(test-ssd-cuda-reserve ssd_cuda_reserve)

if (UNIX)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # The arbiter over the Linux ledger, with cgroup and /proc files injected.
        add_executable(test-ssd-host-arbiter ssd/test-ssd-host-arbiter.cpp)
        target_compile_features(test-ssd-host-arbiter PRIVATE cxx_std_17)
        target_link_libraries(test-ssd-host-arbiter PRIVATE ggml-ssd-host Threads::Threads)
        add_test(NAME ssd_host_arbiter COMMAND test-ssd-host-arbiter)
        set_tests_properties(ssd_host_arbiter PROPERTIES LABELS main TIMEOUT 30)
        inferssd_ci_test(test-ssd-host-arbiter ssd_host_arbiter)
    endif()
    add_executable(test-ssd-host-prefault ssd/test-ssd-host-prefault.cpp)
    target_compile_features(test-ssd-host-prefault PRIVATE cxx_std_17)
    target_include_directories(test-ssd-host-prefault PRIVATE ../ggml/src)
    target_link_libraries(test-ssd-host-prefault PRIVATE ggml-base)
    add_test(NAME ssd_host_prefault COMMAND test-ssd-host-prefault)
    set_tests_properties(ssd_host_prefault PROPERTIES LABELS main TIMEOUT 30)
    inferssd_ci_test(test-ssd-host-prefault ssd_host_prefault)
endif()

# The arbiter over the native budget sources (macOS, Windows) on this machine.
if (APPLE OR WIN32)
    add_executable(test-ssd-host-budget ssd/test-ssd-host-budget.cpp)
    target_include_directories(test-ssd-host-budget PRIVATE ../ggml/src)
    target_link_libraries(test-ssd-host-budget PRIVATE ggml-ssd-host)
    add_test(NAME ssd_host_budget COMMAND test-ssd-host-budget)
    set_tests_properties(ssd_host_budget PROPERTIES LABELS main TIMEOUT 30)
    inferssd_ci_test(test-ssd-host-budget ssd_host_budget)
endif()

# The platform layer runs on every OS: thread placement and arena memory.
add_executable(test-ssd-platform-threads ssd/test-ssd-platform-threads.cpp)
target_include_directories(test-ssd-platform-threads PRIVATE ../ggml/src)
target_link_libraries(test-ssd-platform-threads PRIVATE ggml-ssd-platform)
add_test(NAME ssd_platform_threads COMMAND test-ssd-platform-threads)
set_tests_properties(ssd_platform_threads PROPERTIES LABELS main TIMEOUT 60)
inferssd_ci_test(test-ssd-platform-threads ssd_platform_threads)
add_executable(test-ssd-platform-memory ssd/test-ssd-platform-memory.cpp)
target_include_directories(test-ssd-platform-memory PRIVATE ../ggml/src)
target_link_libraries(test-ssd-platform-memory PRIVATE ggml-ssd-platform)
if (WIN32)
    target_link_libraries(test-ssd-platform-memory PRIVATE psapi)
endif()
add_test(NAME ssd_platform_memory COMMAND test-ssd-platform-memory)
set_tests_properties(ssd_platform_memory PROPERTIES LABELS main TIMEOUT 30)
inferssd_ci_test(test-ssd-platform-memory ssd_platform_memory)
# Platform files and every read ring this OS builds (Windows: both backends).
add_executable(test-ssd-platform-file ssd/test-ssd-platform-file.cpp)
target_link_libraries(test-ssd-platform-file PRIVATE ggml-ssd-platform)
add_test(NAME ssd_platform_file COMMAND test-ssd-platform-file)
set_tests_properties(ssd_platform_file PROPERTIES LABELS main TIMEOUT 60)
inferssd_ci_test(test-ssd-platform-file ssd_platform_file)
add_executable(test-ssd-read-ring ssd/test-ssd-read-ring.cpp)
target_link_libraries(test-ssd-read-ring PRIVATE ggml-ssd-pool-ring)
foreach(ring_target ggml-ssd-linux-ring ggml-ssd-win-ring)
    if (TARGET ${ring_target})
        target_link_libraries(test-ssd-read-ring PRIVATE ${ring_target})
    endif()
endforeach()
add_test(NAME ssd_read_ring COMMAND test-ssd-read-ring)
set_tests_properties(ssd_read_ring PROPERTIES LABELS main TIMEOUT 120)
inferssd_ci_test(test-ssd-read-ring ssd_read_ring)

# The expert store's source index on this OS: direct descriptors, identity, lifetime.
if (TARGET ggml-ssd-c0-store)
    add_executable(test-ssd-source-index ssd/test-source-index.cpp)
    target_include_directories(test-ssd-source-index PRIVATE ../ggml/src)
    target_link_libraries(test-ssd-source-index PRIVATE ggml-ssd-c0-store ggml-base)
    add_test(NAME ssd_source_index COMMAND test-ssd-source-index)
    set_tests_properties(ssd_source_index PROPERTIES LABELS main TIMEOUT 30)
    inferssd_ci_test(test-ssd-source-index ssd_source_index)
endif()

# A model's facts from its headers, as the engine would load it (llama_ssd_model_facts).
if (LLAMA_SSD_STORE)
    add_executable(test-ssd-model-facts ssd/test-model-facts.cpp)
    target_compile_features(test-ssd-model-facts PRIVATE cxx_std_17)
    target_include_directories(test-ssd-model-facts PRIVATE ../ggml/src ../vendor)
    target_link_libraries(test-ssd-model-facts PRIVATE llama ggml-base)
    add_test(NAME ssd_model_facts COMMAND test-ssd-model-facts ${CMAKE_CURRENT_SOURCE_DIR}/ssd/fixtures/model-facts.json)
    set_tests_properties(ssd_model_facts PROPERTIES LABELS main TIMEOUT 30)
    inferssd_ci_test(test-ssd-model-facts ssd_model_facts)
endif()

if (TARGET ggml-cpu)
    add_executable(test-ssd-prepared-graph ssd/test-ssd-prepared-graph.cpp)
    target_compile_features(test-ssd-prepared-graph PRIVATE cxx_std_17)
    target_include_directories(test-ssd-prepared-graph PRIVATE ../ggml/src)
    target_link_libraries(test-ssd-prepared-graph PRIVATE ggml-base ggml-cpu)
    add_test(NAME ssd_prepared_graph COMMAND test-ssd-prepared-graph)
    set_tests_properties(ssd_prepared_graph PROPERTIES LABELS main TIMEOUT 30)
    inferssd_ci_test(test-ssd-prepared-graph ssd_prepared_graph)

    # Not on Windows: they build llama's internal classes, which llama.dll does not export.
    if (UNIX)
        add_executable(test-ssd-runtime-options ssd/test-runtime-options.cpp)
        target_include_directories(test-ssd-runtime-options PRIVATE ../src ../ggml/src)
        target_link_libraries(test-ssd-runtime-options PRIVATE llama ggml-ssd-settings)
        add_test(NAME ssd_runtime_options COMMAND test-ssd-runtime-options)
        set_property(TEST ssd_runtime_options PROPERTY LABELS main)
        inferssd_ci_test(test-ssd-runtime-options ssd_runtime_options)

        add_executable(test-ssd-precompute-memory ssd/test-precompute-memory.cpp)
        target_compile_features(test-ssd-precompute-memory PRIVATE cxx_std_17)
        target_compile_options(test-ssd-precompute-memory PRIVATE -UNDEBUG)
        target_include_directories(test-ssd-precompute-memory PRIVATE ../src ../ggml/src)
        target_link_libraries(test-ssd-precompute-memory PRIVATE llama)
        add_test(NAME ssd_precompute_memory COMMAND test-ssd-precompute-memory)
        inferssd_ci_test(test-ssd-precompute-memory ssd_precompute_memory)

        add_executable(test-ssd-kv-spill ssd/test-kv-spill.cpp)
        target_compile_features(test-ssd-kv-spill PRIVATE cxx_std_17)
        target_include_directories(test-ssd-kv-spill PRIVATE ../src ../ggml/src)
        target_link_libraries(test-ssd-kv-spill PRIVATE llama)
        add_test(NAME ssd_kv_spill COMMAND test-ssd-kv-spill)
        inferssd_ci_test(test-ssd-kv-spill ssd_kv_spill)
    endif()
    add_executable(test-ssd-settings-telemetry ssd/test-settings-telemetry.cpp)
    target_compile_options(test-ssd-settings-telemetry PRIVATE -UNDEBUG)
    target_include_directories(test-ssd-settings-telemetry PRIVATE ../ggml/src)
    target_link_libraries(test-ssd-settings-telemetry PRIVATE ggml-ssd-settings)
    add_test(NAME ssd_settings_telemetry COMMAND test-ssd-settings-telemetry)
    set_property(TEST ssd_settings_telemetry PROPERTY LABELS main)
    inferssd_ci_test(test-ssd-settings-telemetry ssd_settings_telemetry)

    add_executable(test-ssd-context-telemetry ssd/test-ssd-context-telemetry.cpp)
    target_compile_options(test-ssd-context-telemetry PRIVATE -UNDEBUG)
    target_include_directories(test-ssd-context-telemetry PRIVATE ../src ../ggml/src)
    target_link_libraries(test-ssd-context-telemetry PRIVATE ggml-ssd-settings Threads::Threads)
    add_test(NAME ssd_context_telemetry COMMAND test-ssd-context-telemetry)
    set_tests_properties(ssd_context_telemetry PROPERTIES LABELS main TIMEOUT 30)
    inferssd_ci_test(test-ssd-context-telemetry ssd_context_telemetry)
    add_executable(test-ssd-cpu-api ssd/test-ssd-cpu-api.cpp)
    target_compile_features(test-ssd-cpu-api PRIVATE cxx_std_17)
    target_include_directories(test-ssd-cpu-api PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
    target_link_libraries(test-ssd-cpu-api PRIVATE ggml-cpu ggml-base)
    add_test(NAME ssd_cpu_api COMMAND test-ssd-cpu-api)
    set_property(TEST ssd_cpu_api PROPERTY LABELS main)
    inferssd_ci_test(test-ssd-cpu-api ssd_cpu_api)
endif()

# The RAM cache's lease metadata against real file identities, on every OS.
if (TARGET ggml-ssd-host)
    add_executable(test-ssd-cache-lease ssd/test-ssd-cache-lease.cpp)
    target_compile_features(test-ssd-cache-lease PRIVATE cxx_std_17)
    target_include_directories(test-ssd-cache-lease PRIVATE ../ggml/src)
    target_link_libraries(test-ssd-cache-lease PRIVATE ggml-ssd-host ggml-base Threads::Threads)
    add_test(NAME test-ssd-cache-lease COMMAND test-ssd-cache-lease)
    inferssd_ci_test(test-ssd-cache-lease test-ssd-cache-lease)
endif()

# Linux only: the fixed-route oracle (GNU C) and the CUDA VRAM tier fixtures.
if (TARGET ggml-cpu AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test-ssd-cpu-fixed-route ssd/test-cpu-fixed-route.cpp ssd/oracle/routed-math.c)
    target_compile_features(test-ssd-cpu-fixed-route PRIVATE cxx_std_17 c_std_11)
    target_compile_definitions(test-ssd-cpu-fixed-route PRIVATE _GNU_SOURCE)
    target_include_directories(test-ssd-cpu-fixed-route PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
    target_link_libraries(test-ssd-cpu-fixed-route PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME ssd_cpu_fixed_route COMMAND test-ssd-cpu-fixed-route)
    inferssd_ci_test(test-ssd-cpu-fixed-route ssd_cpu_fixed_route)
    foreach(lease_target test-ssd-vram-ready test-ssd-vram-native test-ssd-vram-pool)
        add_executable(${lease_target} ssd/${lease_target}.cpp)
        target_compile_features(${lease_target} PRIVATE cxx_std_17)
        target_include_directories(${lease_target} PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../ggml/src/ggml-cuda/ssd)
        target_link_libraries(${lease_target} PRIVATE ggml-cpu ggml-base Threads::Threads ${CMAKE_DL_LIBS})
        add_test(NAME ${lease_target} COMMAND ${lease_target})
        inferssd_ci_test(${lease_target} ${lease_target})
    endforeach()
    if (TARGET ssd-vram-tier)
        # GPU tests require explicit ownership of a lane and normally linked
        # backend objects; do not acquire a shared GPU automatically in ctest.
        # They link the backend the tier is built into (CUDA or HIP) and compile in its language.
        get_target_property(tier_language ssd-vram-tier-gpu SSD_GPU_LANGUAGE)
        set(tier_sources ssd/test-ssd-vram-async.cpp ssd/test-ssd-vram-deferred.cpp ssd/test-ssd-vram-backup.cpp
            ssd/test-ssd-vram-format-compat.cpp ssd/test-ssd-vram-pool-cuda.cpp ssd/test-ssd-vram-elastic.cpp
            ssd/test-ssd-vram-reclaim.cpp ssd/test-ssd-vram-mixed.cpp ssd/test-ssd-vram-payload.cpp
            ssd/test-ssd-prefill-scratch.cpp ssd/test-ssd-vram-generic.cpp ssd/test-ssd-vram-mm.cpp
            ssd/test-ssd-vram-group.cpp ssd/test-ssd-cublas-retry.cpp)
        if (tier_language STREQUAL "HIP")
            # CMake keeps a language's rules per directory; ggml-hip enabled HIP in its own, for its architectures
            get_target_property(CMAKE_HIP_ARCHITECTURES ssd-vram-tier-gpu SSD_GPU_ARCHITECTURES)
            enable_language(HIP)
            set_source_files_properties(${tier_sources} PROPERTIES LANGUAGE HIP)
        endif()
        foreach(tier_test test-ssd-vram-async test-ssd-vram-deferred test-ssd-vram-backup
                test-ssd-vram-format-compat test-ssd-vram-pool-cuda
                test-ssd-vram-elastic test-ssd-vram-reclaim test-ssd-vram-mixed test-ssd-vram-payload)
            add_executable(${tier_test} ssd/${tier_test}.cpp)
            target_compile_features(${tier_test} PRIVATE cxx_std_17)
            target_compile_options(${tier_test} PRIVATE -fno-fast-math -ffp-contract=off)
            target_include_directories(${tier_test} PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../ggml/src/ggml-cuda/ssd ../vendor)
            target_link_libraries(${tier_test} PRIVATE ggml-ssd-host ggml-cpu ggml-base ssd-vram-tier-gpu Threads::Threads ${CMAKE_DL_LIBS})
            add_dependencies(${tier_test} ssd-vram-tier)
        endforeach()
        add_executable(test-ssd-prefill-scratch ssd/test-ssd-prefill-scratch.cpp)
        target_compile_features(test-ssd-prefill-scratch PRIVATE cxx_std_17)
        target_compile_options(test-ssd-prefill-scratch PRIVATE -fno-fast-math -ffp-contract=off)
        target_include_directories(test-ssd-prefill-scratch PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../vendor)
        target_link_libraries(test-ssd-prefill-scratch PRIVATE ggml-ssd-host ggml-cpu ggml-base ssd-vram-tier-gpu Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-ssd-prefill-scratch ssd-vram-tier)
        add_executable(test-ssd-vram-generic ssd/test-ssd-vram-generic.cpp)
        target_compile_features(test-ssd-vram-generic PRIVATE cxx_std_17)
        target_include_directories(test-ssd-vram-generic PRIVATE ../ggml/src ../ggml/src/ggml-cpu
            ../ggml/src/ggml-cuda/ssd ../vendor)
        target_link_libraries(test-ssd-vram-generic PRIVATE ggml-ssd-host ggml-cpu ggml-base ssd-vram-tier-gpu
            Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-ssd-vram-generic ssd-vram-tier)
        # The GPU backend's grouped-expert products against ggml's CPU MUL_MAT (owned GPU; not in ctest).
        add_executable(test-ssd-vram-mm ssd/test-ssd-vram-mm.cpp)
        target_compile_features(test-ssd-vram-mm PRIVATE cxx_std_17)
        target_include_directories(test-ssd-vram-mm PRIVATE ../ggml/src/ggml-cuda/ssd)
        target_link_libraries(test-ssd-vram-mm PRIVATE ggml-cpu ggml-base ssd-vram-tier-gpu)
        # The tier's grouped prefill engine against the CPU lane, and its timing mode (owned GPU; not in ctest).
        add_executable(test-ssd-vram-group ssd/test-ssd-vram-group.cpp)
        target_compile_features(test-ssd-vram-group PRIVATE cxx_std_17)
        target_include_directories(test-ssd-vram-group PRIVATE ../ggml/src ../ggml/src/ggml-cpu
            ../ggml/src/ggml-cuda/ssd ../vendor)
        target_link_libraries(test-ssd-vram-group PRIVATE ggml-ssd-host ggml-cpu ggml-base ssd-vram-tier-gpu Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-ssd-vram-group ssd-vram-tier)
        # Linux shared-library interposition keeps the cuBLAS failure fixture local to this executable.
        # Like the other CUDA fixtures, run explicitly on an owned GPU, not in hosted CPU ctest.
        if (BUILD_SHARED_LIBS)
            add_executable(test-ssd-cublas-retry ssd/test-ssd-cublas-retry.cpp)
            target_compile_features(test-ssd-cublas-retry PRIVATE cxx_std_17)
            target_include_directories(test-ssd-cublas-retry PRIVATE ../ggml/include ../ggml/src)
            # hipBLAS comes with the HIP backend; cuBLAS is linked for its declarations' library
            target_link_libraries(test-ssd-cublas-retry PRIVATE ssd-vram-tier-gpu)
            if (NOT tier_language STREQUAL "HIP")
                target_link_libraries(test-ssd-cublas-retry PRIVATE CUDA::cublas)
                add_executable(test-ssd-cublas-reserve ssd/test-ssd-cublas-reserve.cpp)
                target_compile_features(test-ssd-cublas-reserve PRIVATE cxx_std_17)
                target_include_directories(test-ssd-cublas-reserve PRIVATE ../ggml/src)
                target_link_libraries(test-ssd-cublas-reserve PRIVATE ssd-vram-tier-gpu ggml-base CUDA::cublas ${CMAKE_DL_LIBS})
            endif()
        endif()
        # The error fixture explicitly injects callbacks around the real linked component.
        target_link_libraries(test-ssd-vram-elastic PRIVATE ssd-vram-tier-gpu)
        target_link_libraries(test-ssd-vram-reclaim PRIVATE ssd-vram-tier-gpu)
    endif()
    if (TARGET ssd-vram-tier AND LLAMA_SSD_STORE)
        add_executable(test-ssd-dense-stream ssd/test-dense-stream.cpp)
        target_compile_features(test-ssd-dense-stream PRIVATE cxx_std_17)
        target_include_directories(test-ssd-dense-stream PRIVATE ../src ../include ../ggml/src)
        target_link_libraries(test-ssd-dense-stream PRIVATE llama ssd-vram-tier-gpu ggml-cpu ggml-ssd)
        add_test(NAME ssd_dense_stream COMMAND test-ssd-dense-stream)
        set_tests_properties(ssd_dense_stream PROPERTIES LABELS main TIMEOUT 120)
    endif()
endif()

# The SSD executor on every OS: the CPU sink, grouped prefill, RAM cache and next-layer prefetch,
# bit for bit against ggml CPU reference graphs. Fixture files sit in the OS temporary directory.
# Cases that need a Linux facility (cgroup v2 limits, /proc, fork with RLIMIT_MEMLOCK, pread
# interposition) skip those parts elsewhere; the executables say which.
if (TARGET ggml-cpu)
    add_executable(test-test-resources ssd/test-test-resources.cpp)
    target_link_libraries(test-test-resources PRIVATE ggml-ssd-host)
    add_test(NAME test_resources COMMAND test-test-resources)
    set_tests_properties(test_resources PROPERTIES LABELS unit SKIP_RETURN_CODE 77)
    inferssd_ci_test(test-test-resources test_resources)
    add_executable(test-ssd-cpu-sink ssd/test-ssd-cpu-sink.cpp)
    target_include_directories(test-ssd-cpu-sink PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../vendor)
    target_link_libraries(test-ssd-cpu-sink PRIVATE ggml-ssd-host ggml-ssd-settings ggml-cpu ggml-base Threads::Threads)
    add_test(NAME ssd_cpu_sink COMMAND test-ssd-cpu-sink)
    add_test(NAME dsv41_q2_cpu_sink_decode COMMAND test-ssd-cpu-sink
        --q2-activation-test --threads 9 --ring-experts 8)

    add_test(NAME dsv41_mixed_iq_cpu_sink_decode COMMAND test-ssd-cpu-sink
        --dsv41-iq-test --threads 9 --ring-experts 8)
    set_tests_properties(dsv41_mixed_iq_cpu_sink_decode PROPERTIES LABELS device SKIP_RETURN_CODE 77)

    add_test(NAME ssd_cpu_sink_schedule_invalid COMMAND test-ssd-cpu-sink --schedule-invalid)
    # Sink nodes in ggml graphs on a ggml threadpool of 1, 2, 3 and all CPUs: the production schedule, and a
    # polling one.
    add_test(NAME ssd_cpu_sink_threadpool COMMAND test-ssd-cpu-sink --threadpool-tests --ring-experts 8
        --fixture ready_first=1 --fixture early_up_gate=1 --fixture role_major=1
        --fixture nonblocking_down_sched=1 --fixture dynamic_rows=1 --fixture dynamic_chunk_rows=64)
    add_test(NAME ssd_cpu_sink_threadpool_poll COMMAND test-ssd-cpu-sink --threadpool-tests --ring-experts 1
        --fixture poll_us=100)
    set_tests_properties(ssd_cpu_sink_threadpool ssd_cpu_sink_threadpool_poll PROPERTIES TIMEOUT 180)
    # SwiGLU experts (qwen4exp's geometry; a BF16 shape with a scalar tail) in decode and grouped prefill on a
    # ggml threadpool, against build_moe_ffn's graph with ggml_swiglu_split: the production schedule and the
    # ordinary one.
    add_test(NAME ssd_cpu_sink_swiglu COMMAND test-ssd-cpu-sink --swiglu-tests --ring-experts 8
        --fixture ready_first=1 --fixture early_up_gate=1 --fixture role_major=1
        --fixture nonblocking_down_sched=1 --fixture dynamic_rows=1 --fixture dynamic_chunk_rows=64)
    add_test(NAME ssd_cpu_sink_swiglu_ordinary COMMAND test-ssd-cpu-sink --swiglu-tests --ring-experts 1
        --fixture ready_first=0)
    # Keep the mixed IQ3_XXS/Q2_0 fixture in separate processes so each SwiGLU suite fits the hosted timeout.
    add_test(NAME ssd_cpu_sink_swiglu_q2 COMMAND test-ssd-cpu-sink --swiglu-q2-tests --ring-experts 8
        --fixture ready_first=1 --fixture early_up_gate=1 --fixture role_major=1
        --fixture nonblocking_down_sched=1 --fixture dynamic_rows=1 --fixture dynamic_chunk_rows=64)
    add_test(NAME ssd_cpu_sink_swiglu_q2_ordinary COMMAND test-ssd-cpu-sink --swiglu-q2-tests --ring-experts 1
        --fixture ready_first=0)
    set_tests_properties(ssd_cpu_sink_swiglu ssd_cpu_sink_swiglu_ordinary
        ssd_cpu_sink_swiglu_q2 ssd_cpu_sink_swiglu_q2_ordinary PROPERTIES TIMEOUT 300)
    # The grouped prefill's cross-layer read-ahead: layer 2's experts read into the pool during layer 1's call, used
    # bit for bit as reads; injected read failures; a decode run between layers; destruction while it reads. The
    # production schedule with several participants, and one participant with one expert in flight.
    add_test(NAME ssd_cpu_sink_lookahead COMMAND test-ssd-cpu-sink --lookahead-tests --threads 4 --ring-experts 8
        --fixture ready_first=1 --fixture early_up_gate=1 --fixture role_major=1
        --fixture nonblocking_down_sched=1 --fixture dynamic_rows=1 --fixture dynamic_chunk_rows=64)
    add_test(NAME ssd_cpu_sink_lookahead_d1 COMMAND test-ssd-cpu-sink --lookahead-tests --threads 2 --ring-experts 1
        --fixture ready_first=0)
    set_tests_properties(ssd_cpu_sink_lookahead ssd_cpu_sink_lookahead_d1 PROPERTIES TIMEOUT 180)
    # Muse Spark's experts with their post norm (Q8_0; BF16, also from native payload ranges and the RAM cache) in
    # decode and grouped prefill, against the MoE graph Muse builds, bit for bit.
    add_test(NAME ssd_cpu_sink_muse_q8_0 COMMAND test-ssd-cpu-sink --muse-q8-test --threads 2)
    add_test(NAME ssd_cpu_sink_muse_bf16 COMMAND test-ssd-cpu-sink --muse-bf16-test --threads 2)
    set_tests_properties(ssd_cpu_sink_muse_q8_0 ssd_cpu_sink_muse_bf16 PROPERTIES TIMEOUT 180)
    # A decode microbenchmark, not a CTest: test-ssd-sink-bench --help.
    add_executable(test-ssd-sink-bench ssd/test-ssd-sink-bench.cpp)
    target_include_directories(test-ssd-sink-bench PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
    target_link_libraries(test-ssd-sink-bench PRIVATE ggml-ssd-host ggml-cpu ggml-base Threads::Threads)
    # Trait-selected kernels (MXFP4/Q8_0, Q8_0 down, Q3_K) and the SiTU and
    # SWIGLU_CLAMP activations against a MUL_MAT_ID graph; the RAM cache cases
    # require a finite cgroup memory limit like the other cache tests.
    foreach(activation_threads 2 16)
        foreach(activation_depth 1 8)
            foreach(activation_ready 0 1)
                set(activation_test ssd_cpu_sink_activation_t${activation_threads}_d${activation_depth}_ready${activation_ready})
                add_test(NAME ${activation_test} COMMAND test-ssd-cpu-sink --activation-tests
                    --threads ${activation_threads} --ring-experts ${activation_depth}
                    --fixture ready_first=${activation_ready} --fixture read_fixed=${activation_ready} --fixture profile=0)
                set_tests_properties(${activation_test} PROPERTIES TIMEOUT 180)
            endforeach()
        endforeach()
    endforeach()
    foreach(early_threads 2 16)
        foreach(early_depth 1 8)
            foreach(early_ready 0 1)
                set(early_test ssd_cpu_sink_early_t${early_threads}_d${early_depth}_ready${early_ready})
                add_test(NAME ${early_test} COMMAND test-ssd-cpu-sink --early-up-gate-tests
                    --threads ${early_threads} --ring-experts ${early_depth}
                    --fixture cache_policy=lru --fixture cache_allocation=uniform --fixture ready_first=${early_ready} --fixture read_fixed=${early_ready} --fixture profile=0)
                set_tests_properties(${early_test} PROPERTIES TIMEOUT 180)
                set(poll_test ssd_cpu_sink_poll_t${early_threads}_d${early_depth}_ready${early_ready})
                add_test(NAME ${poll_test} COMMAND test-ssd-cpu-sink --poll-tests
                    --threads ${early_threads} --ring-experts ${early_depth}
                    --fixture cache_policy=lru --fixture cache_allocation=uniform --fixture ready_first=${early_ready} --fixture read_fixed=${early_ready} --fixture profile=0)
                set_tests_properties(${poll_test} PROPERTIES TIMEOUT 180)
            endforeach()
        endforeach()
    endforeach()
    foreach(role_threads 2 16)
        foreach(role_depth 1 8)
            foreach(role_ready 0 1)
                set(role_test ssd_cpu_sink_role_major_t${role_threads}_d${role_depth}_ready${role_ready})
                add_test(NAME ${role_test} COMMAND test-ssd-cpu-sink --role-major-tests
                    --threads ${role_threads} --ring-experts ${role_depth}
                    --fixture cache_policy=lru --fixture cache_allocation=uniform --fixture ready_first=${role_ready} --fixture read_fixed=${role_ready} --fixture profile=0)
                set_tests_properties(${role_test} PROPERTIES TIMEOUT 180)
            endforeach()
        endforeach()
    endforeach()
    foreach(nonblocking_threads 2 16)
        foreach(nonblocking_depth 1 8)
            foreach(nonblocking_fixed 0 1)
                set(nonblocking_test ssd_cpu_sink_nonblocking_down_t${nonblocking_threads}_d${nonblocking_depth}_fixed${nonblocking_fixed})
                add_test(NAME ${nonblocking_test} COMMAND test-ssd-cpu-sink --nonblocking-down-tests
                    --threads ${nonblocking_threads} --ring-experts ${nonblocking_depth}
                    --fixture cache_policy=lru --fixture cache_allocation=uniform --fixture ready_first=1 --fixture read_fixed=${nonblocking_fixed} --fixture profile=0)
                set_tests_properties(${nonblocking_test} PROPERTIES TIMEOUT 180)
            endforeach()
        endforeach()
    endforeach()
    add_test(NAME ssd_cpu_sink_early_cache COMMAND test-ssd-cpu-sink --cache-tests --threads 16 --ring-experts 8
                    --fixture early_up_gate=1 --fixture ready_first=1 --fixture read_fixed=1)
    set_tests_properties(ssd_cpu_sink_early_cache PROPERTIES TIMEOUT 180)
    add_test(NAME ssd_cpu_sink_poll_early COMMAND test-ssd-cpu-sink --poll-early-tests --threads 16 --ring-experts 8
                    --fixture cache_policy=lru --fixture cache_allocation=uniform --fixture ready_first=1 --fixture read_fixed=0 --fixture profile=0)
    set_tests_properties(ssd_cpu_sink_poll_early PROPERTIES TIMEOUT 180)
    foreach(schedule_threads 2 12 16)
        foreach(schedule_depth 1 8)
            set(schedule_test ssd_cpu_sink_ordinary_schedule_t${schedule_threads}_d${schedule_depth})
            add_test(NAME ${schedule_test} COMMAND test-ssd-cpu-sink --schedule-tests
                --threads ${schedule_threads} --ring-experts ${schedule_depth}
                    --fixture cache_policy=lru --fixture cache_allocation=uniform --fixture read_fixed=0 --fixture profile=0)
            set_tests_properties(${schedule_test} PROPERTIES TIMEOUT 180)
        endforeach()
    endforeach()
    add_test(NAME ssd_cpu_sink_trace_profile COMMAND test-ssd-cpu-sink --trace-profile --threads 4 --ring-experts 1
                    --fixture read_fixed=0)
    set_tests_properties(ssd_cpu_sink_trace_profile PROPERTIES TIMEOUT 180)
    # the RAM cache cases require a finite cgroup memory limit like the other cache tests
    add_test(NAME ssd_cpu_sink_prefetch COMMAND test-ssd-cpu-sink --prefetch-tests --threads 4 --ring-experts 2
                    --fixture read_fixed=0 --fixture profile=0)
    set_tests_properties(ssd_cpu_sink_prefetch PROPERTIES TIMEOUT 180)
    # the RAM cache's storage at load (a finite cgroup memory limit, like the other cache cases)
    add_test(NAME ssd_cpu_sink_cache_load COMMAND test-ssd-cpu-sink --cache-load-tests --threads 4 --ring-experts 2
                    --fixture read_fixed=0 --fixture profile=0)
    set_tests_properties(ssd_cpu_sink_cache_load PROPERTIES TIMEOUT 180)
    # an allocation outside the caches, the CPU backend's work buffer, takes room from a RAM cache that took the
    # envelope at load (a finite memory envelope, like the other cache cases)
    add_test(NAME ssd_cpu_sink_work_reclaim COMMAND test-ssd-cpu-sink --work-reclaim-tests --threads 2 --ring-experts 2
                    --fixture read_fixed=0 --fixture profile=0)
    set_tests_properties(ssd_cpu_sink_work_reclaim PROPERTIES TIMEOUT 180)
    # the prefetch against the GPU tier's residency, with a stand-in tier the executable exports (no GPU)
    add_executable(test-ssd-prefetch-tier ssd/test-ssd-prefetch-tier.cpp)
    target_include_directories(test-ssd-prefetch-tier PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../vendor)
    target_link_libraries(test-ssd-prefetch-tier PRIVATE ggml-ssd-host ggml-ssd-settings ggml-cpu ggml-base Threads::Threads)
    add_test(NAME ssd_prefetch_tier COMMAND test-ssd-prefetch-tier)
    set_tests_properties(ssd_prefetch_tier PROPERTIES TIMEOUT 180)
    add_test(NAME ssd_cpu_sink_registered COMMAND test-ssd-cpu-sink --registered-lifecycle
                    --fixture read_fixed=1)

    # A single compute participant and ring=1, and several participants with ring wraparound.
    foreach(cache_threads 2 4)
        math(EXPR cache_depth "${cache_threads} / 2")
        foreach(cache_fixed RANGE 0 1)
            set(cache_test ssd_cpu_sink_cache_t${cache_threads}_fixed${cache_fixed})
            add_test(NAME ${cache_test} COMMAND test-ssd-cpu-sink --cache-tests
                --threads ${cache_threads} --ring-experts ${cache_depth}
                    --fixture read_fixed=${cache_fixed} --fixture profile=0)
            set_tests_properties(${cache_test} PROPERTIES TIMEOUT 180)
        endforeach()
    endforeach()
    foreach(cache_threads 4 16)
        foreach(cache_depth 1 8)
            set(max_cache_test ssd_cpu_sink_max_cache_t${cache_threads}_d${cache_depth})
            add_test(NAME ${max_cache_test} COMMAND test-ssd-cpu-sink --max-cache-tests
                --threads ${cache_threads} --ring-experts ${cache_depth}
                    --fixture cache_allocation=uniform --fixture ready_first=1 --fixture read_fixed=0 --fixture profile=0)
            set_tests_properties(${max_cache_test} PROPERTIES TIMEOUT 180)
        endforeach()
    endforeach()
    foreach(cache_policy lru lfu decay count_decay2)
        foreach(cache_allocation priority uniform global)
            set(large_test ssd_cpu_sink_large_${cache_policy}_${cache_allocation})
            add_test(NAME ${large_test} COMMAND test-ssd-cpu-sink --cache-large-tests --threads 4 --ring-experts 1
                    --fixture cache_policy=${cache_policy} --fixture cache_allocation=${cache_allocation} --fixture read_fixed=1 --fixture profile=0)
            set_tests_properties(${large_test} PROPERTIES TIMEOUT 180)
            foreach(cache_transport RANGE 0 1)
                set(policy_test ssd_cpu_sink_policy_${cache_policy}_${cache_allocation}_fixed${cache_transport})
                add_test(NAME ${policy_test} COMMAND test-ssd-cpu-sink --cache-policy-tests --threads 4 --ring-experts 2
                    --fixture cache_policy=${cache_policy} --fixture cache_allocation=${cache_allocation} --fixture read_fixed=${cache_transport} --fixture profile=0)
                set_tests_properties(${policy_test} PROPERTIES TIMEOUT 180)
            endforeach()
        endforeach()
    endforeach()
    # Not on Windows: it builds llama's internal graph classes, which llama.dll does not export.
    if (NOT WIN32)
        add_executable(test-ssd-cpu-prefill-graph ssd/test-ssd-cpu-prefill-graph.cpp)
        target_compile_features(test-ssd-cpu-prefill-graph PRIVATE cxx_std_17)
        target_include_directories(test-ssd-cpu-prefill-graph PRIVATE ../src ../ggml/src ../ggml/src/ggml-cpu)
        target_link_libraries(test-ssd-cpu-prefill-graph PRIVATE llama ggml-cpu)
        add_test(NAME ssd_cpu_prefill_graph COMMAND test-ssd-cpu-prefill-graph)
    endif()
    add_executable(test-ssd-cpu-prefill ssd/test-ssd-cpu-prefill.cpp)
    target_include_directories(test-ssd-cpu-prefill PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
    target_link_libraries(test-ssd-cpu-prefill PRIVATE ggml-ssd-host ggml-cpu ggml-base Threads::Threads)
    add_test(NAME ssd_cpu_prefill COMMAND test-ssd-cpu-prefill)
    # A manual x86 kernel benchmark; not on Windows, where ggml-cpu.dll does not export the kernels it calls.
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT WIN32)
        add_executable(test-ssd-mxfp4-gemm ssd/test-ssd-mxfp4-gemm.cpp)
        target_include_directories(test-ssd-mxfp4-gemm PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
        target_link_libraries(test-ssd-mxfp4-gemm PRIVATE ggml-cpu ggml-base)
    endif()
    foreach(retain_depth 1 8)
        add_test(NAME ssd_cpu_prefill_retain_d${retain_depth} COMMAND test-ssd-cpu-prefill
            --cache-retention --threads 4 --ring-experts ${retain_depth}
                    --fixture read_fixed=0 --fixture ready_first=1 --fixture cache_allocation=uniform)
        set_tests_properties(ssd_cpu_prefill_retain_d${retain_depth} PROPERTIES TIMEOUT 180)
    endforeach()
    add_test(NAME ssd_cpu_prefill_registered COMMAND test-ssd-cpu-prefill --threads 4 --ring-experts 2
                    --fixture read_fixed=1)

    foreach(lifecycle_threads 2 4)
        math(EXPR lifecycle_depth "${lifecycle_threads} / 2")
        foreach(lifecycle_fixed RANGE 0 1)
            set(lifecycle_test ssd_cpu_prefill_cache_t${lifecycle_threads}_fixed${lifecycle_fixed})
            add_test(NAME ${lifecycle_test} COMMAND test-ssd-cpu-prefill --cache-lifecycle
                --threads ${lifecycle_threads} --ring-experts ${lifecycle_depth}
                    --fixture read_fixed=${lifecycle_fixed} --fixture profile=0)
            set_tests_properties(${lifecycle_test} PROPERTIES TIMEOUT 180)
        endforeach()
    endforeach()
endif()

add_executable(test-moe-common-cross-role ssd/test-moe-common-cross-role.cpp)
add_test(NAME moe_common_cross_role COMMAND test-moe-common-cross-role)
set_property(TEST moe_common_cross_role PROPERTY LABELS main)
inferssd_ci_test(test-moe-common-cross-role moe_common_cross_role)

# Device allocations admitted against the reserve and the free memory; on unified memory it measures the Linux host
# budget (budget-linux.h, /proc/sys/vm, sysconf).
if (TARGET ssd-vram-tier-gpu OR GGML_CUDA)
    add_executable(test-cuda-memory-reserve ssd/test-cuda-memory-reserve.cpp)
    if (TARGET ssd-vram-tier-gpu)
        # the backend the tier is built into, CUDA or HIP, in its language
        get_target_property(reserve_language ssd-vram-tier-gpu SSD_GPU_LANGUAGE)
        if (reserve_language STREQUAL "HIP")
            get_target_property(CMAKE_HIP_ARCHITECTURES ssd-vram-tier-gpu SSD_GPU_ARCHITECTURES)
            enable_language(HIP)
            set_source_files_properties(ssd/test-cuda-memory-reserve.cpp PROPERTIES LANGUAGE HIP)
        endif()
        target_link_libraries(test-cuda-memory-reserve PRIVATE ggml ssd-vram-tier-gpu)
    else()
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(test-cuda-memory-reserve PRIVATE ggml ggml-cuda CUDA::cudart)
    endif()
    target_include_directories(test-cuda-memory-reserve PRIVATE ../ggml/src)
    add_test(NAME cuda_memory_reserve COMMAND test-cuda-memory-reserve)
    set_tests_properties(cuda_memory_reserve PROPERTIES LABELS "gpu" RUN_SERIAL TRUE SKIP_RETURN_CODE 77)
endif()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge          ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm      ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4           ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt               ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2             ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact            ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder         ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-sampling.cpp)
    llama_build_and_test(test-reasoning-budget.cpp)
    inferssd_ci_test(test-reasoning-budget test-reasoning-budget)
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build(ssd/test-server-tooluse-delivery.cpp)
    target_include_directories(test-server-tooluse-delivery PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-tooluse-delivery PRIVATE server-context)
    llama_build(ssd/test-server-output-parser-delivery.cpp)
    target_include_directories(test-server-output-parser-delivery PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-output-parser-delivery PRIVATE server-context)
    llama_build(ssd/test-server-chat-output-source.cpp)
    target_include_directories(test-server-chat-output-source PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-chat-output-source PRIVATE server-context)
    llama_build(ssd/test-server-chat-literal.cpp)
    target_include_directories(test-server-chat-literal PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-chat-literal PRIVATE server-context)
    llama_build_and_test(ssd/test-server-control-token-delivery.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
    target_include_directories(test-server-control-token-delivery PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-control-token-delivery PRIVATE server-context)
    inferssd_ci_test(test-server-control-token-delivery test-server-control-token-delivery)
    llama_build_and_test(ssd/test-server-prompt-cache.cpp)
    target_include_directories(test-server-prompt-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-prompt-cache PRIVATE server-context)
    inferssd_ci_test(test-server-prompt-cache test-server-prompt-cache)
    llama_build_and_test(ssd/test-server-identifiers.cpp)
    target_include_directories(test-server-identifiers PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-identifiers PRIVATE server-context)
    inferssd_ci_test(test-server-identifiers test-server-identifiers)
    llama_build_and_test(ssd/test-server-prompt-cache-conversation.cpp)
    target_include_directories(test-server-prompt-cache-conversation PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-prompt-cache-conversation PRIVATE server-context)
    inferssd_ci_test(test-server-prompt-cache-conversation test-server-prompt-cache-conversation)
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET ggml-cpu)
        # Requires an explicit tiny GGUF/alias and the admitted 8GiB/noSwap scope;
        # deliberately not an asset-free CTest entry.
        add_executable(test-server-prompt-cache-native ssd/test-server-prompt-cache-native.cpp)
        target_include_directories(test-server-prompt-cache-native PRIVATE ../src ../tools/server ../ggml/src)
        target_link_libraries(test-server-prompt-cache-native PRIVATE server-context ggml-cpu)
        target_compile_features(test-server-prompt-cache-native PRIVATE cxx_std_17)
    endif()
    llama_build_and_test(ssd/test-server-text-reuse.cpp)
    target_include_directories(test-server-text-reuse PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-text-reuse PRIVATE server-context)
    inferssd_ci_test(test-server-text-reuse test-server-text-reuse)
    llama_build_and_test(ssd/test-server-runtime-routing.cpp)
    target_include_directories(test-server-runtime-routing PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-runtime-routing PRIVATE server-context)
    inferssd_ci_test(test-server-runtime-routing test-server-runtime-routing)
    llama_build_and_test(ssd/test-server-reasoning-effort.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-server-reasoning-effort PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-reasoning-effort PRIVATE server-context)
    inferssd_ci_test(test-server-reasoning-effort test-server-reasoning-effort)
    llama_build_and_test(ssd/test-server-turn-memory.cpp)
    target_include_directories(test-server-turn-memory PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-turn-memory PRIVATE server-context)
    inferssd_ci_test(test-server-turn-memory test-server-turn-memory)
    llama_build_and_test(ssd/test-server-usage-events.cpp)
    target_include_directories(test-server-usage-events PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-usage-events PRIVATE server-context)
    inferssd_ci_test(test-server-usage-events test-server-usage-events)
    if (TARGET ggml-cpu AND NOT WIN32)
        add_executable(test-kv-cache-iswa-state-at ssd/test-kv-cache-iswa-state-at.cpp)
        target_include_directories(test-kv-cache-iswa-state-at PRIVATE ../src ../ggml/src)
        target_link_libraries(test-kv-cache-iswa-state-at PRIVATE llama ggml-cpu)
        target_compile_features(test-kv-cache-iswa-state-at PRIVATE cxx_std_17)
        add_test(NAME kv_cache_iswa_state_at COMMAND test-kv-cache-iswa-state-at)
        inferssd_ci_test(test-kv-cache-iswa-state-at kv_cache_iswa_state_at)
    endif()
    llama_build_and_test(ssd/test-kimi-deepseek2-guards.cpp)
    target_include_directories(test-kimi-deepseek2-guards PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    inferssd_ci_test(test-kimi-deepseek2-guards test-kimi-deepseek2-guards)
    # TODO: disabled on loongarch64 because the ggml-ci node lacks Python 3.8
    if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
        llama_build_and_test(test-json-schema-to-grammar.cpp   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        target_include_directories(test-json-schema-to-grammar PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    endif()

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2     ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt       ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact    ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    llama_build_and_test(test-llama-archs.cpp)
    llama_test(test-llama-archs NAME test-kv-cache-grow ARGS --kv-grow -s 42)
    llama_test(test-llama-archs NAME test-dsv41-engram-layouts ARGS --engram-layouts -s 42)
    llama_test(test-llama-archs NAME test-llama-archs-dense-ssd ARGS --dense-ssd -s 42)
    # On the hosted macOS runner's paravirtual GPU it aborts on a MUL_MAT the Metal backend does not encode.
    if (NOT APPLE)
        inferssd_ci_test(test-llama-archs test-kv-cache-grow test-dsv41-engram-layouts test-llama-archs-dense-ssd)
    endif()
    # NextN blocks: every architecture, on the CPU alone
    llama_test(test-llama-archs NAME test-llama-archs-nextn ARGS --nextn -s 42)
    inferssd_ci_test(test-llama-archs test-llama-archs-nextn)
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_build_and_test(ssd/test-chat-literal.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
inferssd_ci_test(test-chat-literal test-chat-literal)
llama_build_and_test(ssd/test-chat-history-prefix.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
inferssd_ci_test(test-chat-history-prefix test-chat-history-prefix)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(
        NAME test-moe-copy-lru-sim
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/integration/test-moe-copy-lru-sim.py ${PROJECT_SOURCE_DIR}/..)
    set_tests_properties(test-moe-copy-lru-sim PROPERTIES LABELS "python;ci")
endif()
llama_build_and_test(test-chat-auto-parser.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp)
llama_build_and_test(test-json-partial.cpp)
llama_build_and_test(test-log.cpp)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-json-serialization.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
)
llama_build_and_test(test-regex-partial.cpp)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "tinyllamas/stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "tinyllamas/stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES
    FIXTURES_REQUIRED test-download-model
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")

llama_build_and_test(test-arg-parser.cpp)
llama_test(test-arg-parser NAME test-ple-args ARGS --ple-only)
inferssd_ci_test(test-arg-parser test-ple-args)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-cpu-softplus.cpp LABEL unit)
llama_build_and_test(ssd/test-moe-router-norm.cpp LABEL unit)
target_include_directories(test-moe-router-norm PRIVATE ../src)
inferssd_ci_test(test-cpu-softplus test-cpu-softplus)
inferssd_ci_test(test-moe-router-norm test-moe-router-norm)
llama_build(ssd/test-model-mlock-buffers.cpp)
target_include_directories(test-model-mlock-buffers PRIVATE ../src ../ggml/src)
target_compile_features(test-model-mlock-buffers PRIVATE cxx_std_17)
foreach(mlock_case single split unlocked no-alloc)
    llama_test(test-model-mlock-buffers NAME model_mlock_${mlock_case}
        ARGS ${mlock_case} "${CMAKE_CURRENT_BINARY_DIR}/test-model-mlock-${mlock_case}.gguf")
endforeach()
inferssd_ci_test(test-model-mlock-buffers
    model_mlock_single model_mlock_split model_mlock_unlocked model_mlock_no-alloc)
llama_build_and_test(ssd/test-dsv41-engram.cpp)
target_include_directories(test-dsv41-engram PRIVATE ../src)
inferssd_ci_test(test-dsv41-engram test-dsv41-engram)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    llama_build(ssd/test-llama-mmap-random.cpp)
    target_include_directories(test-llama-mmap-random PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-llama-mmap-random)
    inferssd_ci_test(test-llama-mmap-random test-llama-mmap-random)
    llama_build(ssd/test-dsv41-candidate.cpp)
    target_include_directories(test-dsv41-candidate PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-dsv41-candidate)
    inferssd_ci_test(test-dsv41-candidate test-dsv41-candidate)
endif()

if (APPLE)
    llama_build(ssd/test-llama-mmap-dontneed.cpp)
    target_include_directories(test-llama-mmap-dontneed PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-llama-mmap-dontneed)
    inferssd_ci_test(test-llama-mmap-dontneed test-llama-mmap-dontneed)
endif()

set(LLAMA_BACKEND_OPS_SMOKE_FILTER
    "DSV4_HC_PRE(n_embd=64,hc_mult=4,n_tokens=1,gated=0,slice=0),\
MUL_MAT(type_a=iq4_xs,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=iq1_s,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=iq1_m,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=q2_K,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT_ID(type_a=iq4_xs,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=1,k=256),\
MUL_MAT_ID(type_a=iq1_s,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=1,k=256),\
MUL_MAT_ID(type_a=iq1_m,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=1,k=256),\
MUL_MAT_ID(type_a=q2_K,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=1,k=256),\
MUL_MAT_ID(type_a=q3_K,type_b=f32,n_mats=4,n_used=2,b=0,m=512,n=1,k=256)")
llama_build(test-backend-ops.cpp get-model.cpp)
llama_test(test-backend-ops ARGS test -o "${LLAMA_BACKEND_OPS_SMOKE_FILTER}")
llama_test(test-backend-ops NAME test-backend-ops-full LABEL backend-full ARGS test)

llama_build_and_test(test-model-load-cancel.cpp LABEL "model")
llama_build_and_test(test-autorelease.cpp       LABEL "model")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model")

# Test for state restore with fragmented KV cache
# Requires a model, uses same args pattern as test-thread-safety
llama_build_and_test(test-state-restore-fragmented.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-state-restore-fragmented PROPERTIES FIXTURES_REQUIRED test-download-model)

if (NOT GGML_BACKEND_DL)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp)
    llama_build_and_test(test-quantize-fns.cpp)
    llama_build_and_test(test-quantize-perf.cpp)
    llama_build(ssd/test-q2-vecdot-bench.cpp)
    target_compile_options(test-q2-vecdot-bench PRIVATE -march=native)
    llama_build_and_test(test-rope.cpp)
endif()

# libmtmd
set(LLAMA_TEST_NAME test-mtmd-c-api)
llama_build_and_test(test-mtmd-c-api.c)
target_link_libraries(${LLAMA_TEST_NAME} PRIVATE mtmd)
unset(LLAMA_TEST_NAME)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data LABEL "model")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp LABEL "model")
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

llama_build_and_test(test-alloc.cpp)
target_include_directories(test-alloc PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-flash-attn-indexed-bias.cpp)
llama_build_and_test(test-inkling-sconv-rollback.cpp)
llama_build_and_test(test-inkling-dspark.cpp)
llama_build_and_test(ssd/test-graph-overhead.cpp)
set_tests_properties(test-graph-overhead PROPERTIES LABELS unit)
inferssd_ci_test(test-graph-overhead test-graph-overhead)

llama_build(export-graph-ops.cpp)
target_include_directories(export-graph-ops PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
if (TARGET gguf-model-data)
    target_link_libraries(export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# unit: model-free tests that require neither a GPU nor cgroup setup.
# Keep this list explicit: other synthetic tests may still require Linux I/O facilities.
foreach(unit_test ssd_prefill_cost test-reasoning-budget
                  test-server-runtime-routing test-server-reasoning-effort test-server-usage-events
                  test-dsv41-engram
                  kv_cache_iswa_state_at)
    if (TEST ${unit_test})
        set_property(TEST ${unit_test} APPEND PROPERTY LABELS unit)
    endif()
endforeach()

# Registered SSD executor fixtures are environment-dependent even without models: device needs a
# direct-I/O fixture file and the OS read ring; the RAM cache cases also need a finite memory
# envelope, labelled cgroup on Linux (a cgroup v2 memory.max) and envelope on macOS and Windows
# (the native ledger's, always present). Keep the portable policy/graph tests separate.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(ssd_envelope_label cgroup)
else()
    set(ssd_envelope_label envelope)
endif()
get_property(ssdash_registered_tests DIRECTORY PROPERTY TESTS)
foreach(resource_test IN LISTS ssdash_registered_tests)
    if (resource_test MATCHES "^(ssd_cpu_sink|ssd_cpu_prefill|ssd_prefetch_tier|dsv41_q2_cpu_sink_decode)")
        # The load-hold and work-buffer cases need about 1.5 GiB of host room that stays put while they run. On the
        # hosted macOS runner that room is the free memory of a 7 GB machine, and they do not always find it.
        if (NOT (APPLE AND resource_test MATCHES "^ssd_cpu_sink_(cache_load|work_reclaim)$"))
            set_property(TEST ${resource_test} APPEND PROPERTY LABELS ci)
        endif()
        if (resource_test MATCHES "^ssd_cpu_prefill_graph$")
            set_property(TEST ${resource_test} APPEND PROPERTY LABELS unit)
        else()
            set_property(TEST ${resource_test} APPEND PROPERTY LABELS device)
            set_property(TEST ${resource_test} PROPERTY SKIP_RETURN_CODE 77)
            if (NOT resource_test MATCHES "^(ssd_cpu_sink|ssd_cpu_sink_schedule_invalid|ssd_cpu_sink_threadpool|ssd_cpu_sink_threadpool_poll|ssd_cpu_sink_swiglu|ssd_cpu_sink_swiglu_ordinary|ssd_cpu_sink_swiglu_q2|ssd_cpu_sink_swiglu_q2_ordinary|ssd_cpu_sink_registered|ssd_cpu_sink_trace_profile|dsv41_q2_cpu_sink_decode|ssd_cpu_prefill|ssd_cpu_prefill_registered)$")
                set_property(TEST ${resource_test} APPEND PROPERTY LABELS ${ssd_envelope_label})
            endif()
        endif()
    endif()
endforeach()
foreach(ssd_executable test-ssd-cpu-sink test-ssd-cpu-prefill test-ssd-cpu-prefill-graph test-ssd-prefetch-tier)
    if (TARGET ${ssd_executable})
        add_dependencies(inferssd-ci-tests ${ssd_executable})
    endif()
endforeach()
foreach(gpu_test dsv41_q2_pack_load ssd_dense_stream)
    if (TEST ${gpu_test})
        set_property(TEST ${gpu_test} APPEND PROPERTY LABELS gpu)
    endif()
endforeach()

# The legacy fixtures keep their Linux scope: they model Linux CUDA provider
# publication with dlopen, O_DIRECT and raw descriptors.
if(TARGET ggml-ssd-c0-store AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The fixed extraction reference is test-only and never a production input.
    add_subdirectory(ssd/fixtures/c0-store ${CMAKE_CURRENT_BINARY_DIR}/ssd-store)
endif()
