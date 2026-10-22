llama_add_compile_flags()

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

add_executable(test-q4-colibri-worker test-q4-colibri-worker.cpp)
find_package(Threads REQUIRED)
target_link_libraries(test-q4-colibri-worker PRIVATE Threads::Threads)
add_test(NAME q4_colibri_worker COMMAND test-q4-colibri-worker)
add_test(NAME q4_colibri_generation COMMAND test-q4-colibri-worker)
set_property(TEST q4_colibri_worker q4_colibri_generation PROPERTY LABELS main)

add_executable(test-moe-common-dynamic-tier test-moe-common-dynamic-tier.cpp)
target_link_libraries(test-moe-common-dynamic-tier PRIVATE Threads::Threads)
add_test(NAME moe_common_dynamic_tier COMMAND test-moe-common-dynamic-tier)
set_property(TEST moe_common_dynamic_tier PROPERTY LABELS main)

add_executable(test-moe-common-io-submit test-moe-common-io-submit.cpp)
add_test(NAME moe_common_io_submit COMMAND test-moe-common-io-submit)

add_executable(test-moe-pack-copy-plan test-moe-pack-copy-plan.cpp)
add_test(NAME moe_pack_copy_plan COMMAND test-moe-pack-copy-plan)
set_property(TEST moe_pack_copy_plan PROPERTY LABELS main)

add_executable(test-kimi-prefill-cost test-kimi-prefill-cost.cpp)
target_compile_features(test-kimi-prefill-cost PRIVATE cxx_std_17)
target_include_directories(test-kimi-prefill-cost PRIVATE ../ggml/src/ggml-cpu)
add_test(NAME kimi_prefill_cost COMMAND test-kimi-prefill-cost)
set_property(TEST kimi_prefill_cost PROPERTY LABELS main)

add_executable(test-kimi-host-gather test-kimi-host-gather.cpp)
target_compile_features(test-kimi-host-gather PRIVATE cxx_std_17)
target_include_directories(test-kimi-host-gather PRIVATE ../src)
target_link_libraries(test-kimi-host-gather PRIVATE ggml-base)
add_test(NAME kimi_host_gather COMMAND test-kimi-host-gather)
set_property(TEST kimi_host_gather PROPERTY LABELS main)

add_executable(test-kimi-host-lock test-kimi-host-lock.cpp)
target_compile_features(test-kimi-host-lock PRIVATE cxx_std_17)
target_include_directories(test-kimi-host-lock PRIVATE ../ggml/src/ggml-cpu)
target_link_libraries(test-kimi-host-lock PRIVATE Threads::Threads)
add_test(NAME kimi_host_lock COMMAND test-kimi-host-lock)
set_property(TEST kimi_host_lock PROPERTY LABELS main)

if (TARGET ggml-cpu AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    foreach(lease_target test-kimi-vram-ready test-kimi-cache-lease test-kimi-vram-native test-kimi-vram-pool)
        add_executable(${lease_target} ${lease_target}.cpp)
        target_compile_features(${lease_target} PRIVATE cxx_std_17)
        target_include_directories(${lease_target} PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../ggml/src/ggml-cuda/kimi-vram-tier)
        target_link_libraries(${lease_target} PRIVATE ggml-cpu ggml-base Threads::Threads ${CMAKE_DL_LIBS})
        add_test(NAME ${lease_target} COMMAND ${lease_target})
    endforeach()
    if (TARGET kimi-vram-tier)
        # GPU tests require explicit ownership of a lane and preloaded exact
        # component DSOs; do not acquire a shared GPU automatically in ctest.
        foreach(tier_test test-kimi-vram-async test-kimi-vram-deferred test-kimi-vram-exclusive test-kimi-q3k-parity
                test-kimi-vram-location test-kimi-vram-location-lifecycle test-kimi-vram-component-error test-kimi-vram-pool-cuda
                test-kimi-vram-elastic test-kimi-vram-reclaim)
            add_executable(${tier_test} ${tier_test}.cpp)
            target_compile_features(${tier_test} PRIVATE cxx_std_17)
            target_compile_options(${tier_test} PRIVATE -fno-fast-math -ffp-contract=off)
            target_include_directories(${tier_test} PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../ggml/src/ggml-cuda/kimi-vram-tier ../vendor)
            target_link_libraries(${tier_test} PRIVATE ggml-cpu ggml-base CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
            add_dependencies(${tier_test} kimi-q3k-parity kimi-vram-tier)
        endforeach()
        # DeepSeek-V4.1 MXFP4 tier: component rows and the sink end to end. Like the
        # tests above they need an owned GPU and preloaded DSOs, so they stay out of ctest.
        foreach(mxfp4_test test-kimi-mxfp4-parity test-kimi-mxfp4-chain test-kimi-vram-mxfp4)
            add_executable(${mxfp4_test} ${mxfp4_test}.cpp)
            target_compile_features(${mxfp4_test} PRIVATE cxx_std_17)
            target_compile_options(${mxfp4_test} PRIVATE -fno-fast-math -ffp-contract=off)
            target_include_directories(${mxfp4_test} PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../ggml/src/ggml-cuda/kimi-vram-tier ../vendor)
            target_link_libraries(${mxfp4_test} PRIVATE ggml-cpu ggml-base CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
            add_dependencies(${mxfp4_test} kimi-mxfp4-parity kimi-vram-tier)
        endforeach()
        add_executable(test-kimi-prefill-scratch test-kimi-prefill-scratch.cpp)
        target_compile_features(test-kimi-prefill-scratch PRIVATE cxx_std_17)
        target_compile_definitions(test-kimi-prefill-scratch PRIVATE KIMI_SCRATCH_MXFP4_ONLY)
        target_compile_options(test-kimi-prefill-scratch PRIVATE -fno-fast-math -ffp-contract=off)
        target_include_directories(test-kimi-prefill-scratch PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../vendor)
        target_link_libraries(test-kimi-prefill-scratch PRIVATE ggml-cpu ggml-base CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-kimi-prefill-scratch kimi-mxfp4-parity kimi-vram-tier)
        add_executable(test-kimi-vram-q2 test-kimi-vram-q2.cpp)
        target_compile_features(test-kimi-vram-q2 PRIVATE cxx_std_17)
        target_include_directories(test-kimi-vram-q2 PRIVATE ../ggml/src ../ggml/src/ggml-cpu
            ../ggml/src/ggml-cuda/kimi-vram-tier ../vendor)
        target_link_libraries(test-kimi-vram-q2 PRIVATE ggml-cpu ggml-base ggml-cuda
            CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-kimi-vram-q2 kimi-vram-tier)
        # The CUDA backend's grouped-expert products against ggml's CPU MUL_MAT (owned GPU; not in ctest).
        add_executable(test-kimi-vram-mm test-kimi-vram-mm.cpp)
        target_compile_features(test-kimi-vram-mm PRIVATE cxx_std_17)
        target_link_libraries(test-kimi-vram-mm PRIVATE ggml-cpu ggml-base ggml-cuda CUDA::cudart)
        # The tier's grouped prefill engine against the CPU lane, and its timing mode (owned GPU; not in ctest).
        add_executable(test-kimi-vram-group test-kimi-vram-group.cpp)
        target_compile_features(test-kimi-vram-group PRIVATE cxx_std_17)
        target_include_directories(test-kimi-vram-group PRIVATE ../ggml/src ../ggml/src/ggml-cpu
            ../ggml/src/ggml-cuda/kimi-vram-tier ../vendor)
        target_link_libraries(test-kimi-vram-group PRIVATE ggml-cpu ggml-base CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-kimi-vram-group kimi-vram-tier)
        # Real-pack, RAM-only crossover measurement for grouped DeepSeek-V4.1
        # prefill.  It is a manually-run performance tool (owned GPU, 7 GB/s
        # H2D governor and 16 GiB cgroup required), so it is intentionally not
        # registered with ctest.
        add_executable(test-kimi-prefill-crossover test-kimi-prefill-crossover.cpp)
        target_compile_features(test-kimi-prefill-crossover PRIVATE cxx_std_17)
        target_compile_options(test-kimi-prefill-crossover PRIVATE -fno-fast-math -ffp-contract=off)
        target_include_directories(test-kimi-prefill-crossover PRIVATE ../ggml/src ../ggml/src/ggml-cpu
            ../ggml/src/ggml-cuda/kimi-vram-tier ../vendor)
        target_link_libraries(test-kimi-prefill-crossover PRIVATE ggml-cpu ggml-base ggml-cuda
            CUDA::cudart Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(test-kimi-prefill-crossover kimi-mxfp4-parity kimi-vram-tier)
        # The error fixture interposes the component API around the real DSO.
        set_target_properties(test-kimi-vram-component-error PROPERTIES ENABLE_EXPORTS ON)
        target_link_libraries(test-kimi-vram-elastic PRIVATE ggml-cuda)
        target_link_libraries(test-kimi-vram-reclaim PRIVATE ggml-cuda)
    endif()
    add_executable(test-linux-test-resources test-linux-test-resources.cpp)
    target_include_directories(test-linux-test-resources PRIVATE ../ggml/src/ggml-cpu)
    add_test(NAME linux_test_resources COMMAND test-linux-test-resources)
    set_tests_properties(linux_test_resources PROPERTIES LABELS unit)
    add_executable(test-kimi-cpu-sink test-kimi-cpu-sink.cpp)
    target_include_directories(test-kimi-cpu-sink PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../vendor)
    target_link_libraries(test-kimi-cpu-sink PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME kimi_cpu_sink COMMAND test-kimi-cpu-sink)
    add_test(NAME kimi_cpu_sink_expert_first_call COMMAND test-kimi-cpu-sink --expert-job-first-call)
    set_tests_properties(kimi_cpu_sink_expert_first_call PROPERTIES LABELS device TIMEOUT 60 SKIP_RETURN_CODE 77)
    add_test(NAME dsv41_q2_cpu_sink_decode COMMAND test-kimi-cpu-sink
        --q2-activation-test --threads 9 --ring-experts 8)
    add_executable(test-kimi-cpu-cache-policy test-kimi-cpu-cache-policy.cpp)
    target_include_directories(test-kimi-cpu-cache-policy PRIVATE ../ggml/src/ggml-cpu)
    target_link_libraries(test-kimi-cpu-cache-policy PRIVATE ggml-cpu ggml-base)
    add_test(NAME kimi_cpu_sink_cache_compute_policy COMMAND test-kimi-cpu-cache-policy)
    if (TARGET ggml-cuda)
        add_executable(test-dsv41-q2-pack test-dsv41-q2-pack.cpp)
        target_compile_features(test-dsv41-q2-pack PRIVATE cxx_std_17)
        target_link_libraries(test-dsv41-q2-pack PRIVATE ggml-cuda)
        add_test(NAME dsv41_q2_pack_load COMMAND test-dsv41-q2-pack)
        # the weights homed on the drive (DENSE_HOMES): the scheduler's rule and the transport's dense stage
        add_executable(test-kimi-dense-stream test-kimi-dense-stream.cpp)
        target_compile_features(test-kimi-dense-stream PRIVATE cxx_std_17)
        target_link_libraries(test-kimi-dense-stream PRIVATE ggml-cuda ggml-cpu ggml-base)
        add_test(NAME kimi_dense_stream COMMAND test-kimi-dense-stream)
    endif()
    foreach(location_depth 1 8)
        add_test(NAME kimi_cpu_sink_cache_location_d${location_depth}
            COMMAND test-kimi-cpu-sink --cache-location-tests --threads 16 --ring-experts ${location_depth})
    endforeach()
    foreach(comp_threads 2 16)
        foreach(comp_depth 1 8)
            add_test(NAME kimi_cpu_sink_cache_compute_t${comp_threads}_d${comp_depth}
                COMMAND test-kimi-cpu-sink --cache-compute-tests --threads ${comp_threads} --ring-experts ${comp_depth})
        endforeach()
    endforeach()

    add_test(NAME kimi_cpu_sink_schedule_invalid COMMAND test-kimi-cpu-sink --schedule-invalid)
    # Trait-selected kernels (MXFP4/Q8_0, Q8_0 down, Q3_K) and the SiTU and
    # SWIGLU_CLAMP activations against a MUL_MAT_ID graph; the RAM cache cases
    # require a finite cgroup memory limit like the other cache tests.
    foreach(activation_threads 2 16)
        foreach(activation_depth 1 8)
            foreach(activation_ready 0 1)
                set(activation_test kimi_cpu_sink_activation_t${activation_threads}_d${activation_depth}_ready${activation_ready})
                add_test(NAME ${activation_test} COMMAND test-kimi-cpu-sink --activation-tests
                    --threads ${activation_threads} --ring-experts ${activation_depth})
                set_tests_properties(${activation_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "GGML_KIMI_CPU_SINK_READY_FIRST=${activation_ready};GGML_KIMI_CPU_SINK_READ_FIXED=${activation_ready};GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
            endforeach()
        endforeach()
    endforeach()
    foreach(early_threads 2 16)
        foreach(early_depth 1 8)
            foreach(early_ready 0 1)
                set(early_test kimi_cpu_sink_early_t${early_threads}_d${early_depth}_ready${early_ready})
                add_test(NAME ${early_test} COMMAND test-kimi-cpu-sink --early-up-gate-tests
                    --threads ${early_threads} --ring-experts ${early_depth})
                set_tests_properties(${early_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READY_FIRST=${early_ready};GGML_KIMI_CPU_SINK_READ_FIXED=${early_ready};GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
                set(poll_test kimi_cpu_sink_poll_t${early_threads}_d${early_depth}_ready${early_ready})
                add_test(NAME ${poll_test} COMMAND test-kimi-cpu-sink --poll-tests
                    --threads ${early_threads} --ring-experts ${early_depth})
                set_tests_properties(${poll_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READY_FIRST=${early_ready};GGML_KIMI_CPU_SINK_READ_FIXED=${early_ready};GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
            endforeach()
        endforeach()
    endforeach()
    foreach(role_threads 2 16)
        foreach(role_depth 1 8)
            foreach(role_ready 0 1)
                set(role_test kimi_cpu_sink_role_major_t${role_threads}_d${role_depth}_ready${role_ready})
                add_test(NAME ${role_test} COMMAND test-kimi-cpu-sink --role-major-tests
                    --threads ${role_threads} --ring-experts ${role_depth})
                set_tests_properties(${role_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READY_FIRST=${role_ready};GGML_KIMI_CPU_SINK_READ_FIXED=${role_ready};GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
            endforeach()
        endforeach()
    endforeach()
    foreach(nonblocking_threads 2 16)
        foreach(nonblocking_depth 1 8)
            foreach(nonblocking_fixed 0 1)
                set(nonblocking_test kimi_cpu_sink_nonblocking_down_t${nonblocking_threads}_d${nonblocking_depth}_fixed${nonblocking_fixed})
                add_test(NAME ${nonblocking_test} COMMAND test-kimi-cpu-sink --nonblocking-down-tests
                    --threads ${nonblocking_threads} --ring-experts ${nonblocking_depth})
                set_tests_properties(${nonblocking_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READY_FIRST=1;GGML_KIMI_CPU_SINK_READ_FIXED=${nonblocking_fixed};GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
            endforeach()
        endforeach()
    endforeach()
    add_test(NAME kimi_cpu_sink_early_cache COMMAND test-kimi-cpu-sink --cache-tests --threads 16 --ring-experts 8)
    set_tests_properties(kimi_cpu_sink_early_cache PROPERTIES TIMEOUT 180
        ENVIRONMENT "GGML_KIMI_CPU_SINK_EARLY_UP_GATE=1;GGML_KIMI_CPU_SINK_EXPERT_JOB=0;GGML_KIMI_CPU_SINK_READY_FIRST=1;GGML_KIMI_CPU_SINK_READ_FIXED=1;GGML_KIMI_CPU_SINK_SPLIT_CV=1")
    add_test(NAME kimi_cpu_sink_poll_early COMMAND test-kimi-cpu-sink --poll-early-tests --threads 16 --ring-experts 8)
    set_tests_properties(kimi_cpu_sink_poll_early PROPERTIES TIMEOUT 180
        ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READY_FIRST=1;GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
    foreach(schedule_threads 2 12 16)
        foreach(schedule_depth 1 8)
            set(schedule_test kimi_cpu_sink_schedule_t${schedule_threads}_d${schedule_depth})
            add_test(NAME ${schedule_test} COMMAND test-kimi-cpu-sink --schedule-tests
                --threads ${schedule_threads} --ring-experts ${schedule_depth})
            set_tests_properties(${schedule_test} PROPERTIES DISABLED TRUE LABELS "known-failure" TIMEOUT 180
                ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_PROFILE=0")
        endforeach()
    endforeach()
    foreach(job_threads 2 12 16)
        foreach(job_depth 1 8)
            set(job_test kimi_cpu_sink_expert_job_t${job_threads}_d${job_depth})
            add_test(NAME ${job_test} COMMAND test-kimi-cpu-sink --expert-job-tests
                --threads ${job_threads} --ring-experts ${job_depth})
            set_tests_properties(${job_test} PROPERTIES DISABLED TRUE LABELS "known-failure" TIMEOUT 180
                ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=lru;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_SPLIT_CV=0;GGML_KIMI_CPU_SINK_PROFILE=0")
        endforeach()
    endforeach()
    add_test(NAME kimi_cpu_sink_trace_profile COMMAND test-kimi-cpu-sink --trace-profile --threads 4 --ring-experts 1)
    set_tests_properties(kimi_cpu_sink_trace_profile PROPERTIES TIMEOUT 180
        ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_SPLIT_CV=1")
    # the RAM cache cases require a finite cgroup memory limit like the other cache tests
    add_test(NAME kimi_cpu_sink_prefetch COMMAND test-kimi-cpu-sink --prefetch-tests --threads 4 --ring-experts 2)
    set_tests_properties(kimi_cpu_sink_prefetch PROPERTIES TIMEOUT 180
        ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
    # the RAM cache's storage at load (a finite cgroup memory limit, like the other cache cases)
    add_test(NAME kimi_cpu_sink_cache_load COMMAND test-kimi-cpu-sink --cache-load-tests --threads 4 --ring-experts 2)
    set_tests_properties(kimi_cpu_sink_cache_load PROPERTIES TIMEOUT 180
        ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
    # the prefetch against the GPU tier's residency, with a stand-in tier the executable exports (no GPU)
    add_executable(test-kimi-prefetch-tier test-kimi-prefetch-tier.cpp)
    target_include_directories(test-kimi-prefetch-tier PRIVATE ../ggml/src ../ggml/src/ggml-cpu ../vendor)
    target_link_libraries(test-kimi-prefetch-tier PRIVATE ggml-cpu ggml-base Threads::Threads)
    set_target_properties(test-kimi-prefetch-tier PROPERTIES ENABLE_EXPORTS ON)
    add_test(NAME kimi_prefetch_tier COMMAND test-kimi-prefetch-tier)
    set_tests_properties(kimi_prefetch_tier PROPERTIES TIMEOUT 180 ENVIRONMENT "GGML_KIMI_CPU_SINK_SPLIT_CV=1")
    add_test(NAME kimi_cpu_sink_registered COMMAND test-kimi-cpu-sink --registered-lifecycle)
    set_tests_properties(kimi_cpu_sink_registered PROPERTIES
        ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=1;GGML_KIMI_CPU_SINK_SPLIT_CV=1")
    foreach(cache_cv RANGE 0 1)
        foreach(cache_fixed RANGE 0 1)
            # Cover a single compute participant and ring=1 in addition to
            # persistent multiworker scheduling and ring wraparound.
            if (cache_cv EQUAL cache_fixed)
                set(cache_threads 2)
                set(cache_depth 1)
            else()
                set(cache_threads 4)
                set(cache_depth 2)
            endif()
            set(cache_test kimi_cpu_sink_cache_cv${cache_cv}_fixed${cache_fixed})
            add_test(NAME ${cache_test} COMMAND test-kimi-cpu-sink --cache-tests
                --threads ${cache_threads} --ring-experts ${cache_depth})
            set_tests_properties(${cache_test} PROPERTIES TIMEOUT 180
                ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=${cache_fixed};GGML_KIMI_CPU_SINK_SPLIT_CV=${cache_cv};GGML_KIMI_CPU_SINK_PROFILE=0")
        endforeach()
    endforeach()
    foreach(cache_threads 4 16)
        foreach(cache_depth 1 8)
            set(max_cache_test kimi_cpu_sink_max_cache_t${cache_threads}_d${cache_depth})
            add_test(NAME ${max_cache_test} COMMAND test-kimi-cpu-sink --max-cache-tests
                --threads ${cache_threads} --ring-experts ${cache_depth})
            set_tests_properties(${max_cache_test} PROPERTIES TIMEOUT 180
                ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform;GGML_KIMI_CPU_SINK_READY_FIRST=1;GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_PROFILE=0")
        endforeach()
    endforeach()
    foreach(cache_policy lru lfu decay count_decay2)
        foreach(cache_allocation priority uniform global)
            set(large_test kimi_cpu_sink_large_${cache_policy}_${cache_allocation})
            add_test(NAME ${large_test} COMMAND test-kimi-cpu-sink --cache-large-tests --threads 4 --ring-experts 1)
            set_tests_properties(${large_test} PROPERTIES TIMEOUT 180
                ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=${cache_policy};GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=${cache_allocation};GGML_KIMI_CPU_SINK_READ_FIXED=1;GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_PROFILE=0")
            foreach(cache_transport RANGE 0 1)
                set(policy_test kimi_cpu_sink_policy_${cache_policy}_${cache_allocation}_fixed${cache_transport})
                add_test(NAME ${policy_test} COMMAND test-kimi-cpu-sink --cache-policy-tests --threads 4 --ring-experts 2)
                set_tests_properties(${policy_test} PROPERTIES TIMEOUT 180
                    ENVIRONMENT "GGML_KIMI_CPU_SINK_CACHE_POLICY=${cache_policy};GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=${cache_allocation};GGML_KIMI_CPU_SINK_READ_FIXED=${cache_transport};GGML_KIMI_CPU_SINK_SPLIT_CV=${cache_transport};GGML_KIMI_CPU_SINK_PROFILE=0")
            endforeach()
        endforeach()
    endforeach()
    add_executable(test-kimi-cpu-prefill-graph test-kimi-cpu-prefill-graph.cpp)
    target_include_directories(test-kimi-cpu-prefill-graph PRIVATE ../src ../ggml/src ../ggml/src/ggml-cpu)
    target_link_libraries(test-kimi-cpu-prefill-graph PRIVATE llama ggml-cpu)
    add_test(NAME kimi_cpu_prefill_graph COMMAND test-kimi-cpu-prefill-graph)
    add_executable(test-kimi-cpu-prefill test-kimi-cpu-prefill.cpp)
    target_include_directories(test-kimi-cpu-prefill PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
    target_link_libraries(test-kimi-cpu-prefill PRIVATE ggml-cpu ggml-base Threads::Threads)
    add_test(NAME kimi_cpu_prefill COMMAND test-kimi-cpu-prefill)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        add_executable(test-kimi-mxfp4-gemm test-kimi-mxfp4-gemm.cpp)
        target_include_directories(test-kimi-mxfp4-gemm PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
        target_link_libraries(test-kimi-mxfp4-gemm PRIVATE ggml-cpu ggml-base)
    endif()
    foreach(retain_depth 1 8)
        add_test(NAME kimi_cpu_prefill_retain_d${retain_depth} COMMAND test-kimi-cpu-prefill
            --cache-retention --threads 4 --ring-experts ${retain_depth})
        set_tests_properties(kimi_cpu_prefill_retain_d${retain_depth} PROPERTIES TIMEOUT 180
            ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=0;GGML_KIMI_CPU_SINK_SPLIT_CV=1;GGML_KIMI_CPU_SINK_READY_FIRST=1;GGML_KIMI_CPU_SINK_CACHE_ALLOCATION=uniform")
    endforeach()
    add_test(NAME kimi_cpu_prefill_registered COMMAND test-kimi-cpu-prefill --threads 4 --ring-experts 2)
    set_tests_properties(kimi_cpu_prefill_registered PROPERTIES
        ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=1;GGML_KIMI_CPU_SINK_SPLIT_CV=1")
    foreach(lifecycle_cv RANGE 0 1)
        foreach(lifecycle_fixed RANGE 0 1)
            if (lifecycle_cv EQUAL lifecycle_fixed)
                set(lifecycle_threads 2)
                set(lifecycle_depth 1)
            else()
                set(lifecycle_threads 4)
                set(lifecycle_depth 2)
            endif()
            set(lifecycle_test kimi_cpu_prefill_cache_cv${lifecycle_cv}_fixed${lifecycle_fixed})
            add_test(NAME ${lifecycle_test} COMMAND test-kimi-cpu-prefill --cache-lifecycle
                --threads ${lifecycle_threads} --ring-experts ${lifecycle_depth})
            set_tests_properties(${lifecycle_test} PROPERTIES TIMEOUT 180
                ENVIRONMENT "GGML_KIMI_CPU_SINK_READ_FIXED=${lifecycle_fixed};GGML_KIMI_CPU_SINK_SPLIT_CV=${lifecycle_cv};GGML_KIMI_CPU_SINK_PROFILE=0")
        endforeach()
    endforeach()
endif()
set_property(TEST moe_common_io_submit PROPERTY LABELS main)

add_executable(test-moe-common-cross-role test-moe-common-cross-role.cpp)
add_test(NAME moe_common_cross_role COMMAND test-moe-common-cross-role)
set_property(TEST moe_common_cross_role PROPERTY LABELS main)

if (GGML_CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(test-cuda-memory-reserve test-cuda-memory-reserve.cpp)
    target_link_libraries(test-cuda-memory-reserve PRIVATE ggml ggml-cuda CUDA::cudart)
    target_include_directories(test-cuda-memory-reserve PRIVATE ../ggml/src)
    add_test(NAME cuda_memory_reserve COMMAND test-cuda-memory-reserve)
    set_tests_properties(cuda_memory_reserve PROPERTIES LABELS "gpu" RUN_SERIAL TRUE SKIP_RETURN_CODE 77
        ENVIRONMENT "GGML_CUDA_MEMORY_RESERVE_MIB=576")
    add_executable(test-moe-object-engine test-moe-object-engine.cpp)
    target_link_libraries(test-moe-object-engine PRIVATE ggml-cuda CUDA::cudart)
    add_test(NAME moe_object_engine_layout COMMAND test-moe-object-engine)
    add_test(NAME moe_object_engine_single_shape COMMAND test-moe-object-engine single)
    add_test(NAME moe_object_stage_ring COMMAND test-moe-object-engine stage)
    add_test(NAME moe_object_host_slot COMMAND test-moe-object-engine host)
    add_test(NAME moe_role_cache_descriptor COMMAND test-moe-object-engine role)
    add_test(NAME moe_device_descriptor_buffer COMMAND test-moe-object-engine descriptor)
    add_test(NAME moe_dependency_transfer_plan COMMAND test-moe-object-engine transfer)
    add_test(NAME moe_object_pipeline_policy COMMAND test-moe-object-engine pipeline)
    add_test(NAME moe_object_transaction_rendezvous COMMAND test-moe-object-engine transaction)
    add_test(NAME moe_object_pipeline_lifetime COMMAND test-moe-object-engine pipeline-lifetime)
    add_test(NAME moe_kimi_iq1_adapter_dispatch COMMAND test-moe-object-engine kimi-iq1-adapter)
    set_property(TEST moe_object_engine_layout PROPERTY
        ENVIRONMENT "GGML_MOE_OBJECT_ENGINE=glm52-enforce")
    set_property(TEST moe_object_engine_single_shape PROPERTY
        ENVIRONMENT "GGML_MOE_OBJECT_ENGINE=glm52-enforce")
    set_property(TEST moe_object_stage_ring PROPERTY
        ENVIRONMENT "GGML_MOE_OBJECT_ENGINE=glm52-enforce")
    set_property(TEST moe_object_transaction_rendezvous PROPERTY
        ENVIRONMENT "GGML_MOE_OBJECT_ENGINE=glm52-enforce")
    set_property(TEST moe_object_engine_layout moe_object_engine_single_shape moe_object_stage_ring moe_object_host_slot moe_role_cache_descriptor moe_device_descriptor_buffer moe_dependency_transfer_plan moe_object_pipeline_policy moe_object_transaction_rendezvous moe_object_pipeline_lifetime moe_kimi_iq1_adapter_dispatch
        PROPERTY LABELS main)
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
    llama_build_and_test(test-grammar-parser.cpp)
    llama_build_and_test(test-grammar-integration.cpp)
    llama_build_and_test(test-llama-grammar.cpp)
    llama_build_and_test(test-chat.cpp WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build(test-server-tooluse-delivery.cpp)
    target_include_directories(test-server-tooluse-delivery PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-tooluse-delivery PRIVATE server-context)
    llama_build(test-server-output-parser-delivery.cpp)
    target_include_directories(test-server-output-parser-delivery PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-output-parser-delivery PRIVATE server-context)
    llama_build(test-server-chat-output-source.cpp)
    target_include_directories(test-server-chat-output-source PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-chat-output-source PRIVATE server-context)
    llama_build(test-server-chat-literal.cpp)
    target_include_directories(test-server-chat-literal PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-chat-literal PRIVATE server-context)
    llama_build_and_test(test-server-control-token-delivery.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
    target_include_directories(test-server-control-token-delivery PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR}/tools/mtmd)
    target_link_libraries(test-server-control-token-delivery PRIVATE server-context)
    llama_build_and_test(test-server-prompt-cache.cpp)
    target_include_directories(test-server-prompt-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-prompt-cache PRIVATE server-context)
    llama_build_and_test(test-server-text-reuse.cpp)
    target_include_directories(test-server-text-reuse PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-text-reuse PRIVATE server-context)
    add_executable(test-server-expert-guard test-server-expert-guard.cpp)
    target_include_directories(test-server-expert-guard PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    add_test(NAME server_expert_guard COMMAND test-server-expert-guard)
    add_executable(test-server-sampler-struct-top1 test-server-sampler-struct-top1.cpp)
    target_include_directories(test-server-sampler-struct-top1 PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-sampler-struct-top1 PRIVATE llama llama-common)
    target_compile_options(test-server-sampler-struct-top1 PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-UNDEBUG>)
    add_test(NAME server_sampler_struct_top1 COMMAND test-server-sampler-struct-top1)
    llama_build_and_test(test-server-runtime-routing.cpp)
    target_include_directories(test-server-runtime-routing PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-runtime-routing PRIVATE server-context)
    if (TARGET ggml-cpu AND NOT WIN32)
        add_executable(test-kv-cache-iswa-state-at test-kv-cache-iswa-state-at.cpp)
        target_include_directories(test-kv-cache-iswa-state-at PRIVATE ../src ../ggml/src)
        target_link_libraries(test-kv-cache-iswa-state-at PRIVATE llama ggml-cpu)
        target_compile_features(test-kv-cache-iswa-state-at PRIVATE cxx_std_17)
        add_test(NAME kv_cache_iswa_state_at COMMAND test-kv-cache-iswa-state-at)
        add_executable(test-kimi-runtime-routing test-kimi-runtime-routing.cpp)
        target_include_directories(test-kimi-runtime-routing PRIVATE ../src ../ggml/src ../ggml/src/ggml-cpu)
        target_link_libraries(test-kimi-runtime-routing PRIVATE llama ggml-cpu)
        target_compile_features(test-kimi-runtime-routing PRIVATE cxx_std_17)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_test(NAME kimi_runtime_routing COMMAND test-kimi-runtime-routing)
            set_tests_properties(kimi_runtime_routing PROPERTIES LABELS device SKIP_RETURN_CODE 77)
        endif()
    endif()
    llama_build_and_test(test-kimi-deepseek2-guards.cpp)
    target_include_directories(test-kimi-deepseek2-guards PRIVATE
        ${PROJECT_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
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
endif()

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp)
llama_build_and_test(test-jinja.cpp)
llama_build_and_test(test-chat-literal.cpp ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-jinja NAME test-jinja-py ARGS -py LABEL python)
find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
    add_test(
        NAME test-deepseek4-native-packers
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-deepseek4-native-packers.py ${PROJECT_SOURCE_DIR})
    set_tests_properties(test-deepseek4-native-packers PROPERTIES LABELS python SKIP_RETURN_CODE 77)
    add_test(
        NAME test-moe-copy-lru-sim
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test-moe-copy-lru-sim.py ${PROJECT_SOURCE_DIR})
    set_tests_properties(test-moe-copy-lru-sim PROPERTIES LABELS python)
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
    -P ${CMAKE_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

llama_build_and_test(test-thread-safety.cpp ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES
    FIXTURES_REQUIRED test-download-model
    ENVIRONMENT "CUDA_VISIBLE_DEVICES=0")

llama_build_and_test(test-arg-parser.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp)
endif()
llama_build_and_test(test-gguf.cpp)
llama_build_and_test(test-dsv41-engram.cpp)
target_include_directories(test-dsv41-engram PRIVATE ../src)

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    llama_build(test-llama-mmap-random.cpp)
    target_include_directories(test-llama-mmap-random PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-llama-mmap-random)
    llama_build(test-dsv41-candidate.cpp)
    target_include_directories(test-dsv41-candidate PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-dsv41-candidate)
endif()

set(LLAMA_BACKEND_OPS_SMOKE_FILTER
    "HC_WEIGHTED_SUM(n_embd=64,hc_mult=4,slice_x=0,slice_w=0),\
MUL_MAT(type_a=iq4_xs,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=iq1_s,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=iq1_m,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=q2_K,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
MUL_MAT(type_a=f8_e4m3_b128,type_b=f32,m=1,n=64,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1),\
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
    llama_build(test-q2-vecdot-bench.cpp)
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
llama_build_and_test(test-graph-overhead.cpp)
set_tests_properties(test-graph-overhead PROPERTIES LABELS unit)

llama_build(export-graph-ops.cpp)
target_include_directories(export-graph-ops PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
if (TARGET gguf-model-data)
    target_link_libraries(export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

# The default SSDash CI gate is model-free and requires neither GPU nor cgroup setup.
# Keep this list explicit: other synthetic tests may still require Linux I/O facilities.
foreach(unit_test moe_pack_copy_plan kimi_prefill_cost test-reasoning-budget
                  server_expert_guard test-server-runtime-routing test-dsv41-engram
                  kv_cache_iswa_state_at)
    if (TEST ${unit_test})
        set_property(TEST ${unit_test} APPEND PROPERTY LABELS unit)
    endif()
endforeach()

# Registered Linux I/O fixtures are environment-dependent even without models.
# Keep the portable policy/graph tests separate and preserve known-failure labels.
get_property(ssdash_registered_tests DIRECTORY PROPERTY TESTS)
foreach(resource_test IN LISTS ssdash_registered_tests)
    if (resource_test MATCHES "^(kimi_cpu_sink|kimi_cpu_prefill|kimi_prefetch_tier|dsv41_q2_cpu_sink_decode)")
        if (resource_test MATCHES "^(kimi_cpu_sink_cache_compute_policy|kimi_cpu_prefill_graph)$")
            set_property(TEST ${resource_test} APPEND PROPERTY LABELS unit)
        else()
            set_property(TEST ${resource_test} APPEND PROPERTY LABELS device)
            set_property(TEST ${resource_test} PROPERTY SKIP_RETURN_CODE 77)
            if (NOT resource_test MATCHES "^(kimi_cpu_sink|kimi_cpu_sink_expert_first_call|kimi_cpu_sink_schedule_invalid|kimi_cpu_sink_registered|kimi_cpu_sink_trace_profile|dsv41_q2_cpu_sink_decode|kimi_cpu_prefill|kimi_cpu_prefill_registered)$")
                set_property(TEST ${resource_test} APPEND PROPERTY LABELS cgroup)
            endif()
        endif()
    endif()
endforeach()
foreach(gpu_test dsv41_q2_pack_load kimi_dense_stream)
    if (TEST ${gpu_test})
        set_property(TEST ${gpu_test} APPEND PROPERTY LABELS gpu)
    endif()
endforeach()
