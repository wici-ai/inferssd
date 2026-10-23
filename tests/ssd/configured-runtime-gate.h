// SPDX-License-Identifier: MIT
#pragma once

#include "../../ggml/src/ggml-ssd/settings/settings.h"
#include <dirent.h>
#include <csignal>

static size_t configured_task_count() {
    DIR * directory = opendir("/proc/self/task");
    require(directory != nullptr, "open task inventory");
    size_t count = 0;
    while (const auto * entry = readdir(directory)) { if (entry->d_name[0] != '.') { ++count; } }
    closedir(directory);
    return count;
}

static ssd::settings::snapshot configured_settings(int threads, bool invalid_affinity = false, int prefetch = 0) {
    std::vector<ssd::settings::override_value> values = {
        { "cpu.threads", std::to_string(threads) }, { "prefetch.depth", std::to_string(prefetch) }, { "vram.budget", "0" },
    };
    if (invalid_affinity) { values.push_back({ "cpu.compute_cpus", std::to_string(CPU_SETSIZE) }); }
    const auto parsed = ssd::settings::parse("", "", values);
    require(bool(parsed), parsed.error);
    return parsed.value;
}

static void configured_writer_failure_test(fixture & f, const ssd::settings::snapshot & settings,
        const std::string & path, const std::vector<float> & input) {
    const pid_t child = fork(); require(child >= 0, "fork configured writer failure");
    if (!child) {
        auto writer = ssd::telemetry::writer::open(path);
        if (!writer) { _exit(1); }
        const ssd::runtime_services services{ writer.value.get(), nullptr };
        char error[512] = {};
        sink_ptr sink(ssd::cpu::create_configured(&settings, &services, fixture::resolve, &f, error, sizeof(error)),
            kimi_cpu_sink_destroy);
        if (!sink) { _exit(2); }
        signal(SIGXFSZ, SIG_IGN);
        const struct rlimit limit = { 0, 0 };
        if (setrlimit(RLIMIT_FSIZE, &limit)) { _exit(3); }
        tensors meta(f, 1); const int32_t id = 1; const float weight = 1;
        std::vector<float> output(f.latent); kimi_cpu_sink_stats stats = {};
        const bool ok = kimi_cpu_sink_run(sink.get(), meta.role[0], meta.role[1], meta.role[2], input.data(), &id,
            &weight, 1, 4, 25, output.data(), &stats, error, sizeof(error));
        sink.reset();
        _exit(!ok && stats.pending_reads == 0 && stats.read_requests == stats.read_completions && error[0] ? 0 : 4);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
        "real telemetry write failure did not reject publication after draining");
    unlink(path.c_str());
}

static void configured_runtime_tests(fixture_inputs & inputs) {
    fixture f; make_synthetic(f);
    const size_t before = configured_task_count();
    auto settings = configured_settings(2);
    const auto original = settings;
    const std::string path = f.path + ".jsonl";
    struct cleanup { std::string path; ~cleanup() { unlink(path.c_str()); } } cleanup{ path };
    auto writer = ssd::telemetry::writer::open(path);
    require(bool(writer) && bool(writer.value->settings(settings)), "configured settings publication");
    ssd::telemetry::debug_channels debug; debug.profile = true;
    const ssd::runtime_services services{ writer.value.get(), &debug };
    char error[512] = {};
    sink_ptr first(ssd::cpu::create_configured(&settings, &services, fixture::resolve, &f, error, sizeof(error)),
        kimi_cpu_sink_destroy);
    require(bool(first) && configured_task_count() == before + 1, std::string("configured two-thread create: ") + error);
    settings = configured_settings(2, false, 1); debug.profile = false;
    const ssd::runtime_services second_services{ nullptr, &debug };
    sink_ptr second(ssd::cpu::create_configured(&settings, &second_services, fixture::resolve, &f, error, sizeof(error)),
        kimi_cpu_sink_destroy);
    require(bool(second) && configured_task_count() == before + 5, "independent second context thread count");
    auto invalid = configured_settings(2, true);
    sink_ptr rejected(ssd::cpu::create_configured(&invalid, nullptr, fixture::resolve, &f, error, sizeof(error)), kimi_cpu_sink_destroy);
    require(!rejected && error[0] && configured_task_count() == before + 5, "affinity rejection leaked a started worker");
    require(ssd::cpu::cache_make_room_configured(nullptr, &original, nullptr, 0, 0) &&
        ssd::cpu::cache_prefill_room_configured(nullptr, &original, nullptr, 0, 0, 0), "early configured ledger admission");
    const std::vector<int32_t> ids{ 1, 2 }; const std::vector<float> weights{ .4f, .6f };
    const std::vector<float> input(f.latent, .13f);
    const auto expected = reference(f, 1, input, ids, weights, 4, 25);
    tensors meta(f, 1); std::vector<float> a, b;
    run_checked(inputs, "configured_profile", f, first.get(), meta, input, ids, weights, expected, 1, 25, 0, 0, &a);
    run_checked(inputs, "configured_quiet", f, second.get(), meta, input, ids, weights, expected, 1, 25, 0, 0, &b);
    require(a == b, "debug channels or independent thread selection changed CPU output bits");
    first.reset(); second.reset(); writer.value.reset();
    require(configured_task_count() == before, "configured context destruction left worker threads");
    std::ifstream records(path); std::string line; size_t sequence = 0; bool profile = false;
    while (std::getline(records, line)) {
        const auto event = nlohmann::json::parse(line);
        require(event.at("v") == 1 && event.at("seq") == sequence++, "configured JSONL sequence/version");
        if (event.at("ev") == "ssd.settings") { require(event.at("values").at("cpu.threads") == 2 && event.at("values").at("prefetch.depth") == 0, "owned settings receipt changed"); }
        else { profile |= event.at("compute_thread_cpu_us").get<double>() > 0; }
    }
    require(sequence == 3 && profile, "copied debug selection or actual layer telemetry missing");
    configured_writer_failure_test(f, original, path + ".failure", input);
    std::printf("CONFIGURED_RUNTIME_IO_GOLDEN_PASS contexts=2 canonical_ring=8 owned_snapshot=1 owned_debug=1\n");
}
