// SPDX-License-Identifier: MIT
#include "options.h"
#if defined(LLAMA_SSD_STORE)
#include "ggml-ssd/store/index.h"
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <mutex>
#if defined(__APPLE__)
#include <crt_externs.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
extern char ** environ;
#endif

namespace {
// One factory in the llama DSO. Backend copies of core code never create a
// process singleton; every executor borrows this context's registration.
#if defined(__linux__)
std::shared_ptr<ssd::host::process_arbiter> host_owner() {
    static std::mutex mutex;
    static std::weak_ptr<ssd::host::process_arbiter> current;
    std::lock_guard<std::mutex> lock(mutex);
    auto owner = current.lock();
    if (!owner) { owner = std::make_shared<ssd::host::process_arbiter>(); current = owner; }
    return owner;
}
#endif

void error_text(char * output, size_t size, const char * message) noexcept {
    if (output && size) { std::snprintf(output, size, "%s", message); }
}

// Inspect names only. Never copy or report environment values.
ssd::result<void> check_environment() {
    std::vector<std::string> names;
#if defined(_WIN32)
    struct environment {
        char * value = GetEnvironmentStringsA();
        ~environment() { if (value) { FreeEnvironmentStringsA(value); } }
    } block;
    if (!block.value) { return {ssd::status::io, "cannot enumerate environment names"}; }
    for (const char * entry = block.value; *entry; entry += std::strlen(entry) + 1) {
        const char * equals = std::strchr(entry, '=');
        if (equals && equals != entry) { names.emplace_back(entry, size_t(equals - entry)); }
    }
#else
#if defined(__APPLE__)
    char ** entries = *_NSGetEnviron();
#else
    char ** entries = environ;
#endif
    for (; entries && *entries; ++entries) {
        const char * equals = std::strchr(*entries, '=');
        if (equals) { names.emplace_back(*entries, size_t(equals - *entries)); }
    }
#endif
    return ssd::settings::reject_retired_environment(names);
}
} // namespace

llama_ssd_options * llama_ssd_options_create(const char * file,
        const llama_ssd_override * overrides, size_t count, const char * telemetry,
        char * error, size_t error_size) {
    error_text(error, error_size, "");
    try {
        if (count && !overrides) { throw std::invalid_argument("missing SSD overrides"); }
        const auto environment = check_environment();
        if (!environment) { error_text(error, error_size, environment.error.c_str()); return nullptr; }
        std::vector<ssd::settings::override_value> values;
        for (size_t i = 0; i < count; ++i) {
            if (!overrides[i].key || !overrides[i].value) { throw std::invalid_argument("null SSD override"); }
            values.push_back({overrides[i].key, overrides[i].value});
        }
#if defined(__APPLE__)
        const char * platform = "darwin";
#else
        const char * platform = "linux";
#endif
        auto parsed = ssd::settings::load(file ? file : "", values, platform);
        if (!parsed) { error_text(error, error_size, parsed.error.c_str()); return nullptr; }
        auto debug = ssd::telemetry::read_debug_environment();
        if (!debug) { error_text(error, error_size, debug.error.c_str()); return nullptr; }
        return new llama_ssd_options{std::move(parsed.value), std::move(debug.value), telemetry ? telemetry : ""};
    } catch (const std::exception & exception) {
        error_text(error, error_size, exception.what());
    } catch (...) {
        error_text(error, error_size, "SSD options failed");
    }
    return nullptr;
}

void llama_ssd_options_free(llama_ssd_options * options) { delete options; }

bool llama_ssd_check_environment(char * error, size_t size) {
    error_text(error, size, "");
    try {
        const auto checked = check_environment();
        if (checked) { return true; }
        error_text(error, size, checked.error.c_str());
    } catch (const std::exception & exception) {
        error_text(error, size, exception.what());
    } catch (...) {
        error_text(error, size, "environment name check failed");
    }
    return false;
}

size_t llama_ssd_schema_json(char * output, size_t size) {
    if (output && size) { output[0] = '\0'; }
    const auto schema = ssd::settings::schema_json();
    if (!schema || schema.value.size() == SIZE_MAX) { return 0; }
    const size_t required = schema.value.size() + 1;
    if (output && size >= required) { std::memcpy(output, schema.value.c_str(), required); }
    return required;
}

llama_ssd_context::llama_ssd_context(const llama_ssd_options & options)
    : settings(options.settings), debug(options.debug) {
#if defined(__linux__)
    if (!settings.sources()->empty()) {
        host_owner = ::host_owner();
        auto registered = host_owner->make_client();
        if (!registered) { throw std::runtime_error(registered.error); }
        host = std::move(registered.value);
#if defined(LLAMA_SSD_STORE)
        auto opened = ssd::store::source_index::open(*settings.sources());
        if (!opened) { throw std::runtime_error(opened.error); }
        sources = opened.value->view();
        source_owner = std::unique_ptr<void, llama_ssd_source_delete>(opened.value.release(),
            llama_ssd_source_delete{[](void * value) noexcept { delete static_cast<ssd::store::source_index *>(value); }});
#else
        throw std::runtime_error("SSD experts require LLAMA_SSD_STORE=ON");
#endif
    }
#endif
    if (!options.telemetry_file.empty()) {
        auto opened = ssd::telemetry::writer::open(options.telemetry_file);
        if (!opened) { throw std::runtime_error(opened.error); }
        events = std::move(opened.value);
        const auto published = events->settings(settings);
        if (!published) { throw std::runtime_error(published.error); }
    }
    const auto effective = settings.json();
    if (!effective) { throw std::runtime_error(effective.error); }
    std::fprintf(stderr, "[ssd.settings] %s\n", effective.value.c_str());
    services = {events.get(), &debug, host.get()};
}

llama_ssd_context::~llama_ssd_context() {
    // Context backends have already synchronized and been destroyed. This
    // callback joins the Mac reader jobs before events closes below.
    if (mac_cache) { destroy_mac(mac_cache); }
}
