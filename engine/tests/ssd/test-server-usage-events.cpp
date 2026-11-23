// SPDX-License-Identifier: Apache-2.0
// A usage event says which route a request came to and who sent it, and why it failed, so a host can show each
// request without reading the server log. The caller is known by its API key: a name the host gave it, written
// inferssd-<name>, or the first 10 characters of any other key, which may be a secret. Writes the events of a few
// requests to a temporary directory and reads them back; no model is loaded.
#include "inferssd/usage.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void require(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error(message); }
}

static server_usage_caller caller(const std::map<std::string, std::string> & headers) {
    return server_usage_caller::from_request("/v1/responses", headers);
}

static void test_caller() {
    require(caller({}).route == "/v1/responses", "the route is kept");
    require(caller({}).client.empty() && caller({}).user_agent.empty(), "no headers give no caller");
    require(caller({ { "Authorization", "Bearer inferssd-codex" } }).client == "codex", "a bearer key names the caller");
    require(caller({ { "authorization", "Bearer inferssd-claude-desktop" } }).client == "claude-desktop", "the header's case does not matter");
    require(caller({ { "x-api-key", "inferssd-my_tool2" } }).client == "my_tool2", "X-Api-Key names the caller too");
    require(caller({ { "Authorization", "Bearer inferssd-a" }, { "X-Api-Key", "inferssd-b" } }).client == "a", "Authorization comes first");
    // any other key may be a secret: its first 10 characters name the caller, and no more is kept
    require(caller({ { "Authorization", "Bearer sk-proj-abcdef0123456789" } }).client == "sk-proj-ab", "another key is cut to 10");
    require(caller({ { "Authorization", "Bearer inferssd" } }).client == "inferssd", "the bare prefix is a key of its own");
    require(caller({ { "Authorization", "Bearer inferssd-" } }).client == "inferssd-", "the prefix with no name too");
    require(caller({ { "X-Api-Key", "My Tool" } }).client == "My Tool", "a short key need not look like a name");
    require(caller({ { "Authorization", "Bearer inferssd-" + std::string(200, 'k') } }).client.size() == 64, "a long name is cut to 64");
    require(caller({ { "Authorization", "Bearer a\tb\x01" "c" } }).client == "abc", "only printable ASCII is kept");
    require(caller({ { "User-Agent", "codex_cli_rs/0.169.0 (Mac OS 26.2; arm64)" } }).user_agent == "codex_cli_rs/0.169.0 (Mac OS 26.2; arm64)",
            "the User-Agent is kept");
    require(caller({ { "user-agent", std::string(300, 'a') } }).user_agent.size() == 120, "a long User-Agent is cut");
    require(caller({ { "User-Agent", "a\tb\xc3\xa9\x01" "c" } }).user_agent == "abc", "only printable ASCII is kept");
}

static std::vector<json> events(const std::filesystem::path & directory) {
    std::ifstream      in(directory / "usage-events.jsonl", std::ios::binary);
    std::vector<json>  result;
    for (std::string line; std::getline(in, line);) {
        result.push_back(json::parse(line));
    }
    return result;
}

static void test_events() {
    const auto directory = std::filesystem::temp_directory_path() / ("inferssd-usage-test-" + std::to_string(std::rand()));
    std::filesystem::remove_all(directory);
#if defined(_WIN32)
    _wputenv_s(L"INFERSSD_USAGE_DIR", directory.c_str());
#else
    setenv("INFERSSD_USAGE_DIR", directory.c_str(), 1);
#endif
    const auto log = server_usage_log::from_environment();
    require(log != nullptr, "the log opens in a new directory");
    {
        // completed: its counts, what the slot already held and the times
        auto request = log->begin(server_usage_caller::from_request(
            "/v1/responses", { { "Authorization", "Bearer inferssd-codex" }, { "User-Agent", "codex_cli_rs/0.169.0" } }));
        require(request->id().rfind("req-", 0) == 0, "a request has an ID");
        request->add(812, 28);
        request->add_work(700, 1234.9, 2000.2);
        request->set_outcome(SERVER_USAGE_COMPLETED);
    }
    {
        // failed before it ran, from a caller with no key: the first reason is the one kept
        auto request = log->begin(server_usage_caller::from_request("/v1/chat/completions", {}));
        request->set_error("image input is not supported");
        request->set_error("a later reason");
    }
    const auto lines = events(directory);
    std::filesystem::remove_all(directory);
    require(lines.size() == 4, "two requests write four events");
    require(lines[0].dump() == json({ { "event_id", lines[0].at("event_id") }, { "type", "request_received" },
                                      { "request_id", lines[0].at("request_id") }, { "occurred_at", lines[0].at("occurred_at") },
                                      { "route", "/v1/responses" }, { "client", "codex" }, { "user_agent", "codex_cli_rs/0.169.0" } }).dump(),
            "request_received names the route and the caller: " + lines[0].dump());
    require(lines[1].at("request_id") == lines[0].at("request_id") && lines[1].at("type") == "usage_finalized", "the two events pair");
    require(lines[1].at("input_tokens") == 812 && lines[1].at("output_tokens") == 28 && lines[1].at("token_count") == 840, "counts");
    require(lines[1].at("cached_tokens") == 700 && lines[1].at("prompt_ms") == 1234 && lines[1].at("output_ms") == 2000, "work");
    require(lines[1].at("outcome") == "completed" && !lines[1].contains("error"), "a completed request has no error");
    require(lines[2].at("route") == "/v1/chat/completions" && !lines[2].contains("client") && !lines[2].contains("user_agent"),
            "an unnamed caller leaves the fields out: " + lines[2].dump());
    require(lines[3].at("outcome") == "failed" && lines[3].at("error") == "image input is not supported", "the first reason is kept");
    require(lines[3].at("cached_tokens") == 0 && lines[3].at("prompt_ms") == 0 && lines[3].at("output_ms") == 0, "nothing ran");
}

int main(int argc, char **) {
    try {
        require(argc == 1, "usage: test-server-usage-events");
        test_caller();
        test_events();
        std::puts("usage events: PASS (no model loaded)");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
