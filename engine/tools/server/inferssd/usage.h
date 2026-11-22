// SPDX-License-Identifier: Apache-2.0
#pragma once

// Usage events for the host that runs the server: a request_received line when an inference request reaches its
// route and at most one usage_finalized line when it ends. INFERSSD_USAGE_DIR selects the directory; without it
// nothing is written. The format is described in docs/usage-events.md.
//
// The server log names each request's ID where the request starts and where a slot takes one of its tasks, so
// a record here and the log lines of the same request can be read together.

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

enum server_usage_outcome {
    SERVER_USAGE_FAILED,
    SERVER_USAGE_CANCELLED,
    SERVER_USAGE_COMPLETED,
};

struct server_usage_log;

// What a request says of itself: the route it came to and, from its headers, who sent it.
struct server_usage_caller {
    std::string route;
    // From the request's API key, by which a host tells apart the tools it points at the server: the name in a
    // key inferssd-<name>, or the first 10 characters of any other key. Empty without a key.
    std::string client;
    // The User-Agent header, printable ASCII, cut to 120 characters; empty without one.
    std::string user_agent;

    static server_usage_caller from_request(const std::string & route, const std::map<std::string, std::string> & headers);
};

// One HTTP request. The route and each of its tasks hold a reference; usage_finalized is written when the last
// reference goes, or earlier by finish() when the server stops.
struct server_usage_request {
    server_usage_request(std::shared_ptr<server_usage_log> log, std::string request_id);
    ~server_usage_request();

    server_usage_request(const server_usage_request &)             = delete;
    server_usage_request & operator=(const server_usage_request &) = delete;

    // the counts of one task that reached a slot
    void add(int64_t input_tokens, int64_t output_tokens);
    // what one task's prompt and output took: the prompt tokens its slot already held, and the milliseconds
    // spent on the rest of the prompt and on the output
    void add_work(int64_t cached_tokens, double prompt_ms, double output_ms);
    void set_outcome(server_usage_outcome outcome);
    // why the request failed, in the words its response gives; the first reason is kept
    void set_error(const std::string & message);
    const std::string & id() const { return request_id; }
    // write usage_finalized unless it is already written
    void finish();

  private:
    std::shared_ptr<server_usage_log> log;
    std::string                       request_id;
    std::atomic<int64_t>              input_tokens{ 0 };
    std::atomic<int64_t>              output_tokens{ 0 };
    std::atomic<int64_t>              cached_tokens{ 0 };
    // microseconds, so that they add up as integers
    std::atomic<int64_t>              prompt_us{ 0 };
    std::atomic<int64_t>              output_us{ 0 };
    std::mutex                        error_mutex;
    std::string                       error;
    std::atomic<int>                  outcome{ SERVER_USAGE_FAILED };
    std::atomic<bool>                 finished{ false };
};

using server_usage_request_ptr = std::shared_ptr<server_usage_request>;

struct server_usage_log : std::enable_shared_from_this<server_usage_log> {
    // the log INFERSSD_USAGE_DIR selects, or nullptr when it is unset or its file cannot be opened
    static std::shared_ptr<server_usage_log> from_environment();

    ~server_usage_log();

    // write request_received for a new request
    server_usage_request_ptr begin(const server_usage_caller & caller);

    // finish every request still open; the server calls this after its loop has stopped
    void finish_open();

  private:
    friend struct server_usage_request;

    server_usage_log(int fd, std::filesystem::path path);

    // what usage_finalized reports besides the request's ID
    struct final_usage {
        server_usage_outcome outcome;
        int64_t              input_tokens;
        int64_t              output_tokens;
        int64_t              cached_tokens;
        int64_t              prompt_us;
        int64_t              output_us;
        std::string          error;
    };

    void request_received_locked(const std::string & request_id, const server_usage_caller & caller);
    void usage_finalized(const std::string & request_id, const final_usage & usage);
    void forget(const server_usage_request * request);
    // append one complete line; the caller holds mutex
    void write_locked(std::string line);

    std::mutex            mutex;
    int                   fd;
    std::filesystem::path path;
    std::string           run_id;
    uint64_t              seq    = 0;
    bool                  failed = false;

    std::unordered_map<const server_usage_request *, std::weak_ptr<server_usage_request>> open;
};
