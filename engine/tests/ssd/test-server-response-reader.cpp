// SPDX-License-Identifier: Apache-2.0

#include "server-queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <thread>

using namespace std::chrono_literals;

int main() {
    ggml_time_init();
    server_response responses;
    responses.add_waiting_task_id(1);
    responses.add_waiting_task_id(2);

    std::atomic<bool> done{false};
    std::promise<void> primed;
    auto ready = primed.get_future();
    std::thread other_task([&] {
        for (size_t i = 0; i < 65 && !done.load(); ++i) {
            auto result = std::make_unique<server_task_result_metrics>();
            result->id = 2;
            responses.send(std::move(result));
            if (i == 0) {
                primed.set_value();
            }
            std::this_thread::sleep_for(40ms);
        }
    });
    ready.get();
    const auto start = std::chrono::steady_clock::now();
    const auto result = responses.recv_with_timeout({1}, 1);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    done.store(true);
    other_task.join();

    if (result || elapsed < 900ms || elapsed >= 2s) {
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        std::fprintf(stderr, "Unrelated results changed the one-second deadline: %lld ms\n", static_cast<long long>(ms));
        return 1;
    }
    return 0;
}
