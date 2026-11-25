// SPDX-License-Identifier: Apache-2.0

#include "server-queue.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <utility>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main() {
    ggml_time_init();
    server_queue tasks;
    server_response responses;
    server_response_reader reader(tasks, responses, 1);

    server_task task(SERVER_TASK_TYPE_METRICS);
    const int id = reader.get_new_id();
    task.id = id;
    reader.post_task(std::move(task));

    // A heartbeat gets a turn when the reader has been idle, with the request still pending.
    const auto connected = [] { return false; };
    require(!reader.next(connected, 1), "idle read did not yield to the heartbeat");
    require(reader.has_next(), "heartbeat ended the request");

    auto result = std::make_unique<server_task_result_metrics>();
    result->id = id;
    responses.send(std::move(result));
    const auto received = reader.next(connected, 1);
    require(received && received->id == id, "completion was not delivered after the heartbeat");
    require(!reader.has_next(), "completed request is still pending");
}
