#include "../ggml/src/ggml-cuda/q4-colibri-worker-state.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

struct delayed_completion {
    std::mutex mutex;
    std::condition_variable cv;
    bool release = false;
    q4_colibri_slot_marker completion;
};

int main() {
    using namespace std::chrono_literals;
    assert(q4_colibri_slot_transition_allowed(q4_colibri_slot_state::free, q4_colibri_slot_state::reserved));
    assert(q4_colibri_slot_transition_allowed(q4_colibri_slot_state::reserved, q4_colibri_slot_state::reading));
    assert(q4_colibri_slot_transition_allowed(q4_colibri_slot_state::reading, q4_colibri_slot_state::h2d_ready));
    assert(!q4_colibri_slot_transition_allowed(q4_colibri_slot_state::reading, q4_colibri_slot_state::free));

    q4_colibri_slot_marker slot{0x101, 7, 4096};
    const q4_colibri_slot_marker current = slot;
    assert(q4_colibri_slot_marker_matches(slot, current));
    assert(!q4_colibri_slot_marker_matches(slot, {slot.key, slot.generation - 1, slot.bytes}));
    assert(!q4_colibri_slot_marker_matches(slot, {0x202, slot.generation, slot.bytes}));

    delayed_completion delayed;
    delayed.completion = current;
    std::atomic<bool> completed{false};
    std::thread worker([&] {
        std::unique_lock<std::mutex> lock(delayed.mutex);
        delayed.cv.wait(lock, [&] { return delayed.release; });
        completed.store(q4_colibri_slot_marker_matches(slot, delayed.completion));
    });
    std::this_thread::sleep_for(5ms);
    slot = {0x303, 8, 4096};
    {
        std::lock_guard<std::mutex> lock(delayed.mutex);
        delayed.release = true;
    }
    delayed.cv.notify_one();
    worker.join();
    assert(!completed.load());

    delayed_completion current_delayed;
    current_delayed.completion = slot;
    std::thread current_worker([&] {
        std::unique_lock<std::mutex> lock(current_delayed.mutex);
        current_delayed.cv.wait(lock, [&] { return current_delayed.release; });
        completed.store(q4_colibri_slot_marker_matches(slot, current_delayed.completion));
    });
    {
        std::lock_guard<std::mutex> lock(current_delayed.mutex);
        current_delayed.release = true;
    }
    current_delayed.cv.notify_one();
    current_worker.join();
    assert(completed.load());

    const uint64_t object_offset = 4096;
    const uint64_t role_bytes = 8257536;
    const uint64_t object_bytes = role_bytes * 3;
    if (object_offset % 4096 != 0 || object_bytes % 4096 != 0 ||
            0 + role_bytes > object_bytes || role_bytes + role_bytes > object_bytes ||
            2 * role_bytes + role_bytes != object_bytes) {
        return 2;
    }

    const int requested_role_slices = 3;
    const int physical_object_jobs = 1;
    if (requested_role_slices - physical_object_jobs != 2) return 3;

    const q4_colibri_object_marker object{1, 2, 17, 0xabc, 4096, (size_t)object_bytes};
    assert(q4_colibri_object_marker_matches(object, object));
    assert(!q4_colibri_object_marker_matches(object,
        {object.source_idx, object.layer_idx, object.expert_idx,
         object.manifest_generation + 1, object.offset, object.bytes}));
    assert(!q4_colibri_object_marker_matches(object,
        {object.source_idx, object.layer_idx, object.expert_idx + 1,
         object.manifest_generation, object.offset, object.bytes}));
    q4_colibri_object_marker recycled = object;
    recycled.manifest_generation++;
    assert(!q4_colibri_object_marker_matches(recycled, object));
    recycled = object;
    recycled.offset += 4096;
    assert(!q4_colibri_object_marker_matches(recycled, object));
    return 0;
}
