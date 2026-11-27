#include "server-task.h"
#include "server-prompt-cache-fixture.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <initializer_list>
#include <list>
#include <string>
#include <utility>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static server_prompt make_prompt(std::initializer_list<llama_token> tokens) {
    server_prompt prompt;
    prompt.tokens = server_tokens(llama_tokens(tokens), false);
    return prompt;
}

static void add_checkpoint(server_prompt & prompt, int64_t n_tokens) {
    server_prompt_checkpoint checkpoint = {};
    checkpoint.pos_min  = 0;
    checkpoint.pos_max  = n_tokens;
    checkpoint.n_tokens = n_tokens;
    checkpoint.data     = { 1, 2, 3, 4 };
    prompt.checkpoints.push_back(std::move(checkpoint));
}

static void add_checkpoint_with_bounds(server_prompt & prompt, llama_pos pos_max, int64_t n_tokens, uint8_t marker) {
    server_prompt_checkpoint checkpoint = {};
    checkpoint.pos_min  = 0;
    checkpoint.pos_max  = pos_max;
    checkpoint.n_tokens = n_tokens;
    checkpoint.data     = { marker };
    prompt.checkpoints.push_back(std::move(checkpoint));
}

static void test_find_checkpoint_before_tail_truncation_pos() {
    server_prompt prompt = make_prompt({ 1, 2, 3, 4, 5, 6, 7, 8 });

    add_checkpoint_with_bounds(prompt, 3, 4, 4);
    add_checkpoint_with_bounds(prompt, 5, 6, 6);
    add_checkpoint_with_bounds(prompt, 7, 12, 12); // invalid: more tokens than prompt

    const server_prompt_checkpoint * latest = server_prompt_find_checkpoint_before_pos(prompt, 7);
    require(latest != nullptr, "expected a checkpoint before tail truncation position");
    require(latest->n_tokens == 6, "expected latest compatible checkpoint before p0");
    require(latest->data == std::vector<uint8_t>{ 6 }, "expected latest compatible checkpoint data");

    const server_prompt_checkpoint * earlier = server_prompt_find_checkpoint_before_pos(prompt, 5);
    require(earlier != nullptr, "expected an earlier checkpoint before p0");
    require(earlier->n_tokens == 4, "expected checkpoint with pos_max strictly before p0");

    require(server_prompt_find_checkpoint_before_pos(prompt, 3) == nullptr,
            "checkpoint at pos_max >= p0 must not be used for tail truncation restore");
}

static void test_oaicompat_chat_streams_reasoning_delta() {
    common_chat_parser_params parser_params;
    parser_params.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
    parser_params.generation_prompt = "<think>";

    task_result_state state(parser_params);

    server_task_result_cmpl_partial partial = {};
    partial.content = "I am thinking";
    partial.n_decoded = 1;
    partial.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
    partial.oaicompat_model = "test-model";
    partial.oaicompat_cmpl_id = "chatcmpl-test";
    partial.update(state);

    json chunks = partial.to_json_oaicompat_chat();
    bool found_reasoning = false;
    for (const auto & chunk : chunks) {
        if (!chunk.contains("choices") || chunk.at("choices").empty()) {
            continue;
        }
        const auto & delta = chunk.at("choices").at(0).at("delta");
        if (delta.contains("reasoning_content") && delta.at("reasoning_content") == "I am thinking") {
            found_reasoning = true;
        }
    }

    require(found_reasoning, "streaming chat response should broadcast reasoning_content deltas");
}

static void test_oaicompat_chat_final_contains_reasoning() {
    server_task_result_cmpl_final final = {};
    final.res_type = TASK_RESPONSE_TYPE_OAI_CHAT;
    final.oaicompat_model = "test-model";
    final.oaicompat_cmpl_id = "chatcmpl-test";
    final.include_usage = true;
    final.oaicompat_msg.role = "assistant";
    final.oaicompat_msg.reasoning_content = "I am thinking";
    final.oaicompat_msg.content = "The answer is 4.";

    json body = final.to_json_oaicompat_chat();
    const auto & message = body.at("choices").at(0).at("message");
    require(message.contains("reasoning_content"), "final chat response should contain reasoning_content");
    require(message.at("reasoning_content") == "I am thinking",
            "final chat response should extract reasoning before </think>");
    require(message.at("content") == "The answer is 4.",
            "final chat response should keep post-thinking content separate");
}

static void test_full_removal_keeps_exact_shorter_without_checkpoint() {
    server_prompt_cache cache(0, 0);

    server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
    server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });

    require(server_prompt_cache_testing::alloc_metadata(cache, long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "failed to allocate initial long prompt");
    require(server_prompt_cache_testing::alloc_metadata(cache, short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "short exact prompt should be cached when the longer state has no restorable prefix checkpoint");
    require(cache.states.size() == 2,
            "cache should retain both long and short prompts without a restorable prefix checkpoint");
}

static void test_full_removal_reuses_longer_checkpoint_for_shorter_prompt() {
    server_prompt_cache cache(0, 0);

    server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
    server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
    add_checkpoint(long_prompt, short_prompt.n_tokens());

    require(server_prompt_cache_testing::alloc_metadata(cache, long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
            "failed to allocate checkpointed long prompt");
    require(server_prompt_cache_testing::alloc_metadata(cache, short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) == nullptr,
            "short exact prompt should be skipped when a longer checkpoint can restore it");
    require(cache.states.size() == 1,
            "checkpointed long prompt should make the exact shorter prompt redundant");
}

static void test_full_removal_only_removes_obsolete_shorter_with_checkpoint() {
    {
        server_prompt_cache cache(0, 0);

        server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
        server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });

        require(server_prompt_cache_testing::alloc_metadata(cache, short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate initial short prompt");
        require(server_prompt_cache_testing::alloc_metadata(cache, long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "long prompt without checkpoint should still be cached");
        require(cache.states.size() == 2,
                "short prompt must not be removed when long prompt cannot restore that prefix");
    }

    {
        server_prompt_cache cache(0, 0);

        server_prompt short_prompt = make_prompt({ 1, 2, 3, 4 });
        server_prompt long_prompt  = make_prompt({ 1, 2, 3, 4, 5, 6 });
        add_checkpoint(long_prompt, short_prompt.n_tokens());

        require(server_prompt_cache_testing::alloc_metadata(cache, short_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate initial short prompt");
        require(server_prompt_cache_testing::alloc_metadata(cache, long_prompt, 8, COMMON_CONTEXT_SEQ_RM_TYPE_FULL) != nullptr,
                "failed to allocate checkpointed long prompt");
        require(cache.states.size() == 1,
                "short prompt should be removed once long prompt has a restorable prefix checkpoint");
        require(cache.states.front().n_tokens() == long_prompt.n_tokens(),
                "remaining cache entry should be the checkpointed long prompt");
    }
}

static server_prompt_points shared_at(int64_t n_shared, int64_t n_context = 0, std::vector<int64_t> branch = {}) {
    server_prompt_points points;
    points.n_shared  = n_shared;
    points.n_context = n_context;
    points.branch    = std::move(branch);
    return points;
}

// The batches update_slots fills for a prompt of n_prompt tokens from n_start, for a memory that is checkpointed only
// between batches.
static std::vector<int64_t> prompt_batches(int64_t n_start, int64_t n_prompt, const server_prompt_points & points,
                                           int64_t n_batch) {
    std::vector<int64_t> batches;
    for (int64_t n_done = n_start; n_done < n_prompt;) {
        int64_t n = 0;
        while (n_done < n_prompt && n < n_batch) {
            ++n_done;
            ++n;
            if (server_prompt_batch_stops(n_done, n_prompt, points, n_batch)) {
                break;
            }
        }
        batches.push_back(n);
    }
    return batches;
}

static void test_prompt_batches_stop_at_shared_prefix_and_end() {
    // a new conversation behind a system prompt and tools of 8503 tokens: the prefix, the rest, the last 4 tokens
    require(prompt_batches(0, 8673, shared_at(8503), 65536) == std::vector<int64_t>({ 8503, 166, 4 }),
            "a new prompt stops after its shared prefix and 4 tokens before its end");
    // the next turn of a chat without a shared prefix, from the previous turn's checkpoint: no stop n_ubatch earlier
    require(prompt_batches(987, 2068, shared_at(0), 65536) == std::vector<int64_t>({ 1077, 4 }),
            "a later turn stops only 4 tokens before its end");
    // a prompt resumed at the shared prefix (from the prompt cache) or after it does not stop there again
    require(prompt_batches(8503, 8675, shared_at(8503), 65536) == std::vector<int64_t>({ 168, 4 }),
            "a prompt resumed at its shared prefix does not stop there");
    require(prompt_batches(8669, 8673, shared_at(8503), 65536) == std::vector<int64_t>({ 4 }),
            "a prompt resumed after its shared prefix stops only at its end");
    // batches still end at n_batch
    require(prompt_batches(0, 1200, shared_at(0), 512) == std::vector<int64_t>({ 512, 512, 172, 4 }),
            "a long prompt is still split at n_batch");
    // a shared prefix that is too short, or ends inside the last 4 tokens, adds no stop
    require(prompt_batches(0, 200, shared_at(40), 65536) == std::vector<int64_t>({ 196, 4 }),
            "a shared prefix under 64 tokens adds no stop");
    require(prompt_batches(0, 100, shared_at(98), 65536) == std::vector<int64_t>({ 96, 4 }),
            "a shared prefix inside the last 4 tokens adds no stop");
}

// Codex with a user's own MCP server: its app tools end at 6000, the server's at 7000, the app text at 11000, the
// user's context at 12500, where the approved commands start at 12400. A memory checkpointed only between batches
// stops at the kept prefixes and the end, not at the branch points: each stop reads the weights once more.
static void test_prompt_batches_stop_at_kept_prefixes_only() {
    const server_prompt_points points = shared_at(11000, 12500, { 6000, 7000, 12400 });
    require(prompt_batches(0, 12700, points, 65536) == std::vector<int64_t>({ 11000, 1500, 196, 4 }),
            "a new prompt stops at its shared and context prefixes and before its end");
    require(prompt_batches(11000, 12700, points, 65536) == std::vector<int64_t>({ 1500, 196, 4 }),
            "a prompt resumed at the shared prefix stops only at the points after it");
}

// A sliding-window model's prompt resumed at 8192 tokens, with batches of 2048 and ubatches of 512: a batch that holds
// a point ends with the ubatch that holds it, so that the state there can still be saved after the batch.
static void test_batches_end_with_the_ubatch_of_a_point() {
    const server_prompt_points points = shared_at(9786, 10730, { 7820, 9205, 9784, 10728 });
    std::vector<int64_t> batches;
    for (int64_t n_done = 8192; n_done < 10994;) {
        int64_t n = 0;
        bool past_point = false;
        while (n_done < 10994 && n < 2048) {
            ++n_done;
            ++n;
            past_point = past_point || points.has(n_done);
            if (server_prompt_batch_ends_for_point(past_point, n, 512)) {
                break;
            }
        }
        batches.push_back(n);
    }
    require(batches == std::vector<int64_t>({ 1024, 1024, 512, 242 }), "each batch ends with the ubatch of its first point");
}

static void test_prompt_points() {
    const server_prompt_points points = shared_at(11000, 12500, { 6000, 7000, 12400 });
    require(points.all() == std::vector<int64_t>({ 6000, 7000, 11000, 12400, 12500 }), "every point, sorted");
    require(points.has(7000) && points.has(12500) && !points.has(7001), "a point is one of them");
    require(points.kept(11000) && points.kept(12500) && !points.kept(7000), "the prefixes are kept, branch points not");
    require(points.has(6000 - 6000 % 128, 128) && points.kept(12500 - 12500 % 128, 128),
            "rounded to a memory's alignment");

    // the prompt's text from token 9000 to 9200 became 30 tokens fewer
    server_prompt_points moved = points;
    moved.replaced(9000, 9200, -30);
    require(moved.n_shared == 10970 && moved.n_context == 12470 &&
                moved.branch == std::vector<int64_t>({ 6000, 7000, 12370 }),
            "points after the replaced tokens move with them");
    server_prompt_points inside = points;
    inside.replaced(6500, 11500, 10);
    require(inside.n_shared == 0 && inside.n_context == 12510 && inside.branch == std::vector<int64_t>({ 6000, 12410 }),
            "points inside the replaced tokens are lost");
}

static void test_checkpoints_at_points_are_kept_by_count() {
    const server_prompt_points points = shared_at(11000, 12500, { 6000, 7000 });
    std::list<server_prompt_checkpoint> checkpoints;
    const auto add = [&](int64_t n) {
        server_prompt_checkpoint cp = {};
        cp.n_tokens = n;
        cp.data     = { 1 };
        checkpoints.push_back(cp);
    };
    add(6000);
    add(7000);
    add(11000);
    // --ctx-checkpoints 1: the points' checkpoints stay, a new one at a point needs no room by count
    require(server_prompt_checkpoint_over_count(checkpoints, points, 1, 12500, 1) == checkpoints.end(),
            "a checkpoint at a point drops none by count");
    require(server_prompt_checkpoint_over_count(checkpoints, points, 1, 12696, 1) == checkpoints.end(),
            "no other checkpoint yet: the first ordinary one drops none");
    add(12696);
    const auto it = server_prompt_checkpoint_over_count(checkpoints, points, 1, 14000, 1);
    require(it != checkpoints.end() && it->n_tokens == 12696, "the next ordinary one drops the ordinary one, not a point's");
    require(server_prompt_checkpoint_over_count(checkpoints, shared_at(0), 1, 14000, 1) == checkpoints.begin(),
            "without points, the oldest goes as before");
}

// Codex's next conversation behind the same 9207 tokens of instructions and tools, while the slot holds the 15,000
// tokens of the conversation before: a sliding-window or hybrid model keeps none of the old conversation's tail, so the
// old state is saved; a model that truncates keeps it as before (upstream's rule: less than half kept).
static void test_a_state_that_cannot_roll_back_is_saved_when_left() {
    const float f_keep = 9207.0f / 15000.0f;
    require(server_prompt_save_before_switch(f_keep, COMMON_CONTEXT_SEQ_RM_TYPE_FULL, 9207, 15000),
            "a state that cannot be rolled back is saved when a new prompt parts from it");
    require(!server_prompt_save_before_switch(f_keep, COMMON_CONTEXT_SEQ_RM_TYPE_PART, 9207, 15000),
            "a state that truncates keeps upstream's rule");
    require(server_prompt_save_before_switch(4000.0f / 15000.0f, COMMON_CONTEXT_SEQ_RM_TYPE_PART, 4000, 15000),
            "less than half kept: saved");
    require(!server_prompt_save_before_switch(1.0f, COMMON_CONTEXT_SEQ_RM_TYPE_FULL, 15000, 15000),
            "the next turn of the same conversation extends the state: not saved");
}

static void test_shared_prefix_serves_a_new_conversation() {
    // The previous conversation as the slot leaves it, with its one checkpoint 4 tokens before the end of its prompt,
    // and the shared prefix the prompt cache keeps; a new conversation shares only the prefix.
    server_prompt previous = make_prompt({ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });
    add_checkpoint(previous, 8);
    const server_prompt prefix = make_prompt({ 1, 2, 3, 4, 5, 6 });
    const server_tokens next(llama_tokens({ 1, 2, 3, 4, 5, 6, 20, 21, 22, 23 }), false);

    const server_prompt_checkpoint * checkpoint = nullptr;
    require(server_prompt_reuse_full(previous, next, &checkpoint) == -1,
            "the previous conversation's state and checkpoint lie past where the new one branches off");
    require(server_prompt_reuse_full(prefix, next, &checkpoint) == 6 && checkpoint == nullptr,
            "the shared prefix restores its whole state for the new conversation");
}

// a cached prompt file as the cache writes it (no checkpoints), with `n_data` bytes of state
static void write_prompt_file(const std::string & path, const std::string & fingerprint,
                              std::initializer_list<llama_token> tokens, uint64_t n_data) {
    FILE * f = std::fopen(path.c_str(), "wb");
    require(f != nullptr, "cannot write a prompt file");
    const uint32_t magic = 0x4643504c, version = 1, n_fp = fingerprint.size(), n_tokens = tokens.size(), n_cp = 0;
    const std::vector<llama_token> t(tokens);
    const std::vector<uint8_t> data(n_data, 7);
    std::fwrite(&magic, sizeof(magic), 1, f);
    std::fwrite(&version, sizeof(version), 1, f);
    std::fwrite(&n_fp, sizeof(n_fp), 1, f);
    std::fwrite(fingerprint.data(), 1, n_fp, f);
    std::fwrite(&n_tokens, sizeof(n_tokens), 1, f);
    std::fwrite(t.data(), sizeof(llama_token), t.size(), f);
    std::fwrite(&n_cp, sizeof(n_cp), 1, f);
    std::fwrite(&n_data, sizeof(n_data), 1, f);
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

static std::filesystem::path fresh_dir(const char * name) {
    const auto dir = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

// a directory written within the same clock tick may keep its time; the cache reads it again when its time changes
static void touch_dir(const std::filesystem::path & dir, int seconds) {
    std::filesystem::last_write_time(dir, std::filesystem::file_time_type::clock::now() + std::chrono::seconds(seconds));
}

static size_t count_read_only(const server_prompt_cache & cache) {
    size_t n = 0;
    for (const auto & s : cache.states) {
        n += s.read_only;
    }
    return n;
}

static void test_prefix_dir_takes_only_this_models_files() {
    const auto dir = fresh_dir("inferssd-test-prefix-dir-models");
    // a file of another build's prompt cache for this model, one for another model, one whose model's fingerprint
    // only starts the same
    write_prompt_file((dir / "a.prompt").string(), "prefix-cache 1|this model|build 2|adapter-cache=1",
                      { 1, 2, 3, 4, 5, 6 }, 64);
    write_prompt_file((dir / "b.prompt").string(), "prefix-cache 1|another model|build 1|adapter-cache=1",
                      { 1, 2, 3, 4, 5, 6 }, 64);
    write_prompt_file((dir / "c.prompt").string(), "prefix-cache 1|this model 2|build 1|adapter-cache=1",
                      { 1, 2, 3, 4, 5, 6 }, 64);
    // An old local file has no base-model provenance, even under a release filename.
    const auto legacy = dir / "prefix-cache-09be34f60064f237dd7bc0fc70e78e5deb14c4af36b315ceab9a92e635c8ae75.prompt";
    write_prompt_file(legacy.string(), "prefix-cache 1|this model|build 2",
                      { 1, 2, 3, 4, 5, 6 }, 64);
    write_prompt_file((dir / "lora.prompt").string(),
                      "prefix-cache 1|this model|build 1|adapter-cache=1|lora=0123456789abcdef",
                      { 1, 2, 3, 4, 5, 6 }, 64);

    server_prompt_cache cache(1, 0);
    cache.use_prefix_dir(dir.string(), "prefix-cache 1|this model");
    require(cache.states.size() == 1 && cache.states.front().read_only, "the cache takes the prefix cache of its model");
    require(cache.states.front().n_tokens() == 6 && cache.states.front().file == (dir / "a.prompt").string(),
            "the prefix cache keeps its tokens and its file");
    require(std::filesystem::exists(dir / "b.prompt"), "a prefix cache of another model stays where it is");
    require(std::filesystem::exists(legacy), "an unverified legacy file stays where it is");
    require(cache.size() == 0 && cache.n_tokens() == 0, "a prefix cache does not count against the cache's limits");
    auto adapted = make_prompt({ 1, 2, 3, 4, 5, 6 });
    adapted.adapter_key = "0123456789abcdef";
    require(!server_prompt_cache_testing::contains(cache, adapted, COMMON_CONTEXT_SEQ_RM_TYPE_FULL),
            "a portable base prefix does not satisfy an adapted request");
    std::filesystem::remove_all(dir);
}

static void test_writable_adapter_fingerprint() {
    const auto dir = fresh_dir("inferssd-test-writable-adapter-cache");
    const std::string current = "prefix-cache 1|this model|unknown|adapter-cache=1";
    write_prompt_file((dir / "legacy.prompt").string(), "prefix-cache 1|this model|unknown", { 1, 2, 3 }, 64);
    write_prompt_file((dir / "base.prompt").string(), current, { 1, 2, 3 }, 64);
    write_prompt_file((dir / "lora.prompt").string(), current + "|lora=0123456789abcdef", { 1, 2, 3 }, 64);

    server_prompt_cache cache(4, 0);
    cache.use_dir(dir.string(), current);
    require(cache.states.size() == 2, "writable caches take only their adapter-aware fingerprint");
    require(!std::filesystem::exists(dir / "legacy.prompt"), "legacy writable state has no adapter provenance");
    require(server_prompt_cache_testing::contains(cache, make_prompt({ 1, 2, 3 }), COMMON_CONTEXT_SEQ_RM_TYPE_FULL),
            "a base-model writable state keeps its empty adapter key");
    auto adapted = make_prompt({ 1, 2, 3 });
    adapted.adapter_key = "0123456789abcdef";
    require(server_prompt_cache_testing::contains(cache, adapted, COMMON_CONTEXT_SEQ_RM_TYPE_FULL),
            "an adapted writable state keeps its distinct adapter key");
    std::filesystem::remove_all(dir);
}

static void test_prefix_cache_outlives_the_cache_limits_and_newer_states() {
    const auto dir = fresh_dir("inferssd-test-prefix-dir-limits");
    write_prompt_file((dir / "p.prompt").string(), "fp|build|adapter-cache=1", { 1, 2, 3, 4, 5, 6 }, 64);

    // a cache of 1 MiB in memory, with the prefix cache and two states of its own of 0.75 MiB each
    server_prompt_cache cache(1, 0);
    cache.use_prefix_dir(dir.string(), "fp");
    // a conversation that starts with the prefix: as obsolete as the prefix is restorable from it, and newer
    server_prompt first = make_prompt({ 1, 2, 3, 4, 5, 6, 7, 8 });
    first.n_conversation = 6;
    require(server_prompt_cache_testing::alloc_metadata(cache, first, 768*1024, COMMON_CONTEXT_SEQ_RM_TYPE_PART) != nullptr,
            "the cache saves the conversation");
    require(count_read_only(cache) == 1, "a newer state that contains the prefix cache leaves it in place");
    server_prompt second = make_prompt({ 1, 2, 3, 4, 5, 6, 9, 10, 11 });
    second.n_conversation = 7;
    require(server_prompt_cache_testing::alloc_metadata(cache, second, 768*1024, COMMON_CONTEXT_SEQ_RM_TYPE_PART) != nullptr,
            "the cache saves the second conversation");
    cache.update();
    require(count_read_only(cache) == 1, "the size limit removes the cache's own states, not the prefix cache");
    require(cache.states.size() == 2 && cache.states.back().n_tokens() == 9, "the newest own state stays");
    require(std::filesystem::exists(dir / "p.prompt"), "the prefix cache's file stays");

    // a prompt the prefix cache already holds is not saved again
    require(server_prompt_cache_testing::alloc_metadata(cache, make_prompt({ 1, 2, 3, 4, 5, 6 }), 64, COMMON_CONTEXT_SEQ_RM_TYPE_PART) == nullptr,
            "the prefix cache's own tokens are not saved again");
    std::filesystem::remove_all(dir);
}

static void test_prefix_dir_is_read_again_when_it_changes() {
    const auto dir = fresh_dir("inferssd-test-prefix-dir-changes");
    const auto original = dir / "p.prompt";
    const auto neighbor = dir / "q.prompt";
    server_prompt_cache cache(1, 0);
    cache.use_prefix_dir(dir.string(), "fp");
    require(cache.states.empty(), "an empty directory has no prefix caches");

    // a download finishes while the server runs
    write_prompt_file(original.string(), "fp|build|adapter-cache=1", { 1, 2, 3, 4 }, 16);
    touch_dir(dir, 2);
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(count_read_only(cache) == 1, "a prefix cache put in the directory is found");
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(count_read_only(cache) == 1, "a prefix cache is taken once");

    const auto indexed_original = [&]() -> server_prompt & {
        const auto entry = std::find_if(cache.states.begin(), cache.states.end(), [&](const auto & prompt) {
            return prompt.read_only && prompt.file == original.string();
        });
        require(entry != cache.states.end(), "the surviving prefix stays in the index");
        return *entry;
    };
    // This in-memory field is not serialized. Reloading the same header would reset it.
    indexed_original().n_conversation = 3;
    write_prompt_file(neighbor.string(), "fp|build|adapter-cache=1", { 5, 6, 7 }, 16);
    touch_dir(dir, 4);
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(count_read_only(cache) == 2, "adding a neighbor indexes the new prefix");
    require(indexed_original().n_conversation == 3, "adding a neighbor must not reindex an unchanged prefix");

    std::filesystem::remove(neighbor);
    touch_dir(dir, 6);
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(count_read_only(cache) == 1, "removing a neighbor removes its indexed prefix");
    require(indexed_original().n_conversation == 3, "removing a neighbor must not reindex an unchanged prefix");

    // A download can replace a prefix while retaining the pathname, size, and modification time.
    const auto replacement = dir / "replacement.tmp";
    const auto original_time = std::filesystem::last_write_time(original);
    write_prompt_file(replacement.string(), "fp|build|adapter-cache=1", { 1, 2, 3, 4, 5 }, 16 - sizeof(llama_token));
    require(std::filesystem::file_size(replacement) == std::filesystem::file_size(original),
            "the replacement must have the same file size");
    std::filesystem::last_write_time(replacement, original_time);
    std::filesystem::rename(replacement, original);
    require(std::filesystem::last_write_time(original) == original_time,
            "the replacement must have the same modification time");
    touch_dir(dir, 8);
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(count_read_only(cache) == 1 && indexed_original().n_tokens() == 5 &&
            indexed_original().file_data_size == 16 - sizeof(llama_token) && indexed_original().n_conversation == 0,
            "a same-path replacement must update the indexed tokens and state size");
    std::filesystem::remove_all(dir);
}

static void test_rejected_prefix_versions() {
    const auto dir = fresh_dir("inferssd-test-prefix-dir-rejected");
    const auto prefix = dir / "prefix.prompt";
    const auto neighbor = dir / "neighbor.prompt";
    write_prompt_file(prefix.string(), "fp|old", { 1, 2, 3 }, 16);
    server_prompt_cache cache(1, 0);
    cache.use_prefix_dir(dir.string(), "fp");
    require(cache.states.empty() && server_prompt_cache_testing::unusable_prefixes(cache) == 1,
            "an unpublished legacy header is rejected and its file version remembered");

    write_prompt_file(neighbor.string(), "fp|old", { 4, 5, 6 }, 16);
    touch_dir(dir, 2);
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(cache.states.empty() && server_prompt_cache_testing::unusable_prefixes(cache) == 2,
            "refresh keeps the earlier rejection and records the new one");

    std::filesystem::remove(neighbor);
    const auto replacement = dir / "replacement.tmp";
    write_prompt_file(replacement.string(), "fp|build|adapter-cache=1", { 1, 2, 3 }, 16);
    std::filesystem::rename(replacement, prefix);
    touch_dir(dir, 4);
    server_prompt_cache_testing::scan_prefix_dir(cache);
    require(count_read_only(cache) == 1 && server_prompt_cache_testing::unusable_prefixes(cache) == 0,
            "a valid replacement is admitted and the removed file's rejection is forgotten");
    std::filesystem::remove_all(dir);
}

// a GGUF file with one value and one tensor
static void write_gguf(const std::string & path, const char * value) {
    gguf_context * gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.name", value);
    ggml_init_params params = { /*.mem_size =*/ 1024*1024, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ false };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 16);
    ggml_set_name(t, "w");
    for (int i = 0; i < 16; ++i) {
        ((float *) t->data)[i] = (float) i;
    }
    gguf_add_tensor(gguf, t);
    require(gguf_write_to_file(gguf, path.c_str(), false), "cannot write a GGUF file");
    gguf_free(gguf);
    ggml_free(ctx);
}

static void test_model_identity_ignores_path_and_time() {
    const auto dir = fresh_dir("inferssd-test-model-identity");
    const std::string a = (dir / "model.gguf").string();
    write_gguf(a, "one");
    const std::string id = server_model_files_identity({ a });
    require(!id.empty(), "a GGUF file has an identity");

    // the same file elsewhere, written later
    std::filesystem::create_directories(dir / "elsewhere");
    const std::string b = (dir / "elsewhere" / "model.gguf").string();
    std::filesystem::copy_file(a, b);
    std::filesystem::last_write_time(b, std::filesystem::file_time_type::clock::now() + std::chrono::hours(1));
    require(server_model_files_identity({ b }) == id, "the identity does not depend on the file's place or time");

    // another model
    write_gguf(b, "two");
    require(server_model_files_identity({ b }) != id, "another header is another identity");
    require(server_model_files_identity({ (dir / "missing.gguf").string() }).empty(), "a missing file has no identity");
    std::filesystem::remove_all(dir);
}

int main() {
    test_prefix_dir_takes_only_this_models_files();
    test_writable_adapter_fingerprint();
    test_prefix_cache_outlives_the_cache_limits_and_newer_states();
    test_prefix_dir_is_read_again_when_it_changes();
    test_rejected_prefix_versions();
    test_model_identity_ignores_path_and_time();
    test_prompt_batches_stop_at_shared_prefix_and_end();
    test_prompt_batches_stop_at_kept_prefixes_only();
    test_prompt_points();
    test_batches_end_with_the_ubatch_of_a_point();
    test_checkpoints_at_points_are_kept_by_count();
    test_shared_prefix_serves_a_new_conversation();
    test_a_state_that_cannot_roll_back_is_saved_when_left();
    test_find_checkpoint_before_tail_truncation_pos();
    test_oaicompat_chat_streams_reasoning_delta();
    test_oaicompat_chat_final_contains_reasoning();
    test_full_removal_keeps_exact_shorter_without_checkpoint();
    test_full_removal_reuses_longer_checkpoint_for_shorter_prompt();
    test_full_removal_only_removes_obsolete_shorter_with_checkpoint();

    return 0;
}
