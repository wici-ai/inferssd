// SPDX-License-Identifier: Apache-2.0
// llama_ssd_model_facts: a model's facts as the engine would load it, from its headers alone.
// usage: test-ssd-model-facts FIXTURE   (FIXTURE: the split toy model's facts, which the framework's tests read too;
//        INFERSSD_UPDATE_FIXTURES=1 rewrites it)
#include "llama-ssd.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-ssd/store/expert-layout.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void require(bool condition, const std::string & message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct tensor { const char * name; ggml_type type; int64_t ne[3]; };

// One file of the toy model: KV (with the split's keys when SPLIT counts more than one file) and TENSORS.
static std::string write(const std::filesystem::path & path, const std::vector<tensor> & tensors, bool head,
        uint16_t split, uint16_t splits, int32_t total) {
    ggml_init_params params = { 1u << 20, nullptr, false };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_empty(), gguf_free);
    if (head) {
        gguf_set_val_str(gguf.get(), "general.architecture", "qwen3moe");
        gguf_set_val_str(gguf.get(), "general.name", "toy \"quoted\"");
        gguf_set_val_u32(gguf.get(), "qwen3moe.block_count", 3);
        gguf_set_val_u32(gguf.get(), "qwen3moe.nextn_predict_layers", 1);
        gguf_set_val_u32(gguf.get(), "qwen3moe.context_length", 64);
        gguf_set_val_u32(gguf.get(), "qwen3moe.embedding_length", 64);
        gguf_set_val_u32(gguf.get(), "qwen3moe.feed_forward_length", 32);
        gguf_set_val_u32(gguf.get(), "qwen3moe.expert_feed_forward_length", 32);
        gguf_set_val_u32(gguf.get(), "qwen3moe.attention.head_count", 2);
        gguf_set_val_u32(gguf.get(), "qwen3moe.attention.head_count_kv", 1);
        gguf_set_val_u32(gguf.get(), "qwen3moe.expert_count", 4);
        gguf_set_val_u32(gguf.get(), "qwen3moe.expert_used_count", 2);
        gguf_set_val_f32(gguf.get(), "qwen3moe.attention.layer_norm_rms_epsilon", 0.5f);
        const bool pattern[3] = { true, false, true };
        gguf_set_arr_data(gguf.get(), "qwen3moe.test_pattern", GGUF_TYPE_BOOL, pattern, 3);
        gguf_set_val_str(gguf.get(), "tokenizer.ggml.model", "gpt2");
    }
    if (splits > 1) {
        gguf_set_val_u16(gguf.get(), "split.no", split);
        gguf_set_val_u16(gguf.get(), "split.count", splits);
        gguf_set_val_i32(gguf.get(), "split.tensors.count", total);
    }
    for (const auto & in : tensors) {
        ggml_tensor * t = in.ne[2] > 1 ? ggml_new_tensor_3d(ctx.get(), in.type, in.ne[0], in.ne[1], in.ne[2])
                                       : ggml_new_tensor_2d(ctx.get(), in.type, in.ne[0], in.ne[1]);
        ggml_set_name(t, in.name);
        std::memset(t->data, 1, ggml_nbytes(t));
        gguf_add_tensor(gguf.get(), t);
    }
    require(gguf_write_to_file(gguf.get(), path.string().c_str(), false), "GGUF write failed: " + path.string());
    return path.string();
}

// Where each tensor of FILE starts in it.
static uint64_t offset_in(const std::string & file, const char * name) {
    gguf_init_params params = { true, nullptr };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_from_file(file.c_str(), params), gguf_free);
    require(bool(gguf), "GGUF read failed: " + file);
    const int64_t id = gguf_find_tensor(gguf.get(), name);
    require(id >= 0, std::string("no tensor ") + name);
    return gguf_get_data_offset(gguf.get()) + gguf_get_tensor_offset(gguf.get(), id);
}

// The facts of the model PATH with its extra SHARDS (--shard/--ple).
static json facts(const std::string & path, std::string * error = nullptr, const std::vector<std::string> & shards = {}) {
    std::vector<const char *> paths = { path.c_str() };
    for (const auto & shard : shards) { paths.push_back(shard.c_str()); }
    std::string text;
    char message[1024];
    const auto append = [](const char * data, size_t size, void * user) { static_cast<std::string *>(user)->append(data, size); };
    if (!llama_ssd_model_facts(paths.data(), paths.size(), append, &text, message, sizeof message)) {
        require(error != nullptr, std::string("model facts failed: ") + message);
        *error = message;
        return nullptr;
    }
    return json::parse(text);
}

// The largest read of ROLE's experts (each read whole 4 KiB units) and the most one expert's three roles read.
static uint64_t read_of(uint64_t offset, uint64_t bytes) { return ssd_expert_read_of(offset, bytes).bytes; }

// MODEL with its files named in FIXTURE's directory, "<dir>", equals FIXTURE: the facts are one format for both sides.
static void contract(json model, const std::string & fixture) {
    for (auto & file : model["files"]) { file = "<dir>/" + std::filesystem::path(file.get<std::string>()).filename().string(); }
    const char * update = std::getenv("INFERSSD_UPDATE_FIXTURES");
    if (update && std::string(update) == "1") {
        std::ofstream(fixture, std::ios::binary) << model.dump(1) << "\n";
        return;
    }
    std::ifstream stream(fixture, std::ios::binary);
    require(bool(stream), "no fixture " + fixture);
    require(json::parse(stream) == model, "the toy model's facts differ from " + fixture + ": " + model.dump());
}

static void split_model(const std::filesystem::path & directory, const std::string & fixture) {
    const std::vector<tensor> first = {
        { "token_embd.weight", GGML_TYPE_F32, { 64, 16, 1 } },
        { "blk.0.attn_q.weight", GGML_TYPE_F32, { 64, 64, 1 } },
        { "blk.0.ffn_up_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.0.ffn_down_exps.weight", GGML_TYPE_Q8_0, { 32, 64, 4 } },
    };
    const std::vector<tensor> second = {
        { "blk.1.ffn_up_exps.weight", GGML_TYPE_Q4_0, { 64, 32, 4 } },
        { "blk.1.ffn_gate_exps.weight", GGML_TYPE_Q4_0, { 64, 32, 4 } },
        { "blk.1.ffn_down_exps.weight", GGML_TYPE_Q8_0, { 32, 64, 4 } },
        { "blk.2.ffn_up_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.2.ffn_gate_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.2.ffn_down_exps.weight", GGML_TYPE_Q8_0, { 32, 64, 4 } },
        { "blk.2.nextn.eh_proj.weight", GGML_TYPE_F32, { 128, 64, 1 } },
    };
    const int32_t total = int32_t(first.size() + second.size());
    const std::string a = write(directory / "toy-00001-of-00002.gguf", first, true, 0, 2, total);
    const std::string b = write(directory / "toy-00002-of-00002.gguf", second, false, 1, 2, total);
    const json model = facts(a);

    require(model["version"] == 1 && model["architecture"] == "qwen3moe", "version or architecture");
    require(model["layers"] == 2, "the NextN block is part of the model");
    require(model["files"] == json::array({ a, b }), "files: " + model["files"].dump());
    const json & kv = model["metadata"];
    require(kv["general.name"] == "toy \"quoted\"" && kv["qwen3moe.block_count"] == 3 &&
            kv["qwen3moe.attention.layer_norm_rms_epsilon"] == 0.5 &&
            kv["qwen3moe.test_pattern"] == json::array({ true, false, true }), "metadata: " + kv.dump());
    require(!kv.contains("tokenizer.ggml.model") && !kv.contains("split.count"), "metadata outside the model's keys");

    std::vector<std::string> names;
    for (const auto & t : model["tensors"]) { names.push_back(t["name"]); }
    require(names.size() == 8, "tensors: " + std::to_string(names.size()));
    for (const auto & name : names) { require(name.rfind("blk.2.", 0) != 0, "a NextN tensor is listed: " + name); }
    for (const auto & t : model["tensors"]) {
        if (t["name"] == "blk.1.ffn_up_exps.weight") {
            require(t["type"] == "q4_0" && t["shape"] == json::array({ 64, 32, 4 }) && t["file"] == 1 &&
                    t["offset"] == offset_in(b, "blk.1.ffn_up_exps.weight") && t["bytes"] == 4 * 1152,
                "blk.1.ffn_up_exps.weight: " + t.dump());
        }
        if (t["name"] == "token_embd.weight") {
            require(t["shape"] == json::array({ 64, 16 }) && t["file"] == 0 && t["bytes"] == 64 * 16 * 4,
                "token_embd.weight: " + t.dump());
        }
    }

    const json & experts = model["experts"];
    require(experts["count"] == 4 && experts["used"] == 2 && experts["latent"] == 64 && experts["hidden"] == 32,
        "expert geometry: " + experts.dump());
    const json formats = json::parse(R"([
        {"up": {"type": "q8_0", "bytes": 2176}, "gate": {"type": "q8_0", "bytes": 2176}, "down": {"type": "q8_0", "bytes": 2176}},
        {"up": {"type": "q4_0", "bytes": 1152}, "gate": {"type": "q4_0", "bytes": 1152}, "down": {"type": "q8_0", "bytes": 2176}}
    ])");
    require(experts["formats"] == formats, "formats: " + experts["formats"].dump());
    require(experts["layers"] == json::parse("[[0, 0], [1, 1]]"), "layers: " + experts["layers"].dump());

    const struct { const std::string & file; const char * name; uint64_t bytes; } roles[2][3] = {
        { { a, "blk.0.ffn_up_exps.weight", 2176 }, { a, "blk.0.ffn_gate_exps.weight", 2176 },
          { a, "blk.0.ffn_down_exps.weight", 2176 } },
        { { b, "blk.1.ffn_up_exps.weight", 1152 }, { b, "blk.1.ffn_gate_exps.weight", 1152 },
          { b, "blk.1.ffn_down_exps.weight", 2176 } },
    };
    uint64_t largest[3] = {}, most = 0;
    for (const auto & layer : roles) {
        for (uint64_t e = 0; e < 4; ++e) {
            uint64_t read = 0;
            for (int r = 0; r < 3; ++r) {
                const uint64_t bytes = read_of(offset_in(layer[r].file, layer[r].name) + e * layer[r].bytes, layer[r].bytes);
                largest[r] = std::max(largest[r], bytes);
                read += bytes;
            }
            most = std::max(most, read);
        }
    }
    require(experts["read_bytes"] == json({ { "up", largest[0] }, { "gate", largest[1] }, { "down", largest[2] } }),
        "read_bytes: " + experts["read_bytes"].dump());
    require(experts["expert_bytes"] == 3 * 2176 && experts["expert_read_bytes"] == most,
        "largest expert: " + experts.dump());
    contract(model, fixture);
}

// A split model whose last shard holds only the per-layer embedding table, kept elsewhere under another name (as
// ISTA-DASLab's DeepSeek-V4.1 GGUFs keep their Engram table): with the table given as an extra shard the facts list it
// among the files the model loads from and its tensor in it; without it the missing shard is named.
static void ple_table_elsewhere(const std::filesystem::path & directory) {
    const std::vector<tensor> first = {
        { "token_embd.weight", GGML_TYPE_F32, { 64, 16, 1 } },
        { "blk.0.ffn_up_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.0.ffn_gate_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.0.ffn_down_exps.weight", GGML_TYPE_Q8_0, { 32, 64, 4 } },
    };
    const std::vector<tensor> second = {
        { "blk.1.ffn_up_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.1.ffn_gate_exps.weight", GGML_TYPE_Q8_0, { 64, 32, 4 } },
        { "blk.1.ffn_down_exps.weight", GGML_TYPE_Q8_0, { 32, 64, 4 } },
    };
    const int32_t total = int32_t(first.size() + second.size() + 1);
    std::filesystem::create_directories(directory / "model");
    std::filesystem::create_directories(directory / "tables");
    const std::string a = write(directory / "model" / "ple-00001-of-00003.gguf", first, true, 0, 3, total);
    const std::string b = write(directory / "model" / "ple-00002-of-00003.gguf", second, false, 1, 3, total);
    // the table keeps the numbering of the shard it stands for, which an extra shard's numbering does not need
    const std::string table = write(directory / "tables" / "engram-table.gguf",
        { { "per_layer_token_embd.weight", GGML_TYPE_Q8_0, { 64, 96, 1 } } }, false, 2, 3, total);

    const json model = facts(a, nullptr, { table });
    require(model["files"] == json::array({ a, b, table }), "files with the table: " + model["files"].dump());
    bool found = false;
    for (const auto & t : model["tensors"]) {
        if (t["name"] == "per_layer_token_embd.weight") {
            found = t["file"] == 2 && t["offset"] == offset_in(table, "per_layer_token_embd.weight");
        }
    }
    require(found && model["tensors"].size() == size_t(total), "the table's tensor: " + model["tensors"].dump());
    require(model["experts"]["layers"] == json::parse("[[0, 0], [1, 0]]"), "layers: " + model["experts"]["layers"].dump());

    std::string error;
    require(facts(a, &error).is_null() && error.find("ple-00003-of-00003.gguf") != std::string::npos,
        "the missing table shard without the table: " + error);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
        return 2;
    }
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / ("inferssd-model-facts-" + std::to_string(stamp));
    std::filesystem::create_directories(directory);
    int status = 0;
    try {
        split_model(directory, argv[1]);
        ple_table_elsewhere(directory);

        // Without routed experts the layout is null; a file that is not there says so.
        const std::string dense = write(directory / "dense.gguf", { { "token_embd.weight", GGML_TYPE_F32, { 64, 16, 1 } } },
            true, 0, 1, 1);
        require(facts(dense)["experts"].is_null(), "a model without routed experts has a layout");
        std::string error;
        require(facts((directory / "missing.gguf").string(), &error).is_null() && error.find("missing.gguf") != std::string::npos,
            "a missing model: " + error);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "test-model-facts: %s\n", e.what());
        status = 1;
    }
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
    if (status == 0) { std::puts("model facts: ok"); }
    return status;
}
