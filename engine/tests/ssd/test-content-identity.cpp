// SPDX-License-Identifier: Apache-2.0
#include "ssd/model-source.h"
#include "ssd/options.h"
#include "ggml-ssd/settings/model-options.h"
#include "llama-model-loader.h"
#include "common.h"
#include "inferssd/options.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <thread>
#include <unistd.h>

using namespace ssd::settings;
static snapshot config(const std::vector<std::string> & hashes) {
    std::string text = "inferssd-settings 1\nmodel.content_manifest=[";
    for (size_t i = 0; i < hashes.size(); ++i) { text += (i ? "," : "") + ssd::json::quote(hashes[i]); }
    auto parsed = parse(text + "]\n", "fixture"); assert(parsed); return parsed.value;
}
int main(int argc, char ** argv) {
    if (argc == 3) {
        // Production settings -> common model/context params -> native loader's
        // actual shard selection -> same admission and resolver used by context.
        common_params common;
        common.ssd_enabled = true;
        common.ssd_settings_file = argv[1];
        common_ssd_prepare(common);
        auto mp = common_model_params_to_llama(common);
        auto cp = common_context_params_to_llama(common);
        assert(mp.ssd_options && mp.ssd_options == cp.ssd_options);
        std::vector<std::string> splits;
        llama_model_loader loader(nullptr, nullptr, nullptr, argv[2], splits, nullptr,
            true, false, false, false, false, nullptr, nullptr);
        llama_ssd::content_identity admitted;
        admitted.verify(mp.ssd_options->settings, loader.files);
        admitted.files = std::move(loader.files);
        llama_ssd_context context(*cp.ssd_options);
        auto fact = [&](const char * key) {
            const auto k = gguf_find_key(loader.metadata, key);
            return k < 0 ? std::string() : gguf_kv_to_str(loader.metadata, k);
        };
        model_identity id {fact("general.uuid"), fact("general.architecture"), fact("general.file_type"),
            fact("general.quantization_version"), "Test GPU", admitted.for_context(context.settings, context.sources)};
        auto result = resolve_model(context.settings, id);
        assert(result);
        std::cout << "{\"loader_content\":" << ssd::json::quote(admitted.digest) << ","
                  << result.value.json(id).substr(1) << '\n';
        return 0;
    }
    char dir[] = "/tmp/ssd-content-XXXXXX"; assert(mkdtemp(dir));
    const std::string path = std::string(dir) + "/model.gguf";
    { std::ofstream f(path); f << "abc"; }
    const std::string abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    auto settings = config({abc});
    llama_files files; files.emplace_back(new llama_file(path.c_str(), "rb"));
    files[0]->seek(1, SEEK_SET);
    llama_ssd::content_identity bound;
    bound.verify(settings, files);
    assert(!bound.digest.empty() && files[0]->tell() == 1);
    bound.files = std::move(files);
    const auto identity = bound.for_context(settings, {});
    assert(identity == bound.digest);
    assert(bound.for_context(config({}), {}).empty());
    assert(bound.for_context(config({std::string(64, '0')}), {}).empty());
    for (const auto & claims : {std::vector<std::string>{}, {abc, abc}, {std::string(64, '0')}, {"bad"}}) {
        llama_ssd::content_identity bad; bad.verify(config(claims), bound.files);
        assert(bad.digest.empty());
    }
    const std::string other_path = std::string(dir) + "/other.gguf";
    { std::ofstream f(other_path); f << "abc"; }
    llama_files other_files;
    other_files.emplace_back(new llama_file(other_path.c_str(), "rb"));
    llama_ssd::content_identity other;
    other.verify(settings, other_files);
    other.files = std::move(other_files);
    assert(other.for_context(settings, {}) == identity);
    const std::string third_path = std::string(dir) + "/different.gguf";
    { std::ofstream f(third_path); f << "def"; }
    auto third_settings = config({"cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34"});
    llama_files third_files;
    third_files.emplace_back(new llama_file(third_path.c_str(), "rb"));
    llama_ssd::content_identity different;
    different.verify(third_settings, third_files);
    different.files = std::move(third_files);
    assert(!different.for_context(third_settings, {}).empty());
    assert(different.for_context(settings, {}).empty());
    const auto rules = parse("inferssd-settings 1\nmodel.optimizations=[" + ssd::json::quote(
        "content|on|" + identity + "|deepseek41|38|2|Test GPU|1|0") + "]\n", "fixture");
    assert(rules);
    model_identity a {"", "deepseek41", "38", "2", "Test GPU", identity};
    auto b = a; b.content = different.for_context(third_settings, {});
    assert(resolve_model(rules.value, a).value.engram_prefetch);
    assert(!resolve_model(rules.value, b).value.engram_prefetch);
    std::thread first([&] { for (int i=0; i<20; ++i) assert(bound.for_context(settings, {}) == identity); });
    std::thread second([&] { for (int i=0; i<20; ++i) {
        assert(different.for_context(settings, {}).empty());
        assert(!resolve_model(rules.value, b).value.engram_prefetch);
    } });
    first.join(); second.join();
    bound.experts.push_back({"blk.0.ffn_up_exps.weight", 0, 0, 3, 1});
    ssd::store::source_view source {&bound, [](void * p, const char *, int, size_t, int64_t,
                                            int * fd, uint64_t * off, size_t * bytes) noexcept {
        *fd = static_cast<llama_ssd::content_identity *>(p)->files[0]->file_id();
        *off = 0; *bytes = 3; return true;
    }, nullptr};
    assert(bound.for_context(settings, source) == identity);
    auto wrong_source = source;
    wrong_source.userdata = &other; // identical bytes in a different unbound source inode
    assert(bound.for_context(settings, wrong_source).empty());
    bound.experts[0].offset = 1;
    assert(bound.for_context(settings, source).empty());
    bound.experts[0].offset = 0;
    { std::ofstream f(path); f << "xyz"; }
    assert(bound.for_context(settings, {}).empty()); // descriptor mutation invalidates receipt
    unlink(path.c_str()); unlink(other_path.c_str()); unlink(third_path.c_str()); rmdir(dir);
}
