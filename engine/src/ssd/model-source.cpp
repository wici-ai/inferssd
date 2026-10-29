// SPDX-License-Identifier: MIT
#include "model-source.h"
#include "llama-hparams.h"
#include <algorithm>
#include <regex>
#include <stdexcept>
#include <cstdio>
extern "C" {
#include "../../examples/gguf-hash/deps/sha256/sha256.h"
}
#ifndef _WIN32
#include <sys/stat.h>
#endif

void llama_coalesce_ranges(std::vector<llama_file_range> & ranges) {
    if (ranges.empty()) {
        return;
    }

    std::sort(ranges.begin(), ranges.end(), [] (const llama_file_range & a, const llama_file_range & b) {
        return a.first < b.first || (a.first == b.first && a.last < b.last);
    });

    std::vector<llama_file_range> merged;
    merged.reserve(ranges.size());
    for (const auto & range : ranges) {
        if (range.first >= range.last) {
            continue;
        }
        if (merged.empty() || range.first > merged.back().last) {
            merged.push_back(range);
        } else {
            merged.back().last = std::max(merged.back().last, range.last);
        }
    }
    ranges = std::move(merged);
}

bool llama_is_expert_tensor_name(const std::string & name, uint32_t & layer) {
    int parsed_layer = -1;
    if (sscanf(name.c_str(), "blk.%d.", &parsed_layer) != 1 || parsed_layer < 0) {
        return false;
    }

    static const std::regex expert_re(R"(^blk\.([0-9]+)\.ffn_(up|down|gate|gate_up)_(ch)?exps\.)");
    if (!std::regex_search(name, expert_re)) {
        return false;
    }

    layer = (uint32_t) parsed_layer;
    return true;
}

void llama_model_loader::build_expert_tensor_index(const llama_hparams & hparams) {
    expert_tensor_index = {};

    if (hparams.n_layer == 0 || hparams.n_expert == 0 || files.empty()) {
        return;
    }

    expert_tensor_index.file_ranges.resize(files.size());

    size_t deferred_bytes = 0;
    for (const auto & it : weights_map) {
        const std::string & name = it.first;
        const llama_tensor_weight & weight = it.second;

        uint32_t layer = 0;
        if (!llama_is_expert_tensor_name(name, layer)) {
            continue;
        }
        if (layer >= hparams.n_layer) {
            throw std::runtime_error(format("expert tensor '%s' has invalid layer index %u", name.c_str(), layer));
        }
        if (weight.idx >= expert_tensor_index.file_ranges.size()) {
            throw std::runtime_error(format("expert tensor '%s' has invalid file index %u", name.c_str(), (unsigned) weight.idx));
        }

        const size_t tensor_bytes = ggml_nbytes(weight.tensor);
        deferred_bytes += tensor_bytes;
        expert_tensor_index.file_ranges[weight.idx].push_back({ weight.offs, weight.offs + tensor_bytes });
    }

    for (auto & ranges : expert_tensor_index.file_ranges) {
        llama_coalesce_ranges(ranges);
    }

    expert_tensor_index.deferred_bytes = deferred_bytes;
    expert_tensor_index.dense_bytes = n_bytes > deferred_bytes ? n_bytes - deferred_bytes : 0;
}

bool llama_model_loader::should_defer_expert_mmaps() const {
    return defer_experts && use_mmap && !no_alloc && !expert_tensor_index.empty();
}

void llama_model_loader::drop_mmap_expert_pages() const {
    if (!use_mmap || mappings.empty() || expert_tensor_index.file_ranges.empty()) {
        return;
    }

    const size_t n_range_sets = std::min(mappings.size(), expert_tensor_index.file_ranges.size());
    for (size_t idx = 0; idx < n_range_sets; ++idx) {
        for (const auto & range : expert_tensor_index.file_ranges[idx]) {
            mappings[idx]->dontneed_fragment(range.first, range.last);
        }
    }
}

namespace llama_ssd {
namespace {
std::vector<std::string> requested(const ssd::settings::snapshot & settings) {
    const auto * e = settings.find("model.content_manifest");
    const auto * v = e ? std::get_if<std::vector<std::string>>(&e->data) : nullptr;
    return v ? *v : std::vector<std::string>{};
}
std::string stamp(int fd) {
#ifdef _WIN32
    (void) fd;
    return {}; // no native stable descriptor receipt on this platform yet
#else
    struct stat s;
    if (fstat(fd, &s)) { return {}; }
#ifdef __APPLE__
    const auto m = s.st_mtimespec, c = s.st_ctimespec;
#else
    const auto m = s.st_mtim, c = s.st_ctim;
#endif
    return std::to_string(s.st_dev) + ":" + std::to_string(s.st_ino) + ":" +
        std::to_string(s.st_size) + ":" + std::to_string(m.tv_sec) + ":" +
        std::to_string(m.tv_nsec) + ":" + std::to_string(c.tv_sec) + ":" + std::to_string(c.tv_nsec);
#endif
}
std::string finish(sha256_t & hash) {
    unsigned char bytes[32];
    sha256_final(&hash, bytes);
    std::string out;
    for (auto b : bytes) { out += "0123456789abcdef"[b >> 4]; out += "0123456789abcdef"[b & 15]; }
    return out;
}
}
void content_identity::verify(const ssd::settings::snapshot & settings, const llama_files & loaded) {
    digest.clear(); manifest.clear(); experts.clear(); files.clear(); stamps.clear();
    const auto claims = requested(settings);
    if (claims.empty() || claims.size() != loaded.size() || claims.size() > 100000) { return; }
    for (const auto & c : claims) {
        if (c.size() != 64 || c.find_first_not_of("0123456789abcdef") != std::string::npos) { return; }
    }
    sha256_t combined; sha256_init(&combined);
    // Fixed-size reads, one full sequential pass over each loader descriptor.
    std::vector<unsigned char> buffer(1024 * 1024);
    for (size_t i = 0; i < loaded.size(); ++i) {
        auto & file = *loaded[i];
        if (file.has_direct_io()) { return; } // unaligned streaming is unsupported
        const auto before = stamp(file.file_id());
        if (before.empty()) { return; }
        const auto position = file.tell();
        const auto size = file.size();
        sha256_t hash; sha256_init(&hash);
        try {
            file.seek(0, SEEK_SET);
            for (size_t offset = 0; offset < size;) {
                const auto n = std::min(buffer.size(), size - offset);
                file.read_raw(buffer.data(), n);
                sha256_update(&hash, buffer.data(), n);
                offset += n;
            }
            file.seek(position, SEEK_SET);
        } catch (...) {
            file.seek(position, SEEK_SET);
            throw; // I/O failure cannot be reported as successful admission
        }
        if (stamp(file.file_id()) != before || file.size() != size || finish(hash) != claims[i]) { return; }
        stamps.push_back(before);
        const auto line = claims[i] + "\n";
        sha256_update(&combined, reinterpret_cast<const unsigned char *>(line.data()), line.size());
    }
    manifest = claims;
    digest = "sha256:" + finish(combined);
}
std::string content_identity::for_context(const ssd::settings::snapshot & settings,
                                         const ssd::store::source_view & sources) const {
    if (digest.empty() || requested(settings) != manifest) { return {}; }
    if (files.size() != stamps.size()) { return {}; }
    for (size_t i = 0; i < files.size(); ++i) {
        if (stamp(files[i]->file_id()) != stamps[i]) { return {}; }
    }
    // External expert indices must resolve to the same admitted GGUF byte ranges.
    // A packed/replaced source needs separate admission; never assume equivalence.
    if (settings.sources() && !settings.sources()->empty() && !sources.resolve) { return {}; }
    if (sources.resolve) {
#ifdef _WIN32
        return {};
#else
        if (files.empty() || experts.empty()) { return {}; }
        for (const auto & e : experts) {
            if (e.shard >= files.size()) { return {}; }
            for (int64_t expert = 0; expert < e.count; ++expert) {
                int fd; uint64_t offset; size_t bytes;
                if (!sources.resolve(sources.userdata, e.name.c_str(), int(expert), e.bytes, e.count,
                                     &fd, &offset, &bytes) || stamp(fd) != stamps[e.shard] ||
                    offset != e.offset + size_t(expert) * e.bytes || bytes != e.bytes) { return {}; }
            }
        }
#endif
    }
    return digest;
}
}
