#include "prototype.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace original {
using namespace c0;
static parser_state g_expert_pack;
#include "original-parsers.inc"

static void clear() {
    for (auto & source : g_expert_pack.sources) {
        if (source.file) { std::fclose(source.file); }
        if (source.fd_direct >= 0) { ::close(source.fd_direct); }
    }
    g_expert_pack.sources.clear();
    g_expert_pack.entries.clear();
    g_expert_pack.inited = false;
    g_expert_pack.enabled = false;
    g_expert_pack.page_cache_sources = 0;
}
static void finalize(std::vector<expert_pack_entry> & entries, bool replace_duplicates) {
#include "finalize.inc"
    g_expert_pack.enabled = true;
    g_expert_pack.inited = true;
}
static bool open(const description & desc) {
    clear();
    g_expert_pack.io_backend = desc.io_backend;
    std::vector<expert_pack_entry> entries;
    for (const auto & path : desc.packs) {
        if (!expert_pack_load_source(path.c_str(), int32_t(g_expert_pack.sources.size()), entries, false)) { return false; }
    }
    std::unordered_map<std::string, int32_t> aliases;
    for (const auto & path : desc.aliases) {
        if (!expert_pack_load_gguf_alias_tsv(path.c_str(), entries, aliases)) { return false; }
    }
    for (const auto & path : desc.page_cache_packs) {
        if (!expert_pack_load_source(path.c_str(), int32_t(g_expert_pack.sources.size()), entries, true)) { return false; }
    }
    if (entries.empty()) { return false; }
    finalize(entries, desc.replace_duplicates || !desc.page_cache_packs.empty());
    if (!g_expert_pack.enabled) { return false; }
    g_expert_pack.entries = std::move(entries);
    return true;
}
static bool locate(const char * name, int expert, size_t bytes, extent & result) {
    if (!g_expert_pack.enabled || !name || !name[0] || expert < 0 || !bytes) { return false; }
    size_t lo = 0;
    size_t hi = g_expert_pack.entries.size();
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const int cmp = expert_pack_entry_cmp(g_expert_pack.entries[mid], name, expert, bytes);
        if (cmp < 0) { lo = mid + 1; } else { hi = mid; }
    }
    if (lo == g_expert_pack.entries.size() ||
        expert_pack_entry_cmp(g_expert_pack.entries[lo], name, expert, bytes)) { return false; }
    const auto & item = g_expert_pack.entries[lo];
    if (item.source_idx < 0 || size_t(item.source_idx) >= g_expert_pack.sources.size()) { return false; }
    const auto & source = g_expert_pack.sources[size_t(item.source_idx)];
    if (item.offset > source.file_size || bytes > source.file_size - item.offset) { return false; }
    result = {item.source_idx, item.offset, item.nbytes, item.alias_source, identity(source)};
    return true;
}
} // namespace original

namespace fs = std::filesystem;
static unsigned checks = 0;
static void check(bool ok, const char * message) {
    ++checks;
    if (!ok) { throw std::runtime_error(message); }
}
struct temporary_directory {
    fs::path path;
    temporary_directory() {
        char name[] = "/tmp/ssdash-c0-store-XXXXXX";
        char * value = ::mkdtemp(name);
        if (!value) { throw std::runtime_error("mkdtemp failed"); }
        path = value;
    }
    ~temporary_directory() { fs::remove_all(path); }
};
struct row {
    std::string tensor;
    int32_t expert;
    uint64_t offset;
    uint64_t bytes;
};
static void write_file(const fs::path & path, size_t bytes, unsigned char value) {
    std::ofstream out(path, std::ios::binary);
    std::vector<char> data(bytes, char(value));
    out.write(data.data(), std::streamsize(data.size()));
}
static void pack(const fs::path & path, const std::vector<row> & rows, uint32_t header_size = 40) {
    std::ofstream out(path, std::ios::binary);
    char magic[16] = "GGMLMOEPACKv1";
    uint32_t version = 1;
    const uint64_t count = rows.size(), start = 4096;
    auto field = [&](const auto & value) { out.write(reinterpret_cast<const char *>(&value), sizeof(value)); };
    out.write(magic, sizeof(magic));
    field(version); field(header_size); field(count); field(start);
    for (const auto & item : rows) {
        char name[128] = {};
        std::memcpy(name, item.tensor.data(), std::min(item.tensor.size(), sizeof(name)));
        const uint32_t reserved = 0;
        out.write(name, sizeof(name)); field(item.expert); field(reserved); field(item.offset); field(item.bytes);
    }
    const std::streamoff end = 16384;
    if (out.tellp() < end) { out.seekp(end - 1); out.put('x'); }
}
static void tsv(const fs::path & path, const std::string & text) {
    std::ofstream(path) << text;
}
static std::string header() { return "tensor\texpert_idx\tnbytes\tsource_path\toffset\n"; }
static std::string alias_row(const std::string & tensor, int expert, uint64_t bytes, const fs::path & source, uint64_t offset) {
    return tensor + "\t" + std::to_string(expert) + "\t" + std::to_string(bytes) + "\t" + source.string() + "\t" + std::to_string(offset) + "\n";
}
static bool same(const c0::expert_pack_entry & a, const c0::expert_pack_entry & b) {
    return !std::strcmp(a.tensor, b.tensor) && a.expert_idx == b.expert_idx && a.source_idx == b.source_idx &&
        a.offset == b.offset && a.nbytes == b.nbytes && a.alias_source == b.alias_source;
}
static void query_pair(const c0::expert_index & index, const row & query) {
    c0::extent a, b;
    const bool old_ok = original::locate(query.tensor.c_str(), query.expert, query.bytes, a);
    const bool new_ok = index.locate(query.tensor.c_str(), query.expert, query.bytes, b);
    check(old_ok == new_ok, "range result differs from original");
    if (old_ok) {
        check(a.source_id == b.source_id && a.offset == b.offset && a.logical_bytes == b.logical_bytes &&
              a.alias == b.alias && a.file == b.file, "range fields differ from original");
    }
}
static std::unique_ptr<c0::expert_index> compare(const c0::description & desc, bool expected_ok) {
    const bool old_ok = original::open(desc);
    auto index = c0::expert_index::open(desc);
    check(old_ok == expected_ok, "fixture expectation differs from original");
    check(bool(index) == old_ok, "store load differs from original");
    if (!index) { original::clear(); return {}; }
    const auto & old_entries = original::g_expert_pack.entries;
    const auto & new_entries = index->entries_for_test();
    check(old_entries.size() == new_entries.size(), "entry count differs from original");
    check(index->source_count() == original::g_expert_pack.sources.size(), "source count differs from original");
    for (size_t i = 0; i < old_entries.size(); ++i) {
        check(same(old_entries[i], new_entries[i]), "index differs from original");
        query_pair(*index, {old_entries[i].tensor, old_entries[i].expert_idx, 0, old_entries[i].nbytes});
        query_pair(*index, {old_entries[i].tensor, old_entries[i].expert_idx, 0, old_entries[i].nbytes + 1});
    }
    for (size_t i = 0; i < index->source_count(); ++i) {
        check(index->source_identity_for_test(i) == c0::identity(original::g_expert_pack.sources[i]), "file identity differs");
    }
    query_pair(*index, {"missing.weight", 0, 0, 16});
    return index;
}
static int open_descriptor_count() {
    int count = 0;
    for (int fd = 0; fd < 1024; ++fd) { if (::fcntl(fd, F_GETFD) >= 0) { ++count; } }
    return count;
}

int main() {
    try {
        temporary_directory tmp;
        const auto p1 = tmp.path / "one.pack", p2 = tmp.path / "two.pack";
        const auto a1 = tmp.path / "one.tsv", a2 = tmp.path / "two.tsv";
        const auto data = tmp.path / "weights.gguf";
        const std::string name = "blk.0.ffn_up_exps.weight";
        write_file(data, 16384, 0x31);
        pack(p1, {{name, 3, 4096, 64}, {name, 3, 8192, 128}});
        auto base = compare({{p1.string()}, {}, {}, false}, true);
        c0::extent found;
        check(base->locate(name.c_str(), 3, 128, found) && found.offset == 8192, "logical bytes must distinguish keys");
        check(!base->locate(name.c_str(), 3, 63, found), "nonmatching logical size must fail");

        pack(p2, {{name, 3, 12288, 64}});
        original::clear();
        const int before = open_descriptor_count();
        compare({{p1.string(), p2.string()}, {}, {}, false}, false);
        check(open_descriptor_count() == before, "failed duplicate load leaked source handles");
        auto overlaid = compare({{p1.string(), p2.string()}, {}, {}, true}, true);
        check(overlaid->locate(name.c_str(), 3, 64, found) && found.source_id == 1 && found.offset == 12288,
              "later pack must win explicit replacement");
        auto page = compare({{p1.string()}, {}, {p2.string()}, false}, true);
        check(page->locate(name.c_str(), 3, 64, found) && found.source_id == 1,
              "page-cache pack must force replacement");

        tsv(a1, header() + alias_row(name, 3, 64, data, 4097) + alias_row(name, 4, 64, data, 8193));
        tsv(a2, header() + alias_row(name, 4, 64, data, 12001) + alias_row(name, 5, 64, data, 12003));
        auto aliases = compare({{p1.string()}, {a1.string(), a2.string()}, {}, true}, true);
        check(aliases->locate(name.c_str(), 3, 64, found) && !found.alias && found.source_id == 0,
              "pack entry must precede duplicate alias");
        check(aliases->locate(name.c_str(), 4, 64, found) && found.alias && found.offset == 8193,
              "first alias must win even when replacement enabled");
        check(aliases->source_count() == 2, "same alias path must reuse one source across TSVs");
        check(aliases->locate(name.c_str(), 5, 64, found) && found.offset % 4096 == 3811,
              "alias must retain unaligned logical offset");

        // Preserve permissive legacy integer parsing unchanged.
        tsv(a1, "source_path\tmodel_offset\texpert\tnbytes\ttensor\r\n" + data.string() +
            "\t4097tail\t3tail\t64tail\t" + name + "\r\n");
        auto permissive = compare({{}, {a1.string()}, {}, false}, true);
        check(permissive->locate(name.c_str(), 3, 64, found) && found.offset == 4097,
              "legacy numeric suffix parsing changed");
        tsv(a1, "tensor\texpert\tnbytes\tsource_path\toffset_bytes\n" + alias_row(name, 3, 64, data, 4097));
        compare({{}, {a1.string()}, {}, false}, true);
        pack(p2, {{name, 3, 4096, 64}}, 64);
        compare({{p2.string()}, {}, {}, false}, true); // Index still starts at byte 40.

        tsv(a1, header() + alias_row(name, 3, 64, data, 16360));
        auto tail = compare({{}, {a1.string()}, {}, false}, true);
        check(!tail->locate(name.c_str(), 3, 64, found), "logical range beyond EOF must fail at lookup");
        tsv(a1, header() + alias_row(name, 3, 64, data, UINT64_MAX));
        auto overflow = compare({{}, {a1.string()}, {}, false}, true);
        check(!overflow->locate(name.c_str(), 3, 64, found), "offset overflow must fail without wrapping");
        tsv(a1, "tensor\texpert\tnbytes\tsource_path\n");
        compare({{}, {a1.string()}, {}, false}, false);
        tsv(a1, header() + "too\tfew\tcolumns\n");
        compare({{}, {a1.string()}, {}, false}, false);
        tsv(a1, header() + alias_row(name, -1, 64, data, 4096));
        compare({{}, {a1.string()}, {}, false}, false);
        write_file(p2, 24, 0);
        compare({{p2.string()}, {}, {}, false}, false);
        auto overwrite = [&](std::streamoff offset, const auto & value) {
            std::fstream out(p2, std::ios::in | std::ios::out | std::ios::binary);
            out.seekp(offset);
            out.write(reinterpret_cast<const char *>(&value), sizeof(value));
        };
        pack(p2, {{name, 3, 4096, 64}});
        overwrite(0, char('X'));
        compare({{p2.string()}, {}, {}, false}, false);
        pack(p2, {{name, 3, 4096, 64}});
        overwrite(16, uint32_t(2));
        compare({{p2.string()}, {}, {}, false}, false);
        pack(p2, {{name, 3, 4096, 64}});
        overwrite(24, uint64_t(10000001));
        compare({{p2.string()}, {}, {}, false}, false);
        pack(p2, {{std::string(128, 'n'), -1, 4096, 0}});
        auto permissive_pack = compare({{p2.string()}, {}, {}, false}, true);
        check(std::strlen(permissive_pack->entries_for_test()[0].tensor) == 127,
              "legacy v1 tensor-name truncation changed");
        check(!permissive_pack->locate(std::string(127, 'n').c_str(), -1, 0, found),
              "invalid logical query should fail without tightening pack loader");
        pack(p2, {{name, 3, 4096, 64}});
        fs::resize_file(p2, 60);
        compare({{p2.string()}, {}, {}, false}, false);

        // Generated 2-layer, 16-expert, 3-role corpus. These are format ranges,
        // not valid quantized tensors or a model golden.
        std::vector<row> corpus;
        std::string manifest = header();
        uint64_t offset = 4097;
        for (int layer = 0; layer < 2; ++layer) {
            for (int expert = 0; expert < 16; ++expert) {
                for (const char * role : {"up", "gate", "down"}) {
                    const std::string key = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
                    corpus.push_back({key, expert, offset, 64});
                    manifest += alias_row(key, expert, 64, data, offset);
                    offset += 65;
                }
            }
        }
        tsv(a1, manifest);
        auto generated = compare({{}, {a1.string()}, {}, false}, true);
        check(generated->entries_for_test().size() == 96, "generated corpus lost entries");
        for (const auto & item : corpus) {
            check(generated->locate(item.tensor.c_str(), item.expert, item.bytes, found) && found.offset == item.offset,
                  "generated corpus differs from specification");
        }
        original::clear();

        tsv(a1, header() + alias_row(name, 3, 64, data, 4097));
        auto retained = c0::expert_index::open({{}, {a1.string()}, {}, false});
        check(bool(retained) && retained->locate(name.c_str(), 3, 64, found), "retained source missing");
        const auto prior_identity = found.file;
        const int retained_fd = retained->borrowed_fd(found.source_id);
        fs::rename(data, tmp.path / "renamed.gguf");
        write_file(data, 16384, 0x72);
        auto replacement = c0::expert_index::open({{}, {a1.string()}, {}, false});
        c0::extent newer;
        check(replacement->locate(name.c_str(), 3, 64, newer) && !(newer.file == prior_identity),
              "path replacement must not reuse stale file identity");
        check(retained->locate(name.c_str(), 3, 64, found) && found.file == prior_identity,
              "existing index must retain opened file identity");
        unsigned char value = 0;
        check(::pread(retained_fd, &value, 1, 4097) == 1 && value == 0x31,
              "borrowed handle must refer to original inode");
        retained.reset();
        errno = 0;
        check(::fcntl(retained_fd, F_GETFD) == -1 && errno == EBADF, "index destruction must close owned handle");
        check(replacement->locate(name.c_str(), 3, 64, newer), "destroying one index must not clear another");
        std::cout << "PASS: " << checks << " checks; original/new differential and fixture expectations\n";
        std::cout << "Coverage: pack-v1, alias columns/order, duplicates/overlays, logical bounds, source identity/lifetime\n";
        std::cout << "NOT COVERED: production CPU sink, O_DIRECT consumer, model logits, CUDA parity\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
