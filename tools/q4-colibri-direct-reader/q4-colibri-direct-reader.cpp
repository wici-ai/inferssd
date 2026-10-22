#include <nlohmann/json.hpp>
#include <liburing.h>
#include <cuda_runtime.h>
extern "C" {
#include "sha256.h"
}
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>
#include <vector>

using json = nlohmann::ordered_json;

struct options {
    std::string pack, manifest, mode = "layer-batch", role = "gate", experts = "0-31", route_list, json_out;
    std::vector<unsigned> depths = {1, 2, 4, 8, 16, 32};
    int layer = 1, expert = 0;
    uint64_t repeat = 1, control_bytes = 8ULL << 30, control_chunk = 8ULL << 20, seed = 20260714;
};
struct hash_slice { size_t offset, length; std::string sha256, label; };
struct request { uint64_t offset; size_t length; std::string label; std::vector<hash_slice> hashes; };
struct route_key { int layer, expert; bool operator<(const route_key & rhs) const { return std::tie(layer, expert) < std::tie(rhs.layer, rhs.expert); } };

static void usage(const char * program) {
    std::cerr << "usage: " << program << " --pack PATH --manifest PATH --json-out PATH --mode role|slab|layer-batch|control-sequential|control-random [options]\n";
}
static uint64_t number(const std::string & text) {
    size_t used = 0; uint64_t value = std::stoull(text, &used, 0);
    if (used != text.size()) throw std::runtime_error("invalid number: " + text);
    return value;
}
static std::vector<std::string> split(const std::string & text, char delimiter) {
    std::vector<std::string> values; std::stringstream stream(text); std::string value;
    while (std::getline(stream, value, delimiter)) if (!value.empty()) values.push_back(value);
    return values;
}
static options parse(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        std::string argument = argv[index];
        auto value = [&]() { if (++index >= argc) throw std::runtime_error("missing value"); return std::string(argv[index]); };
        if (argument == "--pack") result.pack = value(); else if (argument == "--manifest") result.manifest = value();
        else if (argument == "--mode") result.mode = value(); else if (argument == "--role") result.role = value();
        else if (argument == "--experts") result.experts = value(); else if (argument == "--route-list") result.route_list = value();
        else if (argument == "--json-out") result.json_out = value(); else if (argument == "--layer") result.layer = number(value());
        else if (argument == "--expert") result.expert = number(value()); else if (argument == "--repeat") result.repeat = number(value());
        else if (argument == "--control-bytes") result.control_bytes = number(value()); else if (argument == "--control-chunk") result.control_chunk = number(value());
        else if (argument == "--seed") result.seed = number(value()); else if (argument == "--depths") {
            result.depths.clear(); for (const auto & item : split(value(), ',')) result.depths.push_back(number(item));
        } else if (argument == "--help" || argument == "-h") { usage(argv[0]); std::exit(0); }
        else throw std::runtime_error("unknown argument: " + argument);
    }
    const std::set<std::string> modes = {"role", "slab", "layer-batch", "control-sequential", "control-random"};
    if (result.pack.empty() || result.manifest.empty() || result.json_out.empty() || !modes.count(result.mode) || result.depths.empty() || !result.repeat) throw std::runtime_error("invalid arguments");
    return result;
}
static json load_json(const std::string & path) { std::ifstream input(path); if (!input) throw std::runtime_error("cannot open " + path); json value; input >> value; return value; }
static uint64_t align_up(uint64_t value, uint64_t alignment) { return (value + alignment - 1) / alignment * alignment; }
static uint64_t file_size(const std::string & path) { struct stat status {}; if (stat(path.c_str(), &status)) throw std::runtime_error("stat failed"); return status.st_size; }
static const json & object_for(const json & manifest, int layer, int expert) {
    for (const auto & object : manifest.at("objects")) if (object.at("layer") == layer && object.at("expert") == expert) return object;
    throw std::runtime_error("manifest object missing");
}
static request slab(const json & object) {
    request result{object.at("offset"), object.at("length"), "layer=" + std::to_string(object.at("layer").get<int>()) + ",expert=" + std::to_string(object.at("expert").get<int>()), {}};
    for (const auto & role : object.at("roles")) result.hashes.push_back({role.at("relative_offset"), role.at("length"), role.at("sha256"), result.label + ",role=" + role.at("role").get<std::string>()});
    return result;
}
static request role_request(const json & object, const std::string & role_name) {
    for (const auto & role : object.at("roles")) if (role.at("role") == role_name) {
        std::string label = "layer=" + std::to_string(object.at("layer").get<int>()) + ",expert=" + std::to_string(object.at("expert").get<int>()) + ",role=" + role_name;
        return {role.at("offset"), role.at("length"), label, {{0, role.at("length"), role.at("sha256"), label}}};
    }
    throw std::runtime_error("manifest role missing");
}
static std::vector<int> expert_list(const std::string & text) {
    std::set<int> values;
    for (const auto & item : split(text, ',')) { size_t dash = item.find('-'); int first = number(item.substr(0, dash)); int last = dash == std::string::npos ? first : number(item.substr(dash + 1)); for (int expert = first; expert <= last; ++expert) values.insert(expert); }
    return {values.begin(), values.end()};
}
static std::vector<route_key> route_list(const std::string & path) {
    std::ifstream input(path); if (!input) throw std::runtime_error("route list open failed"); std::stringstream buffer; buffer << input.rdbuf(); std::vector<route_key> values;
    auto append = [&](const json & parsed) { const json * entries = &parsed; if (parsed.is_object() && parsed.contains("routes")) entries = &parsed.at("routes"); if (entries->is_object()) values.push_back({entries->at("layer"), entries->at("expert")}); else for (const auto & entry : *entries) values.push_back({entry.at("layer"), entry.at("expert")}); };
    try { append(json::parse(buffer.str())); } catch (const json::parse_error &) { std::stringstream lines(buffer.str()); std::string line; while (std::getline(lines, line)) if (!line.empty()) append(json::parse(line)); }
    std::set<route_key> unique(values.begin(), values.end()); return {unique.begin(), unique.end()};
}
static std::vector<request> make_requests(const options & settings, const json & manifest, uint64_t bytes, uint64_t alignment) {
    if (settings.mode == "role") return {role_request(object_for(manifest, settings.layer, settings.expert), settings.role)};
    if (settings.mode == "slab") return {slab(object_for(manifest, settings.layer, settings.expert))};
    if (settings.mode == "layer-batch") {
        std::vector<route_key> keys; if (!settings.route_list.empty()) keys = route_list(settings.route_list); else for (int expert : expert_list(settings.experts)) keys.push_back({settings.layer, expert});
        std::vector<request> values; for (const auto & key : keys) values.push_back(slab(object_for(manifest, key.layer, key.expert))); return values;
    }
    uint64_t chunk = align_up(settings.control_chunk, alignment), count = (settings.control_bytes + chunk - 1) / chunk, max_slot = (bytes - chunk) / alignment; std::mt19937_64 generator(settings.seed); std::vector<request> values;
    for (uint64_t index = 0; index < count; ++index) { uint64_t slot = settings.mode == "control-sequential" ? (index * chunk / alignment) % (max_slot + 1) : generator() % (max_slot + 1); values.push_back({slot * alignment, static_cast<size_t>(chunk), settings.mode, {}}); }
    return values;
}
class pinned_buffer { public: explicit pinned_buffer(size_t bytes) : bytes(bytes) { if (cudaHostAlloc(&pointer, bytes, cudaHostAllocDefault) != cudaSuccess) throw std::runtime_error("cudaHostAlloc failed"); } ~pinned_buffer() { if (pointer) cudaFreeHost(pointer); } void * pointer = nullptr; size_t bytes; };
static std::string digest(const unsigned char * data, size_t length) { unsigned char hash[SHA256_DIGEST_SIZE]; sha256_hash(hash, data, length); std::ostringstream output; output << std::hex << std::setfill('0'); for (unsigned char byte : hash) output << std::setw(2) << unsigned(byte); return output.str(); }
static bool pread_exact(int descriptor, void * destination, size_t length, uint64_t offset) { size_t completed = 0; while (completed < length) { ssize_t count = pread(descriptor, static_cast<char *>(destination) + completed, length - completed, offset + completed); if (count < 0 && errno == EINTR) continue; if (count <= 0) return false; completed += count; } return true; }
static json verify(int descriptor, const std::vector<request> & requests, uint64_t alignment) {
    size_t maximum = 0; for (const auto & item : requests) maximum = std::max(maximum, item.length); pinned_buffer buffer(align_up(maximum, alignment)); if (reinterpret_cast<uintptr_t>(buffer.pointer) % alignment) throw std::runtime_error("pinned alignment failure"); uint64_t checks = 0;
    for (const auto & item : requests) if (!item.hashes.empty()) { if (!pread_exact(descriptor, buffer.pointer, item.length, item.offset)) throw std::runtime_error("verification read failed"); for (const auto & slice : item.hashes) { ++checks; if (digest(static_cast<unsigned char *>(buffer.pointer) + slice.offset, slice.length) != slice.sha256) throw std::runtime_error("hash mismatch: " + slice.label); } }
    return {{"checks", checks}, {"failures", 0}};
}
static json benchmark(int descriptor, const std::vector<request> & requests, unsigned depth, uint64_t repeat, uint64_t alignment) {
    size_t slot_size = 0; uint64_t alignment_fallbacks = 0; for (const auto & item : requests) { slot_size = std::max(slot_size, item.length); if (item.offset % alignment || item.length % alignment) ++alignment_fallbacks; } slot_size = align_up(slot_size, alignment); pinned_buffer staging(slot_size * depth); if (reinterpret_cast<uintptr_t>(staging.pointer) % alignment) ++alignment_fallbacks;
    if (alignment_fallbacks) return {{"depth", depth}, {"alignment_fallbacks", alignment_fallbacks}, {"direct_fallbacks", requests.size() * repeat}};
    io_uring ring {}; if (io_uring_queue_init(depth, &ring, 0)) return {{"depth", depth}, {"alignment_fallbacks", 0}, {"direct_fallbacks", requests.size() * repeat}};
    struct slot { uint64_t job = 0; }; std::vector<slot> slots(depth); uint64_t total = requests.size() * repeat, next = 0, completed = 0, inflight = 0, maximum = 0, samples = 0, bytes = 0, failures = 0, submits = 0, waits = 0; long double inflight_sum = 0; double submit_ms = 0, wait_ms = 0;
    auto prepare = [&](unsigned slot_index) { io_uring_sqe * submission = io_uring_get_sqe(&ring); if (!submission) return false; const auto & item = requests[next % requests.size()]; io_uring_prep_read(submission, descriptor, static_cast<char *>(staging.pointer) + slot_size * slot_index, item.length, item.offset); submission->user_data = slot_index + 1; slots[slot_index].job = next++; return true; };
    for (unsigned slot_index = 0; slot_index < depth && next < total; ++slot_index) if (prepare(slot_index)) ++inflight;
    auto submit = [&]() { auto start = std::chrono::steady_clock::now(); int status = io_uring_submit(&ring); submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); ++submits; return status >= 0; };
    uint64_t fallbacks = submit() ? 0 : total; auto start = std::chrono::steady_clock::now();
    while (completed < total && !fallbacks) { maximum = std::max(maximum, inflight); inflight_sum += inflight; ++samples; io_uring_cqe * completion = nullptr; auto wait_start = std::chrono::steady_clock::now(); int status = io_uring_wait_cqe(&ring, &completion); wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wait_start).count(); ++waits; if (status || !completion) { failures += total - completed; break; } unsigned slot_index = completion->user_data - 1; const auto & item = requests[slots[slot_index].job % requests.size()]; if (completion->res == static_cast<int>(item.length)) bytes += item.length; else ++failures; io_uring_cqe_seen(&ring, completion); ++completed; --inflight; if (next < total) { if (!prepare(slot_index) || !submit()) { ++fallbacks; break; } ++inflight; } }
    double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); io_uring_queue_exit(&ring); double gibps = bytes / double(1ULL << 30) / (elapsed_ms / 1000.0);
    return {{"depth", depth}, {"jobs", completed}, {"logical_bytes", bytes}, {"physical_bytes", bytes}, {"pinned_bytes", staging.bytes}, {"read_failures", failures}, {"alignment_fallbacks", 0}, {"direct_fallbacks", fallbacks}, {"submit_calls", submits}, {"wait_calls", waits}, {"completed_depth_max", maximum}, {"completed_depth_avg", samples ? double(inflight_sum / samples) : 0.0}, {"elapsed_ms", elapsed_ms}, {"submit_ms", submit_ms}, {"wait_ms", wait_ms}, {"gib_per_second", gibps}};
}
int main(int argc, char ** argv) {
    try { options settings = parse(argc, argv); json manifest = load_json(settings.manifest); if (manifest.at("format") != "kimi-q4kx-colibri-pack-v1" || manifest.at("version") != 1 || !manifest.at("complete")) throw std::runtime_error("invalid manifest"); uint64_t alignment = manifest.at("alignment"), bytes = file_size(settings.pack); if (bytes != manifest.at("pack_size")) throw std::runtime_error("pack size mismatch"); int descriptor = open(settings.pack.c_str(), O_RDONLY | O_DIRECT); if (descriptor < 0) throw std::runtime_error(std::string("O_DIRECT open failed: ") + strerror(errno)); auto requests = make_requests(settings, manifest, bytes, alignment); json output = {{"format", "kimi-q4kx-direct-reader-v1"}, {"mode", settings.mode}, {"pack", settings.pack}, {"manifest", settings.manifest}, {"request_count", requests.size()}, {"repeat", settings.repeat}, {"hash_verification", verify(descriptor, requests, alignment)}, {"results", json::array()}}; uint64_t reads = 0, alignments = 0, fallbacks = 0; for (unsigned depth : settings.depths) { json result = benchmark(descriptor, requests, depth, settings.repeat, alignment); output["results"].push_back(result); reads += result.value("read_failures", 0ULL); alignments += result.value("alignment_fallbacks", 0ULL); fallbacks += result.value("direct_fallbacks", 0ULL); } close(descriptor); output["read_failures"] = reads; output["alignment_fallbacks"] = alignments; output["direct_fallbacks"] = fallbacks; std::ofstream file(settings.json_out); file << output.dump(2) << '\n'; std::cout << output.dump(2) << '\n'; return reads || alignments || fallbacks ? 2 : 0; }
    catch (const std::exception & error) { std::cerr << "llama-q4-colibri-direct-reader: " << error.what() << '\n'; usage(argv[0]); return 1; }
}
