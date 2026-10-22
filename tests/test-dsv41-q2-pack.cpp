#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>

extern "C" bool ggml_cuda_moe_expert_pack_resolve_direct_shape(
        const char * tensor_name,
        int expert_idx,
        size_t logical_nbytes,
        int64_t n_experts,
        int * fd_direct,
        uint64_t * offset,
        size_t * direct_nbytes);

namespace {

constexpr uint64_t k_align = 4096;

uint64_t align_up(uint64_t value) {
    return (value + k_align - 1) & ~(k_align - 1);
}

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool pwrite_all(int fd, const void * data, size_t bytes, uint64_t offset) {
    const uint8_t * src = static_cast<const uint8_t *>(data);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t wrote = pwrite(fd, src + done, bytes - done, off_t(offset + done));
        if (wrote < 0 && errno == EINTR) {
            continue;
        }
        if (wrote <= 0) {
            return false;
        }
        done += size_t(wrote);
    }
    return true;
}

bool pread_all(int fd, void * data, size_t bytes, uint64_t offset) {
    uint8_t * dst = static_cast<uint8_t *>(data);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t got = pread(fd, dst + done, bytes - done, off_t(offset + done));
        if (got < 0 && errno == EINTR) {
            continue;
        }
        if (got <= 0) {
            return false;
        }
        done += size_t(got);
    }
    return true;
}

#pragma pack(push, 1)
struct pack_header {
    char magic[16];
    uint32_t version;
    uint32_t header_size;
    uint64_t n_entries;
    uint64_t data_start;
};

struct pack_entry {
    char tensor[128];
    int32_t expert_idx;
    uint32_t reserved;
    uint64_t offset;
    uint64_t nbytes;
};
#pragma pack(pop)

static_assert(sizeof(pack_header) == 40, "v1 pack header ABI changed");
static_assert(sizeof(pack_entry) == 152, "v1 pack entry ABI changed");

struct role_case {
    const char * name;
    uint64_t logical;
    uint8_t marker;
    uint64_t offset = 0;
};

class synthetic_pack {
public:
    synthetic_pack() {
        const char * tmp = std::getenv("TMPDIR");
        std::string pattern = std::string(tmp && *tmp ? tmp : "/tmp") + "/dsv41-q2-pack-XXXXXX";
        std::vector<char> path_buf(pattern.begin(), pattern.end());
        path_buf.push_back('\0');
        fd = mkstemp(path_buf.data());
        require(fd >= 0, "mkstemp failed");
        path = path_buf.data();

        pack_header header = {};
        std::memcpy(header.magic, "GGMLMOEPACKv1", 13);
        header.version = 1;
        header.header_size = sizeof(pack_header);
        header.n_entries = roles.size();
        header.data_start = align_up(sizeof(pack_header) + roles.size()*sizeof(pack_entry));

        std::array<pack_entry, 3> entries = {};
        uint64_t cursor = header.data_start;
        for (size_t i = 0; i < roles.size(); ++i) {
            roles[i].offset = cursor;
            std::snprintf(entries[i].tensor, sizeof(entries[i].tensor), "%s", roles[i].name);
            entries[i].expert_idx = 0;
            entries[i].offset = cursor;
            entries[i].nbytes = roles[i].logical;
            cursor += align_up(roles[i].logical);
        }

        require(ftruncate(fd, off_t(cursor)) == 0, "synthetic pack truncate failed");
        require(pwrite_all(fd, &header, sizeof(header), 0), "synthetic pack header write failed");
        require(pwrite_all(fd, entries.data(), sizeof(entries), sizeof(header)), "synthetic pack index write failed");
        for (const role_case & role : roles) {
            const uint8_t first = role.marker;
            const uint8_t last = uint8_t(role.marker ^ 0xffu);
            require(pwrite_all(fd, &first, 1, role.offset), "synthetic role first marker write failed");
            require(pwrite_all(fd, &last, 1, role.offset + role.logical - 1), "synthetic role last marker write failed");
        }
        require(fsync(fd) == 0, "synthetic pack fsync failed");
    }

    ~synthetic_pack() {
        if (fd >= 0) {
            close(fd);
        }
        if (!path.empty()) {
            unlink(path.c_str());
        }
    }

    std::array<role_case, 3> roles = {{
        { "blk.0.ffn_gate_exps.weight", 3041280, 0x31 },
        { "blk.0.ffn_up_exps.weight",   3041280, 0x52 },
        { "blk.0.ffn_down_exps.weight", 3870720, 0x73 },
    }};
    std::string path;
    int fd = -1;
};

void run() {
    synthetic_pack pack;
    require(setenv("GGML_MOE_EXPERT_PACK", pack.path.c_str(), 1) == 0, "set pack environment failed");
    require(setenv("GGML_MOE_IO_BACKEND", "direct", 1) == 0, "set direct backend failed");

    uint64_t payload = 0;
    uint64_t physical = 0;
    for (const role_case & role : pack.roles) {
        int fd_direct = -1;
        uint64_t offset = 0;
        size_t direct_nbytes = 0;
        require(ggml_cuda_moe_expert_pack_resolve_direct_shape(
                    role.name, 0, role.logical, 384, &fd_direct, &offset, &direct_nbytes),
                "Q2 role did not resolve through the real pack index");
        require(fd_direct >= 0, "resolver did not return an O_DIRECT descriptor");
        require(offset == role.offset && offset % k_align == 0, "resolver returned the wrong role offset");
        require(direct_nbytes == align_up(role.logical) && direct_nbytes % k_align == 0,
                "resolver did not expose the padded physical read length");

        void * storage = nullptr;
        require(posix_memalign(&storage, k_align, direct_nbytes) == 0 && storage,
                "aligned synthetic read buffer allocation failed");
        require(reinterpret_cast<uintptr_t>(storage) % k_align == 0,
                "synthetic read buffer is not page aligned");
        require(pread_all(fd_direct, storage, direct_nbytes, offset), "padded O_DIRECT role read failed");
        const uint8_t * bytes = static_cast<const uint8_t *>(storage);
        require(bytes[0] == role.marker && bytes[role.logical - 1] == uint8_t(role.marker ^ 0xffu),
                "logical Q2 payload markers changed during direct read");
        for (size_t i = role.logical; i < direct_nbytes; ++i) {
            require(bytes[i] == 0, "Q2 role padding is not zero-filled");
        }
        std::free(storage);
        payload += role.logical;
        physical += direct_nbytes;
    }

    int fd_direct = -1;
    uint64_t offset = 0;
    size_t direct_nbytes = 0;
    require(!ggml_cuda_moe_expert_pack_resolve_direct_shape(
                pack.roles[0].name, 0, pack.roles[0].logical + 512, 384,
                &fd_direct, &offset, &direct_nbytes),
            "resolver accepted a logical byte count absent from the pack index");
    require(payload == 9953280 && physical == 9957376,
            "synthetic Q2 expert geometry changed");
    std::printf("DSV41_Q2_PACK_LOAD_OK logical=%llu physical=%llu padding=%llu roles=3\n",
        (unsigned long long) payload, (unsigned long long) physical,
        (unsigned long long) (physical - payload));
}

} // namespace

int main() {
    try {
        run();
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "DSV41_Q2_PACK_LOAD_FAILED: %s\n", error.what());
        return 1;
    }
}

#else
int main() {
    return 77;
}
#endif
