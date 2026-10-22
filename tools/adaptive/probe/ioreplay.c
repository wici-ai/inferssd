// ioreplay: the drive probe's reader.  It replays a read plan (a fio version 2 iolog, as `expert_source.py iolog`
// writes it: whole role ranges widened to 4 KiB) the way the engine's CPU sink reads experts: O_DIRECT, one read per
// range, through io_uring driven by raw syscalls (no liburing), QD reads in flight, the next one issued as soon as
// one completes (fio's replay_no_stall).  A read that ends at the end of its file may come back short; that is the
// whole range (the last expert of a file is widened past the end), as in the engine.
//
// It prints a report with the fields of fio's JSON that probe.sh and device_profile.py read:
//   {"jobs": [{"jobname", "read": {"io_bytes", "bw_bytes", "iops", "runtime" (ms), "total_ios",
//              "lat_ns": {"min", "max", "mean", "stddev", "percentile": {"50.000000", "90.000000", "99.000000"}}}}]}
// lat_ns is each read's time from its submission to its completion.
//
// usage: ioreplay IOLOG QD [JOBNAME]
// build: cc -O2 -o ioreplay ioreplay.c -lm
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#define PAGE 4096

struct file { char * path; int fd; off_t size; };
struct op { int file; uint64_t off; uint32_t len; };

static struct file * files; static int nfiles;
static struct op * ops; static size_t nops, capops;

static void die(const char * what, const char * detail) {
    fprintf(stderr, "ioreplay: %s%s%s%s%s\n", what, detail ? ": " : "", detail ? detail : "",
            errno ? ": " : "", errno ? strerror(errno) : "");
    exit(1);
}

static int file_index(const char * path) {
    for (int i = 0; i < nfiles; ++i) {
        if (!strcmp(files[i].path, path)) { return i; }
    }
    files = realloc(files, (nfiles + 1) * sizeof *files);
    if (!files) { die("out of memory", NULL); }
    files[nfiles].path = strdup(path);
    errno = 0;
    files[nfiles].fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
    if (files[nfiles].fd < 0) { die("cannot open with O_DIRECT (the engine reads experts only this way)", path); }
    struct stat st;
    if (fstat(files[nfiles].fd, &st)) { die("fstat", path); }
    files[nfiles].size = st.st_size;
    return nfiles++;
}

// splits the last blank-separated token off LINE (which keeps the rest), or returns NULL
static char * last_token(char * line) {
    char * sp = strrchr(line, ' ');
    if (!sp) { return NULL; }
    *sp = 0;
    return sp + 1;
}

static int number(const char * s, unsigned long long * v) {
    char * end;
    errno = 0;
    *v = strtoull(s, &end, 10);
    return *s && !*end && !errno;
}

// A line is "PATH ACTION" or "PATH read OFFSET LENGTH".  PATH may hold blanks, so the line is split from the right.
static void load(const char * iolog) {
    FILE * f = fopen(iolog, "r");
    errno = 0;
    if (!f) { die("cannot read", iolog); }
    char line[8192], copy[8192];
    if (!fgets(line, sizeof line, f) || strncmp(line, "fio version 2 iolog", 19)) { die("not a fio version 2 iolog", iolog); }
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        memcpy(copy, line, sizeof copy);
        char * action = last_token(line);
        if (!action || !*line) { continue; }
        if (!strcmp(action, "add") || !strcmp(action, "open")) { file_index(line); continue; }
        unsigned long long off = 0, len = 0;
        char * len_s = action, * off_s = last_token(line), * read_s = off_s ? last_token(line) : NULL;
        if (!read_s || strcmp(read_s, "read") || !*line) { continue; }   // close, wait: nothing to replay
        if (!number(off_s, &off) || !number(len_s, &len)) { errno = 0; die("bad read line", copy); }
        if (off % PAGE || len % PAGE || !len || len > (1u << 31)) { errno = 0; die("read not 4 KiB aligned", copy); }
        if (nops == capops) {
            capops = capops ? 2 * capops : 1024;
            ops = realloc(ops, capops * sizeof *ops);
            if (!ops) { die("out of memory", NULL); }
        }
        ops[nops++] = (struct op){ file_index(line), off, (uint32_t) len };
    }
    fclose(f);
    if (!nops) { errno = 0; die("no reads in", iolog); }
}

// one ring, single submitter, as the CPU sink's sink_ring
static int ring_fd;
static unsigned * sq_tail, * sq_mask, * sq_array, * cq_head, * cq_tail, * cq_mask;
static struct io_uring_sqe * sqes;
static struct io_uring_cqe * cqes;

static void ring_init(unsigned depth) {
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    errno = 0;
    ring_fd = (int) syscall(__NR_io_uring_setup, depth, &p);
    if (ring_fd < 0) { die("io_uring_setup (check /proc/sys/kernel/io_uring_disabled)", NULL); }
    size_t sq_size = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_size = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) { sq_size = cq_size = sq_size > cq_size ? sq_size : cq_size; }
    char * sq = mmap(NULL, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
    if (sq == MAP_FAILED) { die("mmap SQ ring", NULL); }
    char * cq = (p.features & IORING_FEAT_SINGLE_MMAP) ? sq :
        mmap(NULL, cq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
    if (cq == MAP_FAILED) { die("mmap CQ ring", NULL); }
    sqes = mmap(NULL, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                ring_fd, IORING_OFF_SQES);
    if (sqes == MAP_FAILED) { die("mmap SQEs", NULL); }
    sq_tail = (unsigned *) (sq + p.sq_off.tail); sq_mask = (unsigned *) (sq + p.sq_off.ring_mask);
    sq_array = (unsigned *) (sq + p.sq_off.array);
    cq_head = (unsigned *) (cq + p.cq_off.head); cq_tail = (unsigned *) (cq + p.cq_off.tail);
    cq_mask = (unsigned *) (cq + p.cq_off.ring_mask);
    cqes = (struct io_uring_cqe *) (cq + p.cq_off.cqes);
}

static uint64_t now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t) t.tv_sec * 1000000000ull + (uint64_t) t.tv_nsec;
}

// S as a JSON string
static void json_string(const char * s) {
    putchar('"');
    for (; *s; ++s) {
        const unsigned char c = (unsigned char) *s;
        if (c == '"' || c == '\\') { putchar('\\'); putchar(c); }
        else if (c < 0x20) { printf("\\u%04x", c); }
        else { putchar(c); }
    }
    putchar('"');
}

static int cmp_u64(const void * a, const void * b) {
    uint64_t x = *(const uint64_t *) a, y = *(const uint64_t *) b;
    return x < y ? -1 : x > y;
}

int main(int argc, char ** argv) {
    if (argc < 3 || argc > 4 || atoi(argv[2]) < 1) {
        fprintf(stderr, "usage: ioreplay IOLOG QD [JOBNAME]\n");
        return 2;
    }
    const unsigned qd = (unsigned) atoi(argv[2]);
    load(argv[1]);
    uint32_t maxlen = 0;
    for (size_t i = 0; i < nops; ++i) { if (ops[i].len > maxlen) { maxlen = ops[i].len; } }

    // QD buffers, touched before the clock starts so that page faults are not timed
    const unsigned slots = qd < nops ? qd : (unsigned) nops;
    char ** buf = calloc(slots, sizeof *buf);
    unsigned * free_slots = calloc(slots, sizeof *free_slots);
    uint64_t * issued = calloc(slots, sizeof *issued);
    size_t * op_of = calloc(slots, sizeof *op_of);
    uint64_t * lat = calloc(nops, sizeof *lat);
    if (!buf || !free_slots || !issued || !op_of || !lat) { die("out of memory", NULL); }
    for (unsigned s = 0; s < slots; ++s) {
        if (posix_memalign((void **) &buf[s], PAGE, maxlen)) { die("aligned allocation", NULL); }
        memset(buf[s], 0, maxlen);
        free_slots[s] = s;
    }
    unsigned nfree = slots;
    ring_init(slots);

    size_t next = 0, done = 0;
    uint64_t bytes = 0;
    const uint64_t t0 = now_ns();
    while (done < nops) {
        unsigned queued = 0;
        const uint64_t t_issue = now_ns();
        while (nfree && next < nops) {
            const unsigned s = free_slots[--nfree];
            const unsigned tail = *sq_tail, index = tail & *sq_mask;
            struct io_uring_sqe * e = &sqes[index];
            memset(e, 0, sizeof *e);
            e->opcode = IORING_OP_READ;
            e->fd = files[ops[next].file].fd;
            e->off = ops[next].off;
            e->addr = (uint64_t) (uintptr_t) buf[s];
            e->len = ops[next].len;
            e->user_data = s;
            sq_array[index] = index;
            __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
            issued[s] = t_issue;
            op_of[s] = next++;
            ++queued;
        }
        // submit what was queued and wait for at least one completion, in one call
        for (;;) {
            errno = 0;
            const int r = (int) syscall(__NR_io_uring_enter, ring_fd, queued, 1, IORING_ENTER_GETEVENTS, NULL, 0);
            if (r < 0 && errno == EINTR) { continue; }
            if (r < 0) { die("io_uring_enter", NULL); }
            queued -= (unsigned) r;
            if (!queued) { break; }
        }
        unsigned head = *cq_head;
        while (head != __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) {
            const struct io_uring_cqe * c = &cqes[head & *cq_mask];
            const unsigned s = (unsigned) c->user_data;
            const struct op * o = &ops[op_of[s]];
            const struct file * f = &files[o->file];
            // a read reaching the end of the file returns only the bytes that exist
            const uint64_t want = o->off + o->len <= (uint64_t) f->size ? o->len :
                                  (uint64_t) f->size > o->off ? (uint64_t) f->size - o->off : 0;
            if (c->res < 0 || (uint64_t) c->res != want) {
                errno = c->res < 0 ? -c->res : 0;
                fprintf(stderr, "ioreplay: read of %s at %llu (%u bytes) returned %d\n", f->path,
                        (unsigned long long) o->off, o->len, c->res);
                return 1;
            }
            const uint64_t t = now_ns();
            lat[done++] = t - issued[s];
            bytes += (uint64_t) c->res;
            free_slots[nfree++] = s;
            ++head;
        }
        __atomic_store_n(cq_head, head, __ATOMIC_RELEASE);
    }
    const uint64_t elapsed = now_ns() - t0;

    double sum = 0, sq = 0;
    for (size_t i = 0; i < nops; ++i) { sum += (double) lat[i]; }
    const double mean = sum / (double) nops;
    for (size_t i = 0; i < nops; ++i) { sq += ((double) lat[i] - mean) * ((double) lat[i] - mean); }
    qsort(lat, nops, sizeof *lat, cmp_u64);
#define PCT(p) lat[(size_t) ((p) / 100.0 * (double) (nops - 1) + 0.5)]
    const double runtime_ms = (double) elapsed / 1e6;
    printf("{\n \"ioreplay\": {\"iolog\": ");
    json_string(argv[1]);
    printf(", \"qd\": %u, \"files\": %d, \"engine\": \"io_uring raw syscalls, O_DIRECT\"},\n \"jobs\": [{\"jobname\": ",
           qd, nfiles);
    json_string(argc == 4 ? argv[3] : "ioreplay");
    printf(", \"read\": {\"io_bytes\": %llu, \"bw_bytes\": %llu, \"iops\": %.3f, \"runtime\": %llu, \"total_ios\": %zu,\n",
           (unsigned long long) bytes, (unsigned long long) ((double) bytes * 1e9 / (double) elapsed),
           (double) nops * 1e9 / (double) elapsed, (unsigned long long) (runtime_ms + 0.5), nops);
    printf("   \"lat_ns\": {\"min\": %llu, \"max\": %llu, \"mean\": %.1f, \"stddev\": %.1f, \"percentile\": "
           "{\"50.000000\": %llu, \"90.000000\": %llu, \"99.000000\": %llu}}}}]\n}\n",
           (unsigned long long) lat[0], (unsigned long long) lat[nops - 1], mean, sqrt(sq / (double) nops),
           (unsigned long long) PCT(50), (unsigned long long) PCT(90), (unsigned long long) PCT(99));
    return 0;
}
