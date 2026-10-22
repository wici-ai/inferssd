#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/types.h>

typedef void * (*mmap_fn)(void *, size_t, int, int, int, off_t);
typedef int (*posix_madvise_fn)(void *, size_t, int);

void * mmap(void * addr, size_t length, int prot, int flags, int fd, off_t offset) {
    static mmap_fn real_mmap;

    if (real_mmap == NULL) {
        real_mmap = (mmap_fn) dlsym(RTLD_NEXT, "mmap");
        if (real_mmap == NULL) {
            errno = ENOSYS;
            return MAP_FAILED;
        }
    }

    return real_mmap(addr, length, prot, flags & ~MAP_POPULATE, fd, offset);
}

int posix_madvise(void * addr, size_t length, int advice) {
    static posix_madvise_fn real_posix_madvise;

    if (advice == POSIX_MADV_WILLNEED) {
        return 0;
    }

    if (real_posix_madvise == NULL) {
        real_posix_madvise = (posix_madvise_fn) dlsym(RTLD_NEXT, "posix_madvise");
        if (real_posix_madvise == NULL) {
            return ENOSYS;
        }
    }

    return real_posix_madvise(addr, length, advice);
}
