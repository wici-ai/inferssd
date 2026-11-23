// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Whether a GPU's backend streams the weights a context keeps in host memory to it (the SSD provider's transport):
// a discrete GPU of such a backend can run every layer while its memory holds only some of them.
bool llama_ssd_device_streams(ggml_backend_dev_t dev);

// The layers' matrices of a model whose SSD context homes them (llama_model_params::ssd_homes). Each lives in host
// memory, where the model loaded it (the mapped GGUF); a GPU holds a copy of as many layers as its memory allows, and
// the context streams the others to it for each graph (ggml_backend_sched_ssd_hooks), from a copy in the GPU's pinned
// host memory where the host's budget has room for one (a pinned copy crosses the bus at its full rate, a mapped or
// plain one at about 60% of it). Every token uses every layer, so the layers come before the expert caches; the KV cache
// of the context comes before both as it grows.
//
// The layers held are spread over the model: any count of them is a prefix of one order, the layer indices in
// bit-reversed order, so that a streamed layer sits between held ones and its copy overlaps their work. A held layer's
// matrices point at its device copy; given back, at their host memory again: a context's graphs split before either
// read weights that moved.
class llama_dense_homes {
public:
    llama_dense_homes() = default;
    ~llama_dense_homes();
    llama_dense_homes(const llama_dense_homes &) = delete;
    llama_dense_homes & operator=(const llama_dense_homes &) = delete;

    // a matrix of layer il in host memory, whose layer runs on dev
    void add(int il, ggml_backend_dev_t dev, ggml_tensor * matrix);
    // a model file and where the model maps it: a streamed layer without a pinned copy is read from it, around the
    // page cache
    struct file {
        std::string path;
        const void * base;
        size_t bytes;
    };
    void add_file(std::string path, const void * base, size_t bytes) { files_.push_back({std::move(path), base, bytes}); }
    const std::vector<file> & files() const { return files_; }
    // drops the file pages behind host bytes of the model's mapping (nothing for other memory): a layer's mapped pages
    // go once a device or pinned copy holds its matrices, read from the file again if it streams from them later
    void set_drop_pages(std::function<void(const void *, size_t)> drop) { drop_pages = std::move(drop); }
    // the order the layers are held in, once every matrix is added
    void finish();

    bool empty() const { return layers.empty(); }
    ggml_backend_dev_t device() const { return dev; }
    // the bytes of the largest matrix, and of a copy of it at 16 bits
    size_t largest() const;
    size_t largest_f16() const;
    // the bytes the device holds and those it streams
    size_t held_bytes() const;
    size_t streamed_bytes() const;
    // the device bytes the next layer to hold takes, 0 when the device holds every layer
    size_t next_bytes() const;
    // the host bytes of the streamed layers' pinned copies, and those the next copy takes (0: every streamed layer has
    // one, or the GPU has no pinned host memory)
    size_t pinned_bytes() const;
    size_t next_pin_bytes() const;

    // holds the next layers in order while their copies fit in `budget` bytes of the device's memory: the bytes held
    // anew; a layer held gives up its pinned copy. The work queued on the device must not read the matrices meanwhile.
    size_t hold(size_t budget);
    // a pinned copy of the next streamed layer without one, the last in order first (the last to be held again): false
    // when there is none or it cannot be had
    bool pin_next();
    // gives back held layers, the last held first, until `bytes` of device memory are free again: the bytes given
    // back. The work queued on the device must be complete.
    size_t give_back(size_t bytes);

private:
    struct matrix {
        ggml_tensor * tensor;
        ggml_backend_buffer_t host_buffer;
        void * host_data;
    };
    struct layer {
        int il = -1;
        std::vector<matrix> matrices;
        size_t host_bytes = 0;
        ggml_backend_buffer_t copy = nullptr;   // the device copy while held
        ggml_backend_buffer_t pinned = nullptr; // the pinned host copy it streams from
    };
    ggml_backend_dev_t dev = nullptr;
    std::vector<layer> layers;     // by layer index
    std::vector<size_t> order;     // indices into layers, the order they are held in
    size_t n_held = 0;             // a prefix of order
    std::function<void(const void *, size_t)> drop_pages;
    std::vector<file> files_;

    size_t device_bytes(const layer & l) const;
    size_t pin_bytes(const layer & l) const;
    layer * next_to_pin();
};
