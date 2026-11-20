// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Whether a GPU's backend streams the weights a context keeps in host memory to it (the SSD provider's transport):
// a discrete GPU of such a backend can run every layer while its memory holds only some of them.
bool llama_ssd_device_streams(ggml_backend_dev_t dev);

// The layers' matrices of a model whose SSD context homes them (llama_model_params::ssd_homes). Each lives in host
// memory, where the model loaded it; a GPU holds a copy of as many layers as its memory allows, and the context streams
// the others to it for each graph (ggml_backend_sched_ssd_hooks). Every token uses every layer, so the layers come
// before the expert caches; the KV cache of the context comes before both as it grows.
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

    // holds the next layers in order while their copies fit in `budget` bytes of the device's memory: the bytes held
    // anew. The work queued on the device must not read the matrices meanwhile.
    size_t hold(size_t budget);
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
        ggml_backend_buffer_t copy = nullptr; // the device copy while held
    };
    ggml_backend_dev_t dev = nullptr;
    std::vector<layer> layers;     // by layer index
    std::vector<size_t> order;     // indices into layers, the order they are held in
    size_t n_held = 0;             // a prefix of order

    size_t device_bytes(const layer & l) const;
};
