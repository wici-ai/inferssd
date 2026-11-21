// SPDX-License-Identifier: Apache-2.0
#include "dense-homes.h"
#include "llama-impl.h"
#include "ggml-ssd/gpu/provider.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

bool llama_ssd_device_streams(ggml_backend_dev_t dev) {
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        return false;
    }
    const auto get = reinterpret_cast<ssd::gpu::get_provider_v1>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_ssd_provider_v1"));
    const auto * provider = get ? get() : nullptr;
    return provider && provider->version == 2 && provider->size == sizeof(*provider) && provider->transport;
}

llama_dense_homes::~llama_dense_homes() {
    for (auto & l : layers) {
        if (l.copy) {
            ggml_backend_buffer_free(l.copy);
        }
        if (l.pinned) {
            ggml_backend_buffer_free(l.pinned);
        }
    }
}

void llama_dense_homes::add(int il, ggml_backend_dev_t layer_dev, ggml_tensor * t) {
    if (il < 0 || !layer_dev || !t || !t->buffer || !ggml_backend_buffer_is_host(t->buffer)) {
        throw std::invalid_argument("a layer's matrix to home is not in host memory");
    }
    if (dev && dev != layer_dev) {
        throw std::invalid_argument("the layers' matrices are homed for one GPU");
    }
    dev = layer_dev;
    if (layers.size() <= size_t(il)) {
        layers.resize(size_t(il) + 1);
    }
    layer & l = layers[size_t(il)];
    l.il = il;
    l.matrices.push_back({t, t->buffer, t->data});
    l.host_bytes += ggml_nbytes(t);
}

void llama_dense_homes::finish() {
    // the layers in bit-reversed index order: every prefix of it spreads over the model
    std::vector<size_t> present;
    for (size_t i = 0; i < layers.size(); ++i) {
        if (!layers[i].matrices.empty()) {
            present.push_back(i);
        }
    }
    unsigned bits = 0;
    while ((size_t(1) << bits) < present.size()) {
        ++bits;
    }
    const auto reversed = [bits](size_t v) {
        size_t r = 0;
        for (unsigned b = 0; b < bits; ++b) {
            r = (r << 1) | ((v >> b) & 1);
        }
        return r;
    };
    std::vector<size_t> rank(present.size());
    for (size_t i = 0; i < present.size(); ++i) {
        rank[i] = i;
    }
    std::stable_sort(rank.begin(), rank.end(), [&](size_t a, size_t b) { return reversed(a) < reversed(b); });
    order.clear();
    for (size_t r : rank) {
        order.push_back(present[r]);
    }
}

size_t llama_dense_homes::largest() const {
    size_t bytes = 0;
    for (const auto & l : layers) {
        for (const auto & m : l.matrices) {
            bytes = std::max(bytes, ggml_nbytes(m.tensor));
        }
    }
    return bytes;
}

size_t llama_dense_homes::largest_f16() const {
    size_t bytes = 0;
    for (const auto & l : layers) {
        for (const auto & m : l.matrices) {
            bytes = std::max(bytes, size_t(ggml_nelements(m.tensor)) * 2);
        }
    }
    return bytes;
}

size_t llama_dense_homes::held_bytes() const {
    size_t bytes = 0;
    for (size_t i = 0; i < n_held; ++i) {
        bytes += layers[order[i]].host_bytes;
    }
    return bytes;
}

size_t llama_dense_homes::streamed_bytes() const {
    size_t bytes = 0;
    for (size_t i = n_held; i < order.size(); ++i) {
        bytes += layers[order[i]].host_bytes;
    }
    return bytes;
}

size_t llama_dense_homes::pinned_bytes() const {
    size_t bytes = 0;
    for (const auto & l : layers) {
        bytes += l.pinned ? ggml_backend_buffer_get_size(l.pinned) : 0;
    }
    return bytes;
}

size_t llama_dense_homes::pin_bytes(const layer & l) const {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_host_buffer_type(dev);
    if (!buft) {
        return 0;
    }
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t bytes = 0;
    for (const auto & m : l.matrices) {
        bytes += GGML_PAD(ggml_nbytes(m.tensor), align);
    }
    return bytes;
}

llama_dense_homes::layer * llama_dense_homes::next_to_pin() {
    for (size_t i = order.size(); i > n_held; --i) {
        layer & l = layers[order[i - 1]];
        if (!l.pinned) {
            return &l;
        }
    }
    return nullptr;
}

size_t llama_dense_homes::next_pin_bytes() const {
    const layer * l = const_cast<llama_dense_homes *>(this)->next_to_pin();
    return l ? pin_bytes(*l) : 0;
}

bool llama_dense_homes::pin_next() {
    layer * l = next_to_pin();
    ggml_backend_buffer_type_t buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    const size_t bytes = l ? pin_bytes(*l) : 0;
    if (!bytes || !buft) {
        return false;
    }
    ggml_backend_buffer_t pinned = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (!pinned) {
        return false;
    }
    ggml_backend_buffer_set_usage(pinned, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    char * base = static_cast<char *>(ggml_backend_buffer_get_base(pinned));
    size_t offset = 0;
    for (auto & m : l->matrices) {
        std::memcpy(base + offset, m.tensor->data, ggml_nbytes(m.tensor));
        m.tensor->buffer = pinned;
        m.tensor->data = base + offset;
        offset += GGML_PAD(ggml_nbytes(m.tensor), align);
        if (drop_pages) {
            drop_pages(m.host_data, ggml_nbytes(m.tensor));
        }
    }
    l->pinned = pinned;
    return true;
}

size_t llama_dense_homes::next_bytes() const {
    return n_held < order.size() ? device_bytes(layers[order[n_held]]) : 0;
}

size_t llama_dense_homes::device_bytes(const layer & l) const {
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t bytes = 0;
    for (const auto & m : l.matrices) {
        bytes += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, m.tensor), align);
    }
    return bytes;
}

size_t llama_dense_homes::hold(size_t budget) {
    if (!dev) {
        return 0;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    const size_t align = ggml_backend_buft_get_alignment(buft);
    size_t held = 0;
    while (n_held < order.size()) {
        layer & l = layers[order[n_held]];
        const size_t bytes = device_bytes(l);
        if (bytes > budget - held) {
            break;
        }
        ggml_backend_buffer_t copy = ggml_backend_buft_alloc_buffer(buft, bytes);
        if (!copy) {
            break;
        }
        // a weights buffer: placing a quantized matrix zeroes its pad, as the model's own buffers do
        ggml_backend_buffer_set_usage(copy, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        char * base = static_cast<char *>(ggml_backend_buffer_get_base(copy));
        size_t offset = 0;
        bool placed = true;
        for (auto & m : l.matrices) {
            ggml_tensor * t = m.tensor;
            const void * from = t->data; // the pinned copy where there is one: it crosses the bus faster
            t->buffer = copy;
            t->data = base + offset;
            if (ggml_backend_buffer_init_tensor(copy, t) != GGML_STATUS_SUCCESS) {
                placed = false;
                break;
            }
            ggml_backend_tensor_set(t, from, 0, ggml_nbytes(t));
            offset += GGML_PAD(ggml_backend_buft_get_alloc_size(buft, t), align);
        }
        if (!placed) {
            // the matrices back where they stream from
            size_t pinned_offset = 0;
            const size_t pinned_align = l.pinned ? ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(l.pinned)) : 1;
            for (auto & m : l.matrices) {
                m.tensor->buffer = l.pinned ? l.pinned : m.host_buffer;
                m.tensor->data = l.pinned ? static_cast<char *>(ggml_backend_buffer_get_base(l.pinned)) + pinned_offset :
                    m.host_data;
                pinned_offset += GGML_PAD(ggml_nbytes(m.tensor), pinned_align);
            }
            ggml_backend_buffer_free(copy);
            break;
        }
        if (l.pinned) {
            ggml_backend_buffer_free(l.pinned);
            l.pinned = nullptr;
        }
        if (drop_pages) {
            for (const auto & m : l.matrices) {
                drop_pages(m.host_data, ggml_nbytes(m.tensor));
            }
        }
        l.copy = copy;
        held += bytes;
        ++n_held;
    }
    return held;
}

size_t llama_dense_homes::give_back(size_t bytes) {
    size_t given = 0;
    while (n_held > 0 && given < bytes) {
        layer & l = layers[order[n_held - 1]];
        given += ggml_backend_buffer_get_size(l.copy);
        for (auto & m : l.matrices) {
            m.tensor->buffer = m.host_buffer;
            m.tensor->data = m.host_data;
        }
        ggml_backend_buffer_free(l.copy);
        l.copy = nullptr;
        --n_held;
    }
    return given;
}
