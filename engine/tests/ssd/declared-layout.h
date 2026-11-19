// SPDX-License-Identifier: Apache-2.0
#pragma once
// The expert layout a GPU tier is declared (ssd_vram_tier_api declare): its formats and which layer has which. A
// tier reads no extent, so none is given.

#include "ggml.h"
#include "../../ggml/src/ggml-ssd/store/expert-layout.h"

#include <array>
#include <cstring>
#include <vector>

struct declared_layout {
    std::vector<ssd_expert_format> formats;
    std::vector<ssd_expert_layer> layers;
    ssd_expert_layout c = {};
    // layer_types: the up, gate and down types of each listed layer, entry for entry with ids (layers 0, 1, ... when
    // ids is empty); every layer has latent, hidden and experts.
    declared_layout(int latent, int hidden, int experts, const std::vector<std::array<int32_t, 3>> & layer_types,
            const std::vector<int32_t> & ids = {}) {
        for (size_t i = 0; i < layer_types.size(); ++i) {
            ssd_expert_format f = {};
            for (int r = 0; r < SSD_EXPERT_ROLES; ++r) {
                const int columns = r == SSD_EXPERT_DOWN ? hidden : latent, rows = r == SSD_EXPERT_DOWN ? latent : hidden;
                const uint64_t row = ggml_row_size(ggml_type(layer_types[i][r]), columns);
                f.role[r] = { layer_types[i][r], columns, rows, row, row * uint64_t(rows) };
            }
            size_t k = 0;
            while (k < formats.size() && std::memcmp(&formats[k], &f, sizeof(f))) { ++k; }
            if (k == formats.size()) { formats.push_back(f); }
            layers.push_back({ ids.empty() ? int32_t(i) : ids[i], int32_t(k), nullptr });
        }
        c = { SSD_EXPERT_LAYOUT_VERSION, uint32_t(sizeof(ssd_expert_layout)), latent, hidden, experts,
              int32_t(formats.size()), formats.data(), int32_t(layers.size()), layers.data(), {} };
    }
    // every listed layer of one format
    static declared_layout uniform(int latent, int hidden, int experts, ggml_type up, ggml_type gate, ggml_type down,
            const std::vector<int32_t> & ids) {
        return declared_layout(latent, hidden, experts,
            std::vector<std::array<int32_t, 3>>(ids.size(), { int32_t(up), int32_t(gate), int32_t(down) }), ids);
    }
    declared_layout(const declared_layout & other) : formats(other.formats), layers(other.layers), c(other.c) { relink(); }
    declared_layout & operator=(const declared_layout & other) {
        formats = other.formats; layers = other.layers; c = other.c; relink(); return *this;
    }
private:
    void relink() { c.formats = formats.data(); c.layers = layers.data(); }
};
