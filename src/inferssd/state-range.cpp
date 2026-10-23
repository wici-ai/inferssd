#include "llama-kv-cache.h"
#include "llama-io.h"

#include <vector>

bool llama_kv_cache::seq_has_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1) const {
    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    if (p1 < p0) {
        return true;
    }

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    std::vector<bool> seen(p1 - p0 + 1, false);
    size_t n_seen = 0;

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
            continue;
        }
        const llama_pos pos = cells.pos_get(i);
        if (pos >= p0 && pos <= p1 && !seen[pos - p0]) {
            seen[pos - p0] = true;
            ++n_seen;
        }
    }

    return n_seen == seen.size();
}

void llama_kv_cache::state_write_range(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos p0, llama_pos p1) const {
    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.is_empty(i) && (seq_id == -1 || cells.seq_has(i, seq_id)) &&
                    cells.pos_get(i) >= p0 && cells.pos_get(i) <= p1) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

