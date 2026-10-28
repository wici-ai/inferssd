#pragma once
// MemAvailable at the kernel's full watermark boost (Linux).
//
// MemAvailable (mm/show_mem.c, si_mem_available) keeps back min(file LRU / 2, wmark_low) of the page cache and
// min(reclaimable / 2, wmark_low) of the reclaimable kernel memory, where wmark_low is the sum over the zones of
// low_wmark_pages(zone) = _watermark[WMARK_LOW] + watermark_boost (include/linux/mmzone.h). The kernel raises a
// zone's boost by a pageblock whenever an allocation falls back to another migratetype (mm/page_alloc.c,
// boost_watermark), up to max(pageblock, high watermark * vm.watermark_boost_factor / 10000), and kswapd takes it
// back once the zone is balanced. So MemAvailable falls and rises by up to that much with nothing allocated or
// freed, and large allocations into a nearly full DRAM are what raise the boost (on a DGX Spark (GB10) the Normal
// zone's largest boost is 437 MiB and MemAvailable moves by about 560 MiB between no boost and the largest).
//
// Whoever fills the DRAM up to a floor measured on MemAvailable must not take what the boost can withdraw: memory
// taken while the boost is low leaves MemAvailable under the floor when the boost comes back. headroom() is how far
// MemAvailable can still fall that way: what si_mem_available keeps back at every zone's largest boost less what it
// keeps back now, from the same /proc/meminfo and /proc/zoneinfo (/proc/zoneinfo prints min, low and high with the
// boost added). 0 when every zone is at its largest boost, when vm.watermark_boost_factor is 0, or when a text is
// missing a field.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>

namespace kimi_memory_boost {

// "key: value kB" in /proc/meminfo, in bytes ("key: value" without a unit as is).
inline bool meminfo_field(const std::string & text, const char * key, uint64_t & value) {
    const size_t length = std::strlen(key);
    for (size_t at = 0; at < text.size();) {
        const size_t end = std::min(text.find('\n', at), text.size());
        if (end - at > length && !text.compare(at, length, key) && text[at + length] == ':') {
            size_t begin = at + length + 1;
            while (begin < end && (text[begin] == ' ' || text[begin] == '\t')) { ++begin; }
            if (begin == end || text[begin] < '0' || text[begin] > '9') { return false; }
            value = 0;
            for (; begin < end && text[begin] >= '0' && text[begin] <= '9'; ++begin) {
                if (value > (UINT64_MAX - 9) / 10) { return false; }
                value = value * 10 + uint64_t(text[begin] - '0');
            }
            if (!text.compare(begin, 3, " kB")) {
                if (value > UINT64_MAX / 1024) { return false; }
                value *= 1024;
            }
            return true;
        }
        at = end + 1;
    }
    return false;
}

// Bytes MemAvailable can still fall by as every zone's watermark boost grows to its largest. meminfo and zoneinfo
// are the texts of /proc/meminfo and /proc/zoneinfo, boost_factor the text of /proc/sys/vm/watermark_boost_factor.
inline uint64_t headroom(const std::string & meminfo, const std::string & zoneinfo, const std::string & boost_factor,
                         uint64_t page_size) {
    uint64_t factor = 0, active = 0, inactive = 0, reclaimable = 0, hugepage = 0, hugepages = 0;
    {
        size_t at = boost_factor.find_first_of("0123456789");
        if (at == std::string::npos) { return 0; }
        for (; at < boost_factor.size() && boost_factor[at] >= '0' && boost_factor[at] <= '9'; ++at) {
            if (factor > (UINT64_MAX - 9) / 10) { return 0; }
            factor = factor * 10 + uint64_t(boost_factor[at] - '0');
        }
    }
    if (!factor || !page_size || !meminfo_field(meminfo, "Active(file)", active) ||
            !meminfo_field(meminfo, "Inactive(file)", inactive)) { return 0; }
    // KReclaimable = the reclaimable slab and NR_KERNEL_MISC_RECLAIMABLE; SReclaimable alone before Linux 4.20
    if (!meminfo_field(meminfo, "KReclaimable", reclaimable) && !meminfo_field(meminfo, "SReclaimable", reclaimable)) {
        return 0;
    }
    // hugetlbfs pools do not count in MemAvailable's terms; the pageblock is a PMD (Hugepagesize) where there is one
    if (meminfo_field(meminfo, "HugePages_Total", hugepages) && hugepages) { return 0; }
    const uint64_t pageblock = meminfo_field(meminfo, "Hugepagesize", hugepage) ? hugepage / page_size : 0;
    // Per zone (its block starts at "Node"): boost, and low and high with the boost added; managed pages.
    uint64_t low_now = 0, low_full = 0;
    uint64_t boost = 0, low = 0, high = 0, managed = 0;
    bool zone = false, has_boost = false;
    auto close_zone = [&] {
        if (!zone) { return; }
        uint64_t largest = boost;
        if (has_boost && low >= boost && high >= boost && managed >= 4 * pageblock) {
            const uint64_t scaled = (high - boost) / 10000 * factor + (high - boost) % 10000 * factor / 10000;
            if (scaled) { largest = std::max(largest, std::max(pageblock, scaled)); }
        }
        low_now += low;
        low_full += low - std::min(low, boost) + largest;
        boost = low = high = managed = 0;
        has_boost = false;
    };
    for (size_t at = 0; at < zoneinfo.size();) {
        const size_t end = std::min(zoneinfo.find('\n', at), zoneinfo.size());
        size_t begin = at;
        while (begin < end && zoneinfo[begin] == ' ') { ++begin; }
        size_t word = begin;
        while (word < end && zoneinfo[word] != ' ') { ++word; }
        const std::string key = zoneinfo.substr(begin, word - begin);
        if (!zoneinfo.compare(at, 4, "Node")) {
            close_zone();
            zone = true;
        } else if (zone && (key == "boost" || key == "low" || key == "high" || key == "managed")) {
            while (word < end && zoneinfo[word] == ' ') { ++word; }
            uint64_t value = 0;
            for (; word < end && zoneinfo[word] >= '0' && zoneinfo[word] <= '9'; ++word) { value = value * 10 + uint64_t(zoneinfo[word] - '0'); }
            if (key == "boost") { boost = value; has_boost = true; }
            else if (key == "low") { low = value; }
            else if (key == "high") { high = value; }
            else { managed = value; }
        }
        at = end + 1;
    }
    close_zone();
    if (low_full <= low_now) { return 0; }
    const uint64_t file = (active + inactive) / page_size, kernel = reclaimable / page_size;
    auto kept = [&](uint64_t wmark_low) { return std::min(file / 2, wmark_low) + std::min(kernel / 2, wmark_low); };
    return (kept(low_full) - kept(low_now)) * page_size;
}

} // namespace kimi_memory_boost
