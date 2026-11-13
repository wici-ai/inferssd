// SPDX-License-Identifier: Apache-2.0
// Experimental replica routing. Caller owns replica validation and completion lifetime.
#pragma once
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <mutex>
#include <cstdlib>
#include <cstdio>

namespace ssd { namespace io {
struct replica_piece { size_t drive=0, offset=0, bytes=0; };
struct replica_plan { std::array<replica_piece,8> piece{}; size_t count=0; double finish_us=0; };
class replica_router {
    struct drive_state {
        uint64_t pending=0, completed=0, busy_start=0, last_progress=0;
        unsigned requests=0;
        // Neutral initial estimate; replaced by measured demand busy-interval throughput.
        double bytes_per_us=6000.0;
    };
    std::array<drive_state,8> drives{};
    bool tail_aware=std::getenv("INFERSSD_TAIL_AWARE")!=nullptr;
    mutable uint64_t tail_penalties=0;
public:
    ~replica_router() {
        if(tail_aware) std::fprintf(stderr,"[replica_tail] enabled=1 penalized_alternatives=%llu\n",
            (unsigned long long)tail_penalties);
    }
    replica_plan plan(size_t bytes, size_t count, bool split, size_t tie, size_t max_width=4, uint64_t now=0) const {
        if (!bytes || !count || count>8) throw std::runtime_error("invalid replica plan");
        replica_plan best; best.finish_us=std::numeric_limits<double>::infinity();
        for (size_t width : {size_t(1),size_t(2),size_t(4),size_t(8)}) {
            if (width>max_width || width>count || (width>1 && (!split || bytes%4096 || bytes<width*4096))) continue;
            const size_t chunk=width==1?bytes:((bytes/4096+width-1)/width)*4096;
            replica_plan candidate; std::array<bool,8> used{};
            for (size_t pos=0;pos<bytes;pos+=chunk) {
                const size_t n=std::min(chunk,bytes-pos);
                size_t selected=0; double soonest=std::numeric_limits<double>::infinity();
                for(size_t i=0;i<count;++i) {
                    size_t d=(i+tie)%count;
                    if(used[d])continue;
                    double finish=(double(drives[d].pending)+n)/drives[d].bytes_per_us;
                    // Lack of observed progress is a routing signal, not proof of a slow physical SSD.
                    // No duplicated buffers, cancellation, or alteration of completion ownership.
                    if(tail_aware && drives[d].requests && now>drives[d].last_progress) {
                        const double age=double(now-drives[d].last_progress);
                        const double drain=double(drives[d].pending)/drives[d].bytes_per_us;
                        if(age>2000.0 && age>4.0*drain) {
                            finish+=std::min(age,20000.0); ++tail_penalties;
                        }
                    }
                    if(finish<soonest){soonest=finish;selected=d;}
                }
                used[selected]=true;
                candidate.piece[candidate.count++]={selected,pos,n};
                candidate.finish_us=std::max(candidate.finish_us,soonest);
            }
            // Explicit fan-in cost estimate, not a measured hardware constant.
            if(candidate.count>1)candidate.finish_us+=20.0*(candidate.count-1);
            if(candidate.finish_us<best.finish_us)best=candidate;
        }
        return best;
    }
    void queued(size_t d,size_t bytes,uint64_t now) {
        auto &s=drives.at(d);
        if(!s.requests){s.busy_start=now;s.last_progress=now;s.completed=0;}
        s.pending+=bytes;++s.requests;
    }
    void completed(size_t d,size_t expected,size_t actual,uint64_t now,bool ok) {
        auto &s=drives.at(d);
        if(!s.requests || s.pending<expected)throw std::runtime_error("replica accounting underflow");
        s.pending-=expected;--s.requests;
        s.completed+=ok?actual:0; s.last_progress=now;
        if(!s.requests && ok && now>s.busy_start && s.completed) {
            double rate=double(s.completed)/double(now-s.busy_start);
            rate=std::clamp(rate,100.0,20000.0);
            s.bytes_per_us=0.75*s.bytes_per_us+0.25*rate;
        }
    }
    uint64_t pending(size_t d)const{return drives.at(d).pending;}
};
// One sink/context owns this pool. Demand and speculative readers reserve atomically;
// their immutable replica index is shared, not inferred from arbitrary fd numbers.
// A future process-wide pool must additionally key by file identity and device.
class replica_pool {
    mutable std::mutex mutex;
    replica_router router;
public:
    struct counters { std::array<uint64_t,8> demand{}, prefetch{}, requests{}, pending{}; };
private:
    counters counts;
public:
    replica_plan reserve(size_t bytes,size_t n,bool split,size_t tie,uint64_t now,bool speculative,size_t max_width=8) {
        std::lock_guard<std::mutex> lock(mutex);
        auto p=router.plan(bytes,n,split,tie,max_width,now);
        for(size_t j=0;j<p.count;++j){const auto & q=p.piece[j];
            router.queued(q.drive,q.bytes,now);
            (speculative?counts.prefetch:counts.demand)[q.drive]+=q.bytes;
            ++counts.requests[q.drive];
        }
        return p;
    }
    void complete(size_t drive,size_t expected,size_t actual,uint64_t now,bool ok) {
        std::lock_guard<std::mutex> lock(mutex);
        router.completed(drive,expected,actual,now,ok);
    }
    counters snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        auto c=counts;for(size_t i=0;i<8;++i)c.pending[i]=router.pending(i);return c;
    }
};
} }
