#include "vram-slot-table.h"
#include "vram-compute-api.h"
#include "kimi-prefill-cost.h"
#include "kimi-vram-location-api.h"
#include "kimi-mxfp4-batch.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <algorithm>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <sys/stat.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <map>
#include <set>

// The production DSO provides this CUDA adapter.  Keep it weak because several
// lifecycle unit tests include this file directly and exercise only the generic
// tier with no format-compute object linked into the test executable.
extern "C" const vram_format_compute_api * kimi_vram_q2_format_compute_get_api() __attribute__((weak));

namespace {
void cuda_check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
uint64_t env_size(const char * name,uint64_t fallback,uint64_t maximum){
    const char * s=std::getenv(name);if(!s || !*s)return fallback;
    char * end=nullptr;errno=0;uint64_t value=std::strtoull(s,&end,10);
    if(errno || s==end || *end || *s=='-' || value>maximum)throw std::runtime_error(std::string("invalid ")+name);
    return value;
}
double env_count(){
    const char * s=std::getenv("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT");if(!s || !*s)return 0;
    char * end=nullptr;errno=0;double x=std::strtod(s,&end);
    if(errno || end==s || *end || !std::isfinite(x) || x<0 || x>4)throw std::runtime_error("invalid GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT");
    return x;
}
double env_half_life(){
    const char * s=std::getenv("GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE");
    if(!s || !*s)return 2;
    char * end=nullptr;errno=0;double x=std::strtod(s,&end);
    if(errno || end==s || *end || !std::isfinite(x) || (x!=0 && (x<1 || x>64)))
        throw std::runtime_error("invalid GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE");
    return x==0?2:x; // zero leaves the original two-token half-life in use
}
// ggml's own device-memory query (MemAvailable on a unified-memory GPU) and its run-time reserve. Looked up
// at run time like the OOM reclaimer: lifecycle tests link this file without the CUDA backend, and may replace
// these functions with a model of the memory before the first tier asks.
struct ggml_memory_query {
    using get_fn=void (*)(int,size_t *,size_t *);
    using unified_fn=bool (*)(int);
    using reserve_fn=void (*)(int,size_t);
    get_fn get=reinterpret_cast<get_fn>(dlsym(RTLD_DEFAULT,"ggml_backend_cuda_get_device_memory"));
    unified_fn unified=reinterpret_cast<unified_fn>(dlsym(RTLD_DEFAULT,"ggml_backend_cuda_device_is_unified"));
    reserve_fn set_reserve=reinterpret_cast<reserve_fn>(dlsym(RTLD_DEFAULT,"ggml_cuda_set_memory_reserve"));
    static ggml_memory_query & instance(){static ggml_memory_query q;return q;}
};
using clock_type=std::chrono::steady_clock;
double micros(clock_type::time_point t){return std::chrono::duration<double,std::micro>(clock_type::now()-t).count();}
struct tier {
    static constexpr size_t elastic_block_slots=16;
    static constexpr uint64_t automatic_budget=UINT64_MAX; // GGML_KIMI_CPU_SINK_VRAM_BYTES=auto
    using reclaimer_callback=int (*)(void *,size_t,size_t *);
    using reclaimer_register_fn=bool (*)(int,reclaimer_callback,void *);
    using reclaimer_unregister_fn=bool (*)(int,reclaimer_callback,void *);
    // ggml-cuda's prefill seed registry (ggml_cuda_moe_seed_register): the scheduler offers each routed expert
    // tensor it copied to the device for a graph prefill pass, with the pass's routing counts.
    using seed_callback=int (*)(void *,const char *,const void *,size_t,int64_t,const int32_t *,void *,void **);
    using seed_register_fn=bool (*)(int,seed_callback,void *);
    uint64_t budget,epoch=0,token=0,capacity_bytes=0,safety_bytes;
    uint64_t gpu_hits=0,cpu_requests=0,uploads=0,upload_bytes=0,borrowed_uploads=0;
    double upload_us=0,upload_wait_us=0,compute_us=0;
    bool active=false,have_token=false,async_upload=false;
    bool memory_trace=false;
    // Opt-in: keep the arena, slot heat and residency through prefill for the same model/context epoch.
    bool keep_across_requests=false;uint64_t arena_epoch=0;
    // Opt-in timing split (GGML_KIMI_CPU_SINK_VRAM_TIMING=1): launch-to-worker handoff and the sink's wait in finish.
    bool timing=false;uint64_t launches=0;double handoff_us=0,finish_drain_us=0;clock_type::time_point launch_time{};
    uint64_t memory_samples=0,memory_free_min=UINT64_MAX,memory_free_max=0;
    uint64_t ready_sequence=0,live_ready_generation=0;
    int device=-1,latent=0,hidden=0;
    // Unified memory (DGX Spark, Jetson): the tier's blocks are system memory. It keeps the host-memory ledger's
    // floor (set_host_floor) and the RAM cache's uncommitted slots (set_host_demand) of MemAvailable free and
    // publishes the same reserve to ggml, so an allocation there reclaims tier blocks first.
    bool unified=false;size_t host_floor=0,host_demand=0;
    uint64_t host_reclaims=0,host_reclaim_refusals=0,live_reclaim_refusals=0;
    uint64_t phase_decode_tokens=0; // decode tokens begun since the last decode phase_complete
    // Decode jobs computed from host memory (kimi_vram_host_api): in place where the device reads host pages through
    // its page tables (ATS); on a discrete GPU after a copy from page-locked memory into device memory, the expert's
    // admission reservation or a host scratch slot. device[] is what the engine reads; start/end time the copy.
    bool host_readable=false,layer_input_ready=false,host_queued=false;
    struct host_task { kimi_vram_host_job job; const void * device[3]; int scratch; cudaEvent_t start,end; };
    std::vector<host_task> host_pending;std::vector<kimi_vram_host_job> host_done;std::vector<float> host_down;
    double resident_us_per_expert=0,host_us_per_expert=0;uint64_t host_jobs=0,host_batches=0;
    static constexpr size_t host_scratch_slots=2;
    char * host_scratch=nullptr;std::vector<uint8_t> host_scratch_busy;bool host_scratch_wanted=false,host_scratch_failed=false;
    cudaStream_t host_copy_stream=nullptr;std::vector<cudaEvent_t> host_events; // free timing events, two a copied job
    uint64_t host_copies=0,host_reserved_copies=0,host_refused=0;double host_copy_bytes=0,host_copy_us=0;
    // The first page-locked range, until a page-locked copy is measured: a decode token boundary copies it into a
    // scratch slot (asked for at the registration) and the next one reads the copy's time (start_host_probe,
    // finish_host_probe). Without it nothing moves: the plan moves only at a measured copy rate.
    const void * host_probe=nullptr;size_t host_probe_bytes=0,host_probe_copied=0;bool host_probe_taken=false;
    cudaEvent_t host_probe_start=nullptr,host_probe_end=nullptr;
    // The measured lines the sink plans with: (experts, us) of each engine call of residents and of host jobs, and
    // (MB, us) of each page-locked copy into device memory (host jobs' and deferred admissions'), the last 64 each.
    struct samples { std::deque<std::pair<double,double>> s; void add(double x,double y){s.emplace_back(x,y);if(s.size()>64)s.pop_front();} };
    samples resident_calls,host_calls,locked_copies;
    // Admission threshold from measured costs (kimi_vram_policy_api); MIN_COUNT is only the starting value.
    double miss_us=0,upload_us_per_expert=0,last_min_count=0;bool adaptive_admission=false;uint64_t admission_updates=0;
    double last_cost_hits=0,last_expected[3]={};uint64_t last_evictions=0;
    size_t role_bytes[3]={},stride=0;
    char * arena=nullptr;cudaStream_t upload_stream=nullptr;
    struct arena_block { char * data=nullptr;size_t first_slot=0,slot_count=0; };
    std::vector<arena_block> blocks;
    std::vector<char *> slot_locations;
    std::mutex lifecycle_mutex;
    bool elastic_requested=false,elastic_active=false,elastic_demand_reclaim=false,reclaimer_registered=false,reclaimer_armed=false,poisoned=false;
    bool decode_growth_done=false;
    size_t base_slots=0,max_slots=0,elastic_safety_blocks=0,elastic_safety_mib=0,elastic_max_slots=0;
    uint64_t phase_generation=0,grow_events=0,proactive_blocks=0,reactive_blocks=0,base_blocks_reclaimed=0;
    uint64_t phase_reactive_start=0,phase_base_reclaimed_start=0,phase_reclaim_requests=0;
    size_t phase_slots_before=0,phase_free_before=0;
    uint64_t malloc_calls=0,free_calls=0,fail_malloc_at=0,fail_free_at=0;
    // Closed-loop headroom at decode token boundaries (see watermark_tick).
    bool watermark=true;size_t watermark_low_mib=128,watermark_high_mib=448;
    uint64_t watermark_ticks=0,watermark_busy=0,watermark_shrinks=0,watermark_grows=0;size_t watermark_free_min=SIZE_MAX;
    reclaimer_register_fn reclaimer_register=nullptr;
    reclaimer_unregister_fn reclaimer_unregister=nullptr;
    std::unique_ptr<vram_slot_table> table;
    const vram_compute_api * compute_api=nullptr;void * engine=nullptr;
    // Explicit-format engine (kimi_vram_format_api). A context uses one of the two engines.
    const vram_format_compute_api * format_api=nullptr;bool format_mode=false;
    const kimi_mxfp4_batch_api * mxfp4_batch_api=nullptr;
    kimi_mxfp4_batch * mxfp4_batch=nullptr;
    size_t mxfp4_reserved_accumulator_bytes=0;
    bool mxfp4_batch_enabled=false;
    int weight_type=-1,down_type=-1,n_experts=0;kimi_vram_activation activation={};
    std::thread worker;std::mutex mutex;std::condition_variable cv;
    bool stopping=false,queued=false,prefill_queued=false,done=true;
    struct upload_job { size_t request; int slot; const void * data[3]; bool borrow; };
    std::deque<upload_job> upload_jobs;
    std::vector<vram_compute_expert> work;
    std::vector<uint8_t> input;
    std::vector<float> down;
    std::vector<size_t> output_rows;
    std::vector<size_t> pending_uploads;
    // Read-only grouped-prefill job. The full distinct-expert union is pinned
    // once; the worker reuses the existing <=16-route format engine per token.
    bool prefill_live=false;
    // Decode can initialize the format again after releasing prefill scratch.
    bool prefill_transient_phase=true;
    uint64_t prefill_generation=0;
    int prefill_nslots=0,prefill_ntokens=0;
    size_t prefill_input_stride=0;
    std::vector<int> prefill_slots;
    std::vector<uint8_t> prefill_input;
    std::vector<int32_t> prefill_routes;
    std::vector<float> prefill_weights,prefill_output;
    std::vector<kimi_vram_prefill_transient> prefill_transient;
    char * prefill_transient_arena=nullptr;
    size_t prefill_transient_bytes=0;
    static constexpr size_t mxfp4_prefill_transient_slots=2;
    size_t prefill_transient_min_routes=0;
    size_t prefill_ssd_transient_min_routes=0;
    size_t cold_prefill_admit_per_layer=0;
    bool cold_prefill_admission_phase=false;
    // Storage made outside a decode (a cold prefill, the load) has no decode epoch: the next decode adopts it.
    bool adopt_next_decode=false;
    // Seeds of the graph prefill (seed_offer): empty slots filled with the prompt's most-routed experts of each
    // layer, copied device to device from the scheduler's copy on the tier's own stream, published when the prefill
    // completes. A layer plans its share of the free slots (free / layers not yet planned) at its first offered role.
    struct pending_seed { int slot; kimi_vram_request key; uint8_t roles; };
    std::vector<pending_seed> seeds;
    std::set<int> seed_planned;uint64_t seed_plan_generation=UINT64_MAX;
    bool seed_registered=false;cudaStream_t seed_stream=nullptr;cudaEvent_t seed_done=nullptr;
    std::map<int,struct stat> seed_files; // fstat of the resolver's descriptors, for the seeds' keys
    // the expert-pack resolver (ggml-cuda's, found at registration): tensor name, expert, logical bytes, experts ->
    // descriptor, logical offset, bytes
    using resolve_fn=bool (*)(const char *,int,size_t,int64_t,int *,uint64_t *,size_t *);
    resolve_fn seed_resolve=nullptr;
    uint64_t seeds_planned=0,seeds_published=0,seeds_cancelled=0,seeds_freed=0,seed_copies=0,seed_bytes=0;
    double seed_offer_us=0,seed_wait_us=0;
    uint64_t prefill_count=0; // prefills begun: a seed plan is one prefill's
    uint64_t cold_prefill_admissions=0,cold_prefill_admission_bytes=0,cold_prefill_admit_calls=0;
    double cold_prefill_admission_us=0;
    uint64_t fail_cold_prefill_admit_at=0;
    kimi_vram_prefill_stats prefill_job_stats={};
    kimi_vram_activation prefill_activation={};
    // Opt-in deferred uploads (GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD=1): a borrowed upload runs on its own thread and
    // stream and may outlive finish; its slot stays reserved until collect publishes it (kimi_vram_deferred_api).
    bool deferred_upload=false;
    // aborted: its slot went with a block an OOM or host reclaim gave back; collect reports it, nothing is published
    struct deferred_job { uint64_t id; int slot; kimi_vram_request key; const void * data[3]; bool started, completed, aborted; };
    std::deque<deferred_job> deferred; // admission order; completed entries wait for collect
    std::thread upload_thread;std::condition_variable upload_cv;cudaStream_t deferred_stream=nullptr;
    size_t deferred_capacity=32,deferred_peak=0; // uploads in flight before admit waits
    uint64_t deferred_sequence=0,deferred_uploads=0,deferred_published=0,deferred_layer_waits=0,deferred_capacity_waits=0,deferred_aborted=0;
    double deferred_wait_us=0;
    bool (*deferred_hook)(void *)=nullptr;void * deferred_hook_arg=nullptr; // tests: runs before each copy, false fails it
    float beta=0,linear_beta=0;
    std::string failure;
    struct native_pin { kimi_cache_location_lease lease; int slot; kimi_vram_request key; };
    std::vector<native_pin> native_pins;
    std::vector<kimi_vram_request> native_candidates;
    std::vector<float *> native_outputs;
    uint64_t native_generation=0;
    uint64_t native_queries=0,native_query_hits=0,native_gpu_jobs=0;
    int native_latent=0,native_hidden=0;
    bool native_batch=false,native_begun=false,native_launched=false,native_cancelled=false;

    explicit tier(uint64_t b):budget(b),safety_bytes(env_size("GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB",512,8192)<<20),
        async_upload(env_size("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD",0,1)!=0),
        memory_trace(env_size("GGML_KIMI_CPU_SINK_VRAM_MEMORY_TRACE",0,1)!=0),
        keep_across_requests(env_size("GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS",0,1)!=0),
        timing(env_size("GGML_KIMI_CPU_SINK_VRAM_TIMING",0,1)!=0),
        elastic_requested(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC",0,1)!=0),
        // auto has no base to shrink to at a prefill: it gives back only what an allocation needs
        elastic_demand_reclaim(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_DEMAND_RECLAIM",0,1)!=0 || b==automatic_budget),
        elastic_safety_blocks(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS",0,4096)),
        // cudaGraphInstantiate bypasses the backend-buffer allocator. A real
        // safety=0 run left 5.25 MiB and aborted there; 32 MiB is a compact
        // power-of-two reserve above the measured 21 MiB of late allocations.
        elastic_safety_mib(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB",32,8192)),
        elastic_max_slots(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_MAX_SLOTS",0,1000000)),
        fail_malloc_at(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_FAIL_MALLOC_AT",0,UINT64_MAX)),
        fail_free_at(env_size("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_FAIL_FREE_AT",0,UINT64_MAX)),
        prefill_transient_min_routes(env_size("GGML_KIMI_CPU_PREFILL_Q2_RAM_GPU_MIN_ROUTES",0,2048)),
        prefill_ssd_transient_min_routes(env_size("GGML_KIMI_CPU_PREFILL_Q2_SSD_GPU_MIN_ROUTES",0,2048)),
        cold_prefill_admit_per_layer(env_size("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_PER_LAYER",0,384)),
        fail_cold_prefill_admit_at(env_size("GGML_KIMI_CPU_PREFILL_Q2_COLD_ADMIT_FAIL_AT",0,UINT64_MAX)),
        deferred_upload(env_size("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD",0,1)!=0){
        mxfp4_batch_enabled=env_size("GGML_KIMI_MXFP4_PREFILL_BATCH",1,1)!=0;
        watermark=env_size("GGML_KIMI_CPU_SINK_VRAM_WATERMARK",1,1)!=0;
        watermark_low_mib=env_size("GGML_KIMI_CPU_SINK_VRAM_WATERMARK_LOW_MIB",128,8192);
        watermark_high_mib=env_size("GGML_KIMI_CPU_SINK_VRAM_WATERMARK_HIGH_MIB",448,8192);
        if(watermark && watermark_high_mib<=watermark_low_mib)throw std::runtime_error("GGML_KIMI_CPU_SINK_VRAM_WATERMARK_HIGH_MIB must exceed LOW_MIB");
        pending_uploads.reserve(16);
        if(deferred_upload && !async_upload)throw std::runtime_error("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD requires ASYNC_UPLOAD=1");
        // auto sizes an elastic tier only; a fixed arena would take all free memory and could never give it back
        if(budget==automatic_budget && !(elastic_requested && keep_across_requests && env_size("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL",0,1)))
            throw std::runtime_error("GGML_KIMI_CPU_SINK_VRAM_BYTES=auto requires VRAM_ELASTIC=1, VRAM_GLOBAL_POOL=1 and VRAM_KEEP_ACROSS_REQUESTS=1");
    }
    ~tier(){
        try {
            disarm_reclaimer();unregister_oom_reclaimer();unregister_seed();
            // A server can stop after the response while the last deferred
            // upload is complete but has not been collected. release() first
            // aborts those unpublished table reservations, then frees blocks;
            // freeing blocks directly would reject the still-uploading tail.
            release();
            memory_token_report();
            {std::lock_guard<std::mutex> lock(mutex);stopping=true;}cv.notify_all();upload_cv.notify_all();
            if(worker.joinable())worker.join();
            if(upload_thread.joinable())upload_thread.join();
            if(device>=0)cuda_check(cudaSetDevice(device));
            report("destroy");
            if(watermark)std::fprintf(stderr,"[kimi_vram_watermark] low_mib=%zu high_mib=%zu ticks=%llu busy=%llu shrinks=%llu grows=%llu free_min=%zu\n",
                watermark_low_mib,watermark_high_mib,(unsigned long long)watermark_ticks,(unsigned long long)watermark_busy,
                (unsigned long long)watermark_shrinks,(unsigned long long)watermark_grows,watermark_free_min==SIZE_MAX?size_t(0):watermark_free_min);
            if(deferred_upload)std::fprintf(stderr,"[kimi_vram_deferred] uploads=%llu published=%llu aborted=%llu layer_waits=%llu capacity_waits=%llu wait_us=%.0f peak=%zu\n",
                (unsigned long long)deferred_uploads,(unsigned long long)deferred_published,(unsigned long long)deferred_aborted,
                (unsigned long long)deferred_layer_waits,(unsigned long long)deferred_capacity_waits,deferred_wait_us,deferred_peak);
            if(unified)std::fprintf(stderr,"[kimi_vram_host_reserve] reason=destroy floor=%zu demand=%zu host_reclaims=%llu host_reclaim_refusals=%llu live_reclaim_refusals=%llu\n",
                host_floor,host_demand,(unsigned long long)host_reclaims,(unsigned long long)host_reclaim_refusals,(unsigned long long)live_reclaim_refusals);
            if(adaptive_admission)admission_log("final ");
            if(seeds_planned)std::fprintf(stderr,"[kimi_vram_seed] reason=destroy planned=%llu published=%llu cancelled=%llu freed_unused=%llu copies=%llu bytes=%llu offer_us=%.0f wait_us=%.0f\n",
                (unsigned long long)seeds_planned,(unsigned long long)seeds_published,(unsigned long long)seeds_cancelled,
                (unsigned long long)seeds_freed,(unsigned long long)seed_copies,(unsigned long long)seed_bytes,seed_offer_us,seed_wait_us);
            if(host_readable || host_jobs || host_refused || host_probe_taken){
                kimi_vram_host_costs c={};costs2(c);
                std::fprintf(stderr,"[kimi_vram_host] in_place=%d jobs=%llu batches=%llu copies=%llu reserved_copies=%llu refused=%llu copy_mb=%.1f copy_us=%.0f "
                    "resident_us_per_expert=%.1f host_us_per_expert=%.1f resident_call_us=%.1f resident_expert_us=%.1f host_call_us=%.1f host_expert_us=%.1f "
                    "copy_us_per_mb=%.2f scratch_slots=%zu host_floor=%zu\n",
                    int(host_readable),(unsigned long long)host_jobs,(unsigned long long)host_batches,(unsigned long long)host_copies,
                    (unsigned long long)host_reserved_copies,(unsigned long long)host_refused,host_copy_bytes/1e6,host_copy_us,
                    resident_us_per_expert,host_us_per_expert,c.resident_call_us,c.resident_expert_us,c.host_call_us,c.host_expert_us,
                    c.copy_us_per_mb,host_scratch?host_scratch_slots:size_t(0),host_floor);
            }
            if(timing)std::fprintf(stderr,"[kimi_vram_tier_timing] launches=%llu handoff_us=%.0f finish_drain_us=%.0f compute_us=%.0f upload_us=%.0f upload_wait_us=%.0f uploads=%llu hits=%llu cpu_requests=%llu\n",
                (unsigned long long)launches,handoff_us,finish_drain_us,compute_us,upload_us,upload_wait_us,(unsigned long long)uploads,(unsigned long long)gpu_hits,(unsigned long long)cpu_requests);
            if(mxfp4_batch)mxfp4_batch_api->destroy(mxfp4_batch);
            if(engine){if(format_mode)format_api->destroy(engine);else compute_api->destroy(engine);}
            if(prefill_transient_arena)cuda_check(cudaFree(prefill_transient_arena));
            if(upload_stream)cuda_check(cudaStreamDestroy(upload_stream));
            if(deferred_stream)cuda_check(cudaStreamDestroy(deferred_stream));
            if(seed_done)cuda_check(cudaEventDestroy(seed_done));
            if(seed_stream)cuda_check(cudaStreamDestroy(seed_stream));
            if(host_copy_stream)cuda_check(cudaStreamDestroy(host_copy_stream));
            for(cudaEvent_t e:host_events)cuda_check(cudaEventDestroy(e));
        } catch(const std::exception & e) {
            std::fprintf(stderr,"[kimi_vram_tier] destroy could not drain owned work: %s\n",e.what());std::abort();
        }
    }
    // Free memory the tier may still take. On unified memory that is MemAvailable, as it reads at the kernel's full
    // watermark boost (ggml's device-memory query), less the ledger's floor and the RAM cache's demand: a boost the
    // kernel raises after the tier grew cannot take MemAvailable under the floor.
    void mem_info(size_t * free_bytes,size_t * total){
        const auto & q=ggml_memory_query::instance();
        if(q.get)q.get(device,free_bytes,total);else cuda_check(cudaMemGetInfo(free_bytes,total));
        const size_t keep=host_floor+host_demand;
        if(unified)*free_bytes=*free_bytes>keep?*free_bytes-keep:0;
    }
    void publish_reserve(){
        const auto & q=ggml_memory_query::instance();
        if(unified && q.set_reserve)q.set_reserve(device,host_floor+host_demand);
    }
    // Room on unified memory for `bytes` of host memory about to be allocated outside the tier (the host-memory
    // ledger asks; 0 at a decode token boundary restores the tier's own margin after the floor or the demand grew).
    // Gives the last blocks back while the free memory is below bytes plus the low watermark, keeping one block;
    // deferred uploads do not stop it (reclaim_last_block). False when a layer is live or not enough could go.
    bool reclaim_host(size_t bytes,const char * reason){
        if(!unified || !elastic_active || !table)return false;
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if(poisoned)return false;
        const size_t low=watermark_low_mib<<20,need=bytes>SIZE_MAX-low?SIZE_MAX:bytes+low;
        size_t free_bytes=0,total=0;cuda_check(cudaSetDevice(device));mem_info(&free_bytes,&total);
        if(free_bytes>=need)return true;
        if(busy()){
            ++host_reclaim_refusals;
            std::fprintf(stderr,"[kimi_vram_elastic] event=host_reclaim status=live reason=%s request_bytes=%zu free_bytes=%zu\n",reason,bytes,free_bytes);
            return false;
        }
        const size_t before=slot_locations.size(),free_before=free_bytes;
        while(free_bytes<need && blocks.size()>1 && reclaim_last_block(reason,false))mem_info(&free_bytes,&total);
        if(slot_locations.size()<before){
            ++host_reclaims;
            std::fprintf(stderr,"[kimi_vram_elastic] event=host_reclaim status=%s reason=%s request_bytes=%zu free_before=%zu free_after=%zu freed_slots=%zu slots=%zu floor=%zu demand=%zu\n",
                free_bytes>=need?"pass":"short",reason,bytes,free_before,free_bytes,before-slot_locations.size(),slot_locations.size(),host_floor,host_demand);
        }
        return free_bytes>=need;
    }
    // An admission pays one upload and earns the hits the table expects at its selection count; evicting a
    // resident gives up the hits that entry is still expected to earn. Each hit saves a miss minus a hit.
    // Admit when the difference, times that saving, exceeds the upload. Starts at the token boundary after both
    // costs are measured: the upload by probe_upload when the tier gets its storage, the miss over the last decode
    // token (the sink reports one per layer), as at every later token, not from one layer. A first token's misses
    // are first touches read from the drive (4.6-5.0 ms a miss on the Spark against about 1.2 ms from the fourth
    // token, when recurring experts come from the RAM cache): the price follows what a miss costs at the time.
    void update_admission(){
        static const bool enabled=env_size("GGML_KIMI_CPU_SINK_VRAM_ADAPTIVE_ADMISSION",1,1)!=0; // diagnostic A/B switch
        if(!enabled || !table || !table->count_decay)return;
        double miss=0,hit=0,upload=0; // the sink, the worker and the upload thread write them under mutex
        {std::lock_guard<std::mutex> lock(mutex);miss=miss_us;hit=resident_us_per_expert;upload=upload_us_per_expert;}
        if(!(miss>0) || !(upload>0))return;
        const double saved=miss-hit;
        table->admission_hits=saved>0?upload/saved:1e9;
        table->refresh_expected();
        const double before=last_min_count,next=table->admission_count();
        last_min_count=next;adaptive_admission=true;++admission_updates;
        last_cost_hits=table->admission_hits;last_evictions=table->earned_samples;
        for(int c=1;c<=3;++c)last_expected[c-1]=table->expected_hits(c);
        if(std::fabs(next-before)>0.25 || admission_updates==1 || admission_updates%2048==0)admission_log("");
    }
    // From the last update, so that it can also report after the table is gone.
    void admission_log(const char * tag){
        std::fprintf(stderr,"[kimi_vram_admission] %smin_count=%.3f cost_hits=%.3f upload_us=%.1f miss_us=%.1f hit_us=%.1f expected_hits_c1=%.2f c2=%.2f c3=%.2f evictions=%llu updates=%llu\n",
            tag,last_min_count,last_cost_hits,upload_us_per_expert,miss_us,resident_us_per_expert,
            last_expected[0],last_expected[1],last_expected[2],(unsigned long long)last_evictions,(unsigned long long)admission_updates);
    }
    void grow_after_first_token(){
        if(!unified || !elastic_active || decode_growth_done || ++phase_decode_tokens<2)return;
        const auto start=clock_type::now(); // the token boundary waits from here: deferred uploads, then growth
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if(settle())grow_decode_locked(start); // otherwise (a live layer) the next token boundary tries again
    }
    void detect_unified(){
        const auto & q=ggml_memory_query::instance();
        unified=device>=0 && q.unified && q.unified(device);
        // The GPU may compute experts that sit in host memory only when it reads them in place through
        // the host page tables (ATS); with HMM a GPU read migrates the page into device memory.
        int in_place=0;
        host_readable=device>=0 && cudaDeviceGetAttribute(&in_place,cudaDevAttrPageableMemoryAccessUsesHostPageTables,device)==cudaSuccess && in_place;
        if(host_readable)std::fprintf(stderr,"[kimi_vram_host] device=%d host_page_tables=1 unified=%d\n",device,int(unified));
        if(unified)publish_reserve();
    }
    // Diagnostic only: sample at completed layer boundaries. No allocation or
    // additional stream synchronization; speed runs leave this switch at 0.
    void memory_sample(const char * event=nullptr,double elapsed_us=0){
        if(!memory_trace || device<0)return;
        cuda_check(cudaSetDevice(device));
        size_t available=0,total=0;mem_info(&available,&total);
        if(active && !event){++memory_samples;memory_free_min=std::min(memory_free_min,uint64_t(available));memory_free_max=std::max(memory_free_max,uint64_t(available));}
        if(event)std::fprintf(stderr,"[kimi_vram_memory] event=%s epoch=%llu token=%llu decode=%d budget_bytes=%llu capacity_bytes=%llu free_bytes=%zu total_bytes=%zu safety_bytes=%llu elapsed_us=%.3f\n",
            event,(unsigned long long)epoch,(unsigned long long)token,int(active),
            (unsigned long long)budget,(unsigned long long)capacity_bytes,available,total,(unsigned long long)safety_bytes,elapsed_us);
    }
    void memory_token_report(){
        if(!memory_trace || !memory_samples)return;
        std::fprintf(stderr,"[kimi_vram_memory] event=token_end epoch=%llu token=%llu samples=%llu min_free_bytes=%llu max_free_bytes=%llu capacity_bytes=%llu\n",
            (unsigned long long)epoch,(unsigned long long)token,(unsigned long long)memory_samples,
            (unsigned long long)memory_free_min,(unsigned long long)memory_free_max,(unsigned long long)capacity_bytes);
        memory_samples=0;memory_free_min=UINT64_MAX;memory_free_max=0;
    }
    void report(const char * reason){
        std::fprintf(stderr,"[kimi_vram_tier] reason=%s budget_bytes=%llu capacity_bytes=%llu slots=%zu hits=%llu cpu_requests=%llu uploads=%llu upload_bytes=%llu upload_us=%.3f compute_us=%.3f device_scratch=%zu host_scratch=%zu borrowed_uploads=%llu upload_wait_us=%.3f cold_prefill_admissions=%llu cold_prefill_admission_bytes=%llu cold_prefill_admission_us=%.3f min_count=%.3f async_upload=%d global_pool=%d half_life=%.3f deferred_upload=%d elastic=%d demand_reclaim=%d blocks=%zu base_slots=%zu safety_blocks=%zu safety_mib=%zu proactive_blocks=%llu reactive_blocks=%llu base_blocks_reclaimed=%llu poisoned=%d\n",
            reason,(unsigned long long)budget,(unsigned long long)capacity_bytes,table?table->slots.size():0,
            (unsigned long long)gpu_hits,(unsigned long long)cpu_requests,(unsigned long long)uploads,
            (unsigned long long)upload_bytes,upload_us,compute_us,engine?(format_mode?format_api->device_scratch_bytes(engine):compute_api->device_scratch_bytes(engine)):0,
            engine?(format_mode?format_api->host_scratch_bytes(engine):compute_api->host_scratch_bytes(engine)):0,(unsigned long long)borrowed_uploads,upload_wait_us,
            (unsigned long long)cold_prefill_admissions,(unsigned long long)cold_prefill_admission_bytes,cold_prefill_admission_us,
            table?table->minimum_count:0,int(async_upload),table?int(table->global_pool):0,table?table->half_life:2,int(deferred_upload),int(elastic_active),int(elastic_demand_reclaim),blocks.size(),base_slots,elastic_safety_blocks,elastic_safety_mib,
            (unsigned long long)proactive_blocks,(unsigned long long)reactive_blocks,
            (unsigned long long)base_blocks_reclaimed,int(poisoned));
        if(native_queries)std::fprintf(stderr,"[kimi_vram_native] reason=%s queries=%llu ready_hits=%llu gpu_jobs=%llu\n",reason,
            (unsigned long long)native_queries,(unsigned long long)native_query_hits,(unsigned long long)native_gpu_jobs);
    }
    bool has_storage() const{return arena || !blocks.empty();}
    char * slot_base(size_t slot) const{
        if(!table || slot>=table->slots.size())throw std::runtime_error("VRAM slot address out of range");
        if(!elastic_active){if(!arena)throw std::runtime_error("VRAM arena is unavailable");return arena+slot*stride;}
        if(slot>=slot_locations.size() || !slot_locations[slot])throw std::runtime_error("VRAM elastic slot is unavailable");
        return slot_locations[slot];
    }
    // Slots in use by anything but a deferred upload: a layer, a residency lease, a native batch, grouped prefill.
    bool busy(){
        if((table && (table->in_flight || table->lease_generation || table->native_pinned())) || live_ready_generation ||
                native_batch || !native_pins.empty() || prefill_live)return true;
        std::lock_guard<std::mutex> lock(mutex);
        return !done || queued || prefill_queued || !upload_jobs.empty() || !pending_uploads.empty();
    }
    bool quiescent(){
        if(!table || busy())return false;
        std::lock_guard<std::mutex> lock(mutex);
        return deferred.empty();
    }
    bool deferred_settled(){
        std::lock_guard<std::mutex> lock(mutex);
        return std::all_of(deferred.begin(),deferred.end(),[](const deferred_job & d){return d.completed;});
    }
    // Unified memory resizes at a token boundary the way host reclaim does: deferred uploads do not stop it. Waits
    // for their copies, so that no upload reads the block map while it changes; completed uploads keep their slot
    // reservations for collect (growth only appends, a shrink aborts the ones in the block it frees). False when
    // a layer, lease or batch holds slots. Requires lifecycle_mutex.
    bool settle(){
        if(!table || busy())return false;
        drain_deferred();
        return true;
    }
    static int oom_reclaim_thunk(void * p,size_t requested,size_t * freed){
        auto * t=static_cast<tier *>(p);
        try{return t->reclaim_one(requested,freed);}catch(const std::exception & e){
            std::fprintf(stderr,"[kimi_vram_elastic] event=reactive_reclaim status=fail error=%s\n",e.what());
            if(freed)*freed=0;
            return -1;
        }
    }
    void register_oom_reclaimer(){
        if(!elastic_active || reclaimer_registered)return;
        reclaimer_register=reinterpret_cast<reclaimer_register_fn>(dlsym(RTLD_DEFAULT,"ggml_cuda_oom_reclaimer_register"));
        reclaimer_unregister=reinterpret_cast<reclaimer_unregister_fn>(dlsym(RTLD_DEFAULT,"ggml_cuda_oom_reclaimer_unregister"));
        if(!reclaimer_register || !reclaimer_unregister || !reclaimer_register(device,oom_reclaim_thunk,this))
            throw std::runtime_error("CUDA OOM reclaimer registry is unavailable or busy");
        reclaimer_registered=true;
    }
    void unregister_oom_reclaimer(){
        if(!reclaimer_registered)return;
        if(!reclaimer_unregister || !reclaimer_unregister(device,oom_reclaim_thunk,this))
            throw std::runtime_error("CUDA OOM reclaimer unregister failed");
        reclaimer_registered=false;
    }
    void disarm_reclaimer(){
        std::lock_guard<std::mutex> lock(lifecycle_mutex);reclaimer_armed=false;
    }
    static int seed_offer_thunk(void * p,const char * name,const void * data,size_t expert_bytes,int64_t n,
            const int32_t * counts,void * ready,void ** done){
        auto * t=static_cast<tier *>(p);
        try{return t->seed_offer(name,static_cast<const char *>(data),expert_bytes,n,counts,static_cast<cudaEvent_t>(ready),done);}
        catch(const std::exception & e){
            std::fprintf(stderr,"[kimi_vram_seed] event=offer status=fail error=%s\n",e.what());
            if(done)*done=nullptr;
            return -1;
        }
    }
    void register_seed(){
        if(seed_registered || device<0)return;
        if(!seed_resolve)seed_resolve=reinterpret_cast<resolve_fn>(dlsym(RTLD_DEFAULT,"ggml_cuda_moe_expert_pack_resolve_raw_direct_shape"));
        auto reg=reinterpret_cast<seed_register_fn>(dlsym(RTLD_DEFAULT,"ggml_cuda_moe_seed_register"));
        seed_registered=reg && reg(device,seed_offer_thunk,this);
    }
    void unregister_seed(){
        if(!seed_registered)return;
        auto unreg=reinterpret_cast<seed_register_fn>(dlsym(RTLD_DEFAULT,"ggml_cuda_moe_seed_unregister"));
        if(!unreg || !unreg(device,seed_offer_thunk,this))throw std::runtime_error("CUDA prefill seed unregister failed");
        seed_registered=false;
    }
    // A seed's key as the sink keys a decode request's payload (kimi_vram_source_identity): each role's file, its
    // logical offset and its logical bytes, from the resolver the sink's sources come from.
    bool seed_key(int layer,int expert,kimi_vram_request & r){
        static const char * const names[3]={"up","gate","down"};
        if(!seed_resolve)return false;
        r={};r.layer=layer;r.expert=expert;
        for(int i=0;i<3;++i){
            char name[64];std::snprintf(name,sizeof(name),"blk.%d.ffn_%s_exps.weight",layer,names[i]);
            int fd=-1;uint64_t offset=0;size_t bytes=0;
            if(!seed_resolve(name,expert,role_bytes[i],n_experts,&fd,&offset,&bytes) || fd<0)return false;
            auto it=seed_files.find(fd);
            if(it==seed_files.end()){struct stat st={};if(fstat(fd,&st))return false;it=seed_files.emplace(fd,st).first;}
            r.role[i]=kimi_vram_source_identity(&it->second,offset,role_bytes[i]);
        }
        return true;
    }
    // This layer's share of the empty slots, the free ones over the layers not planned yet in this prefill (what an
    // earlier layer left rolls forward), filled with the experts the pass routed most, those not resident already.
    void seed_plan(int layer,const int32_t * counts){
        const size_t left=table->pool_layers.size()+1>seed_planned.size()?table->pool_layers.size()+1-seed_planned.size():1;
        const size_t quota=table->free_slots(layer)/left;
        if(!quota)return;
        std::vector<int> order;
        for(int e=0;e<n_experts;++e)if(counts[e]>0)order.push_back(e);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return counts[a]>counts[b];});
        size_t taken=0;
        for(int e:order){
            if(taken==quota)break;
            kimi_vram_request r;
            if(!seed_key(layer,e,r))continue;
            const int slot=table->reserve_cold_prefill(r);
            if(slot>=0){seeds.push_back({slot,r,0});++taken;}
            else if(!table->free_slots(layer))break;
        }
        seeds_planned+=taken;
    }
    // ggml-cuda's prefill seed callback, on the scheduler's thread after it copied one routed expert tensor (`data`,
    // `expert_bytes` an expert) of a graph prefill pass to the device and recorded `ready` behind the copy. The
    // planned experts' role goes into their slots on the tier's stream; *done is recorded behind those copies, and
    // the scheduler's stream waits for it before it can reuse the source.
    int seed_offer(const char * name,const char * data,size_t expert_bytes,int64_t n,const int32_t * counts,
            cudaEvent_t ready,void ** done){
        if(done)*done=nullptr;
        int layer=-1,consumed=0;char role_name[8]={};
        if(!name || !data || !counts || !ready || !done ||
                std::sscanf(name,"blk.%d.ffn_%7[a-z]_exps.weight%n",&layer,role_name,&consumed)!=2 || !consumed || name[consumed])return 0;
        const int role=!std::strcmp(role_name,"up")?0:!std::strcmp(role_name,"gate")?1:!std::strcmp(role_name,"down")?2:-1;
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if(role<0 || poisoned || !format_mode || !table || !has_storage() || active || n!=n_experts ||
                expert_bytes!=role_bytes[role] || table->in_flight || table->lease_generation ||
                std::find(table->pool_layers.begin(),table->pool_layers.end(),layer)==table->pool_layers.end())return 0;
        const auto start=clock_type::now();
        if(seed_plan_generation!=prefill_count){seed_plan_generation=prefill_count;seed_planned.clear();seed_files.clear();}
        if(seed_planned.insert(layer).second)seed_plan(layer,counts);
        size_t offset=0;for(int i=0;i<role;++i)offset+=role_bytes[i];
        bool copied=false;
        for(auto & s:seeds){
            if(s.key.layer!=layer || ((s.roles>>role)&1))continue;
            if(!copied){
                cuda_check(cudaSetDevice(device));
                if(!seed_stream)cuda_check(cudaStreamCreateWithFlags(&seed_stream,cudaStreamNonBlocking));
                if(!seed_done)cuda_check(cudaEventCreateWithFlags(&seed_done,cudaEventDisableTiming));
                cuda_check(cudaStreamWaitEvent(seed_stream,ready,0));
                copied=true;
            }
            cuda_check(cudaMemcpyAsync(slot_base(size_t(s.slot))+offset,data+size_t(s.key.expert)*expert_bytes,role_bytes[role],
                cudaMemcpyDeviceToDevice,seed_stream));
            s.roles=uint8_t(s.roles|(1u<<role));++seed_copies;seed_bytes+=role_bytes[role];
        }
        if(copied){cuda_check(cudaEventRecord(seed_done,seed_stream));*done=seed_done;}
        seed_offer_us+=micros(start);
        return copied?1:0;
    }
    // Waits for the seed copies, then drops the seeds in slots from `first` on (a block given back, or all of them).
    // Requires lifecycle_mutex.
    void cancel_seeds(size_t first=0){
        if(seeds.empty())return;
        cuda_check(cudaSetDevice(device));cuda_check(cudaStreamSynchronize(seed_stream));
        const auto end=std::remove_if(seeds.begin(),seeds.end(),[&](const pending_seed & s){
            if(size_t(s.slot)<first)return false;
            if(table)table->cancel_cold_prefill(size_t(s.slot),s.key);
            ++seeds_cancelled;return true;});
        seeds.erase(end,seeds.end());
    }
    // The prefill is over: its seeds are published once their copies are done; one missing a role (its pass ended
    // early) is dropped. Requires lifecycle_mutex.
    void publish_seeds(){
        if(seeds.empty())return;
        const auto start=clock_type::now();
        cuda_check(cudaSetDevice(device));cuda_check(cudaStreamSynchronize(seed_stream));
        seed_wait_us+=micros(start);
        size_t published=0,cancelled=0;
        for(const auto & s:seeds){
            if(s.roles==7){table->publish_cold_prefill(size_t(s.slot),s.key);++published;}
            else{table->cancel_cold_prefill(size_t(s.slot),s.key);++cancelled;}
        }
        seeds_published+=published;seeds_cancelled+=cancelled;
        std::fprintf(stderr,"[kimi_vram_seed] event=publish layers=%zu seeded=%zu cancelled=%zu slots=%zu copies=%llu bytes=%llu offer_us=%.0f wait_us=%.0f\n",
            seed_planned.size(),published,cancelled,table->slots.size(),(unsigned long long)seed_copies,(unsigned long long)seed_bytes,
            seed_offer_us,seed_wait_us);
        seeds.clear();
    }
    bool allocate_block(size_t count,const char * reason){
        if(!elastic_active || !count || count>elastic_block_slots || slot_locations.size()+count>max_slots)return false;
        ++malloc_calls;
        if(fail_malloc_at && malloc_calls==fail_malloc_at){
            std::fprintf(stderr,"[kimi_vram_elastic] event=grow status=injected_fail reason=%s call=%llu slots=%zu\n",
                reason,(unsigned long long)malloc_calls,count);return false;
        }
        char * data=nullptr;const size_t bytes=count*stride;
        cuda_check(cudaSetDevice(device));const cudaError_t rc=cudaMalloc(reinterpret_cast<void **>(&data),bytes);
        if(rc!=cudaSuccess){(void)cudaGetLastError();std::fprintf(stderr,"[kimi_vram_elastic] event=grow status=oom reason=%s slots=%zu bytes=%zu\n",reason,count,bytes);return false;}
        const size_t first=slot_locations.size();
        blocks.push_back({data,first,count});
        for(size_t i=0;i<count;++i)slot_locations.push_back(data+i*stride);
        capacity_bytes+=bytes;
        if(table && table->slots.size()<slot_locations.size())table->resize_global(slot_locations.size());
        return true;
    }
    bool free_last_block(const char * reason,bool reactive){
        if(blocks.empty())return false;
        const auto block=blocks.back();
        cancel_seeds(block.first_slot);
        ++free_calls;
        if(fail_free_at && free_calls==fail_free_at){poisoned=true;throw std::runtime_error("injected elastic cudaFree failure");}
        cuda_check(cudaSetDevice(device));const cudaError_t rc=cudaFree(block.data);
        if(rc!=cudaSuccess){(void)cudaGetLastError();poisoned=true;throw std::runtime_error(std::string("elastic cudaFree failed: ")+cudaGetErrorString(rc));}
        if(table)table->resize_global(block.first_slot);
        slot_locations.resize(block.first_slot);blocks.pop_back();capacity_bytes-=block.slot_count*stride;
        if(reactive){++reactive_blocks;if(block.first_slot<base_slots)++base_blocks_reclaimed;}
        else if(!std::strcmp(reason,"prefill"))++proactive_blocks;
        std::fprintf(stderr,"[kimi_vram_elastic] event=shrink status=pass reason=%s generation=%llu class=%s freed_slots=%zu freed_bytes=%zu slots=%zu blocks=%zu\n",
            reason,(unsigned long long)phase_generation,block.first_slot>=base_slots?"elastic":"base",block.slot_count,
            block.slot_count*stride,slot_locations.size(),blocks.size());
        return true;
    }
    // free_last_block while deferred uploads may still be in flight: waits for their copies, then aborts the
    // reservations of those that target the block; collect reports them as aborted. Requires lifecycle_mutex and a
    // tier that is not busy().
    bool reclaim_last_block(const char * reason,bool reactive){
        if(blocks.empty())return false;
        if(deferred_upload){
            drain_deferred();
            const size_t first=blocks.back().first_slot;
            std::lock_guard<std::mutex> lock(mutex);
            for(auto & d:deferred){
                if(d.aborted || size_t(d.slot)<first)continue;
                if(table)table->cancel_upload(size_t(d.slot),d.key);
                d.aborted=true;++deferred_aborted;
            }
        }
        return free_last_block(reason,reactive);
    }
    int reclaim_one(size_t requested,size_t * freed){
        if(!freed)return -1;
        *freed=0;
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if(!elastic_active || !reclaimer_armed || poisoned)return poisoned?-1:0;
        // A live layer holds its slots: the allocation fails as it would without the tier. Deferred uploads alone do
        // not stop a reclaim (reclaim_last_block waits for them and aborts the ones in the freed block).
        if(busy()){
            ++live_reclaim_refusals;
            std::fprintf(stderr,"[kimi_vram_elastic] event=reactive_reclaim status=live generation=%llu request_bytes=%zu refusals=%llu\n",
                (unsigned long long)phase_generation,requested,(unsigned long long)live_reclaim_refusals);
            return 0;
        }
        ++phase_reclaim_requests;
        const size_t before=capacity_bytes;
        if(!reclaim_last_block("oom",true))return 0;
        *freed=before-capacity_bytes;
        std::fprintf(stderr,"[kimi_vram_elastic] event=reactive_reclaim status=pass generation=%llu request_bytes=%zu freed_bytes=%zu below_base=%d\n",
            (unsigned long long)phase_generation,requested,*freed,slot_locations.size()<base_slots?1:0);
        return 1;
    }
    // Slots of the first allocation. A byte budget (GGML_KIMI_CPU_SINK_VRAM_BYTES=N) keeps its legacy meaning, the
    // base. Without one (auto, elastic only) the tier starts from one block and grows into what is free. The caller
    // has checked that one slot fits above the safety reserve.
    size_t initial_slots(size_t free_bytes,size_t logical_max,bool global) const{
        if(budget==automatic_budget && elastic_requested && global && keep_across_requests){
            const size_t low=watermark_low_mib<<20,above=free_bytes>low?(free_bytes-low)/stride:0;
            return std::min(logical_max,std::max<size_t>(1,std::min(elastic_block_slots,above)));
        }
        return size_t(std::min<uint64_t>(logical_max,std::min<uint64_t>(budget,free_bytes-safety_bytes)/stride));
    }
    void configure_elastic(bool global,size_t logical_max){
        elastic_active=elastic_requested && global && keep_across_requests;
        max_slots=logical_max;
        if(elastic_max_slots)max_slots=std::min(max_slots,std::max(base_slots,elastic_max_slots));
        if(!elastic_active){
            if(elastic_requested)std::fprintf(stderr,"[kimi_vram_elastic] event=disabled global_pool=%d keep_across_requests=%d\n",int(global),int(keep_across_requests));
            return;
        }
        blocks.reserve((max_slots+elastic_block_slots-1)/elastic_block_slots);
        slot_locations.reserve(max_slots);table->reserve_global(max_slots);
    }
    void allocate_base(){
        while(slot_locations.size()<base_slots){
            const size_t count=std::min(elastic_block_slots,base_slots-slot_locations.size());
            if(!allocate_block(count,"base"))throw std::runtime_error("elastic VRAM base allocation failed");
        }
        register_oom_reclaimer();
    }
    void prepare_prefill_reclaim(){
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if(!elastic_active)return;
        if(poisoned)throw std::runtime_error("elastic VRAM tier is poisoned");
        if(!quiescent())throw std::runtime_error("elastic VRAM prefill transition is not quiescent");
        cancel_seeds();seeds_freed+=table->free_unused_seeds();
        ++phase_generation;reclaimer_armed=false;
        cuda_check(cudaSetDevice(device));size_t total=0;mem_info(&phase_free_before,&total);
        phase_slots_before=slot_locations.size();
        phase_reactive_start=reactive_blocks;phase_base_reclaimed_start=base_blocks_reclaimed;phase_reclaim_requests=0;
        const size_t planned_blocks=elastic_demand_reclaim?0:
            (slot_locations.size()>base_slots?(slot_locations.size()-base_slots+elastic_block_slots-1)/elastic_block_slots:0);
        if(!elastic_demand_reclaim){while(slot_locations.size()>base_slots)free_last_block("prefill",false);}
        decode_growth_done=false;
        reclaimer_armed=true;
        const uint64_t safety=uint64_t(elastic_safety_blocks)*uint64_t(elastic_block_slots)*uint64_t(stride)+
            uint64_t(elastic_safety_mib)*uint64_t(1U<<20);
        std::fprintf(stderr,"[kimi_vram_elastic] event=prefill_armed generation=%llu policy=%s free_before=%zu safety_bytes=%llu slots_before=%zu slots_after=%zu planned_blocks=%zu blocks=%zu base_slots=%zu\n",
            (unsigned long long)phase_generation,elastic_demand_reclaim?"demand":"base",phase_free_before,
            (unsigned long long)safety,phase_slots_before,slot_locations.size(),planned_blocks,blocks.size(),base_slots);
    }
    // start: when the caller began to wait for the growth, for the receipt's duration
    void grow_decode_locked(clock_type::time_point start=clock_type::now()){
        if(!elastic_active || poisoned || !table)return;
        if(busy() || !deferred_settled())throw std::runtime_error("elastic VRAM decode grow is not quiescent");
        size_t free_bytes=0,total=0;cuda_check(cudaSetDevice(device));mem_info(&free_bytes,&total);
        const uint64_t safety=uint64_t(elastic_safety_blocks)*uint64_t(elastic_block_slots)*uint64_t(stride)+
            uint64_t(elastic_safety_mib)*uint64_t(1U<<20);
        const size_t available=free_bytes>safety?(free_bytes-size_t(safety))/stride:0;
        const size_t target=std::min(max_slots,slot_locations.size()+available);
        const size_t before=slot_locations.size();
        while(slot_locations.size()<target){
            const size_t count=std::min(elastic_block_slots,target-slot_locations.size());
            // On unified memory a block can cost more than the estimate it was planned from (page cache that
            // will not be reclaimed): measure again before each one.
            if(unified){size_t now=0,t=0;mem_info(&now,&t);if(now<safety+count*stride)break;}
            if(!allocate_block(count,"decode"))break;
        }
        ++grow_events;
        decode_growth_done=true;
        std::fprintf(stderr,"[kimi_vram_elastic] event=decode_grow generation=%llu status=pass epoch=%llu token=%llu decode_token=%llu us=%.0f before_slots=%zu target_slots=%zu after_slots=%zu added_slots=%zu blocks=%zu free_before=%zu safety_blocks=%zu safety_mib=%zu safety_bytes=%llu\n",
            (unsigned long long)phase_generation,(unsigned long long)epoch,(unsigned long long)token,
            (unsigned long long)phase_decode_tokens,micros(start),before,target,slot_locations.size(),slot_locations.size()-before,
            blocks.size(),free_bytes,elastic_safety_blocks,elastic_safety_mib,(unsigned long long)safety);
    }
    // Decode growth leaves only the elastic safety margin free, and device memory keeps moving after it:
    // CUDA graphs are instantiated per shape (each time the context crosses a 256-position bucket), a
    // longer context grows the reserved graph, kernels load on first use. Instead of predicting all of
    // that with one fixed margin, measure free memory at every decode token boundary and hold it between
    // the watermarks: below the low one give the last block back, above the high one grow back to the
    // safety margin. A discrete GPU resizes only a quiescent tier; a busy boundary waits for the next one. On
    // unified memory deferred uploads do not stop a resize (they are settled first, and only when there is one).
    void watermark_tick(){
        if(!watermark || !elastic_active || poisoned || !table || !decode_growth_done)return;
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        ++watermark_ticks;
        if(unified?busy():!quiescent()){++watermark_busy;return;}
        size_t free_bytes=0,total=0;cuda_check(cudaSetDevice(device));mem_info(&free_bytes,&total);
        watermark_free_min=std::min(watermark_free_min,free_bytes);
        const size_t low=watermark_low_mib<<20,high=watermark_high_mib<<20;
        const size_t safety=size_t(elastic_safety_blocks)*elastic_block_slots*stride+(elastic_safety_mib<<20);
        if(free_bytes<low){
            if(blocks.size()>1 && (unified?reclaim_last_block("watermark",false):free_last_block("watermark",false))){
                ++watermark_shrinks;
                std::fprintf(stderr,"[kimi_vram_elastic] event=watermark_shrink token=%llu free_before=%zu low_bytes=%zu slots=%zu shrinks=%llu\n",
                    (unsigned long long)token,free_bytes,low,slot_locations.size(),(unsigned long long)watermark_shrinks);
            }
            return;
        }
        if(free_bytes<=high || free_bytes<=safety || slot_locations.size()>=max_slots)return;
        if(unified && !settle())return;
        const size_t before=slot_locations.size();
        const size_t target=std::min(max_slots,before+(free_bytes-safety)/stride);
        while(slot_locations.size()<target){
            const size_t count=std::min(elastic_block_slots,target-slot_locations.size());
            if(unified){size_t now=0,t=0;mem_info(&now,&t);if(now<std::max(high,safety)+count*stride)break;}
            if(!allocate_block(count,"watermark"))break;
        }
        if(slot_locations.size()>before){
            ++watermark_grows;
            std::fprintf(stderr,"[kimi_vram_elastic] event=watermark_grow token=%llu free_before=%zu high_bytes=%zu added_slots=%zu slots=%zu grows=%llu\n",
                (unsigned long long)token,free_bytes,high,slot_locations.size()-before,slot_locations.size(),(unsigned long long)watermark_grows);
        }
    }
    // A tier retained from decode leaves only the watermark margin free, often less than the MXFP4 prefill
    // scratch; without the scratch the whole prefill falls back to the CPU. Give blocks back for it, as
    // the accumulator reserve does.
    void make_room_for_prefill_transient(size_t slots){
        if(!elastic_active || prefill_transient_arena || !stride || !slots)return;
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        if(!quiescent())return;
        size_t free_bytes=0,total=0;
        cuda_check(cudaSetDevice(device));mem_info(&free_bytes,&total);
        const size_t need=stride*slots+watermark_low_mib*size_t(1024*1024);
        while(free_bytes<need && free_last_block("prefill_scratch",false))
            mem_info(&free_bytes,&total);
    }
    void allocate_prefill_transient(size_t slots){
        if(prefill_transient_arena || !stride || !slots)return;
        const size_t bytes=stride*slots;
        cuda_check(cudaSetDevice(device));
        const cudaError_t rc=cudaMalloc(reinterpret_cast<void **>(&prefill_transient_arena),bytes);
        if(rc==cudaSuccess){
            prefill_transient_bytes=bytes;
            std::fprintf(stderr,"[kimi_vram_prefill_scratch] status=ready bytes=%zu slots=%zu ram_min_routes=%zu ssd_min_routes=%zu displacement_slots=0\n",
                bytes,slots,prefill_transient_min_routes,prefill_ssd_transient_min_routes);
        }else{
            (void)cudaGetLastError();
            std::fprintf(stderr,"[kimi_vram_prefill_scratch] status=unavailable bytes=%zu slots=%zu ram_min_routes=%zu ssd_min_routes=%zu error=%s fallback=cpu\n",
                bytes,slots,prefill_transient_min_routes,prefill_ssd_transient_min_routes,cudaGetErrorString(rc));
        }
    }
    // Unified memory: the storage at load (kimi_vram_elastic_api reserve), grown into mem_info as the first decode
    // token would grow it, so that the page population leaves the request path. Only the first prefill's graph and
    // workspace are taken back and regrown at the token boundary after its first decode step, as for every later
    // request. A discrete GPU allocates after the first prefill's library allocations (initialize_format).
    void reserve(const kimi_vram_request & r,const kimi_vram_expert_format & f){
        if(table || has_storage())return;
        const auto & q=ggml_memory_query::instance();
        int current=-1;cuda_check(cudaGetDevice(&current));
        if(!q.unified || !q.unified(current))return;
        const auto start=clock_type::now();
        initialize_format(&r,f,true);
        if(!elastic_active)return;
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        grow_decode_locked(start);
    }
    void reserve_mxfp4_accumulator(int ntokens){
        if(!mxfp4_batch)return;
        if(ntokens<1 || latent<1)throw std::runtime_error("invalid MXFP4 accumulator geometry");
        const size_t bytes=size_t(ntokens)*size_t(latent)*sizeof(float);
        if(bytes<=mxfp4_reserved_accumulator_bytes)return;
        cuda_check(cudaSetDevice(device));
        if(elastic_active){
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            if(!quiescent())throw std::runtime_error("MXFP4 accumulator reserve requires a quiescent VRAM tier");
            size_t free_bytes=0,total=0;
            mem_info(&free_bytes,&total);
            const size_t safety=watermark_low_mib*size_t(1024*1024);
            while(free_bytes<bytes+safety && free_last_block("prefill_accumulator",false))
                mem_info(&free_bytes,&total);
        }
        char error[512]={};
        if(!mxfp4_batch_api->begin_accumulate(mxfp4_batch,ntokens,error,sizeof(error)))
            throw std::runtime_error(error[0]?error:"MXFP4 accumulator reserve failed");
        mxfp4_reserved_accumulator_bytes=bytes;
    }
    void release_mxfp4_accumulator(){
        if(!mxfp4_batch)return;
        char error[512]={};
        if(!mxfp4_batch_api->release_accumulate(mxfp4_batch,error,sizeof(error)))
            throw std::runtime_error(error[0]?error:"MXFP4 accumulator release failed");
        mxfp4_reserved_accumulator_bytes=0;
    }
    void phase_complete(bool decode){
        if(decode)prefill_transient_phase=false;
        if(decode && mxfp4_batch){drain();release_mxfp4_accumulator();}
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        reclaimer_armed=false;
        if(decode && table)publish_seeds();
        if(decode && weight_type==int(GGML_TYPE_MXFP4) && prefill_transient_arena){
            if(prefill_live || (table && !quiescent()))
                throw std::runtime_error("MXFP4 prefill scratch release is not quiescent");
            {std::lock_guard<std::mutex> jobs(mutex);
                if(!done || queued || prefill_queued || !upload_jobs.empty())
                    throw std::runtime_error("MXFP4 prefill scratch release has pending work");}
            cuda_check(cudaSetDevice(device));cuda_check(cudaFree(prefill_transient_arena));
            std::fprintf(stderr,"[kimi_vram_prefill_scratch] status=released bytes=%zu phase=decode\n",prefill_transient_bytes);
            prefill_transient_arena=nullptr;prefill_transient_bytes=0;
        }
        if(!elastic_active)return;
        // The server restores and reserves the decode graph before the first
        // decode cache begin().  A tier created outside decode was grown
        // against a larger graph, so re-arm growth here while the tier is
        // still quiescent.  Waiting for begin() is too late: the next
        // phase-complete callback can be request_end with deferred uploads
        // still in flight.
        if(decode && adopt_next_decode && !have_token)decode_growth_done=false;
        if(decode)phase_decode_tokens=0;
        // On unified memory the first decode step's own allocations (graph instantiation, the RAM cache filling
        // its slots) come out of the same memory, so the tier grows at the token boundary after that step,
        // when they are visible (begin -> grow_after_first_token).
        if(decode && !decode_growth_done && !unified)grow_decode_locked();
        else {
            size_t free_bytes=0,total=0;cuda_check(cudaSetDevice(device));mem_info(&free_bytes,&total);
            std::fprintf(stderr,"[kimi_vram_elastic] event=prefill_disarmed generation=%llu policy=%s free_before=%zu free_after=%zu slots_before=%zu slots_after=%zu actual_reactive_blocks=%llu base_blocks_reclaimed=%llu allocation_retries=%llu blocks=%zu\n",
                (unsigned long long)phase_generation,elastic_demand_reclaim?"demand":"base",phase_free_before,free_bytes,
                phase_slots_before,slot_locations.size(),(unsigned long long)(reactive_blocks-phase_reactive_start),
                (unsigned long long)(base_blocks_reclaimed-phase_base_reclaimed_start),
                (unsigned long long)phase_reclaim_requests,blocks.size());
        }
    }
    void drain(){
        std::unique_lock<std::mutex> lock(mutex);cv.wait(lock,[&]{return done;});lock.unlock();
        if(upload_stream){const auto t=clock_type::now();cuda_check(cudaSetDevice(device));cuda_check(cudaStreamSynchronize(upload_stream));upload_wait_us+=micros(t);}
        if(host_copy_stream){cuda_check(cudaSetDevice(device));cuda_check(cudaStreamSynchronize(host_copy_stream));}
    }
    // Waits until every deferred upload has reached the device or failed. Completed entries stay for collect.
    void drain_deferred(){
        if(!deferred_upload)return;
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock,[&]{return std::all_of(deferred.begin(),deferred.end(),[](const deferred_job & d){return d.completed;});});
    }
    void release(){
        if(!native_pins.empty())throw std::runtime_error("release with native VRAM leases");
        memory_sample("before_release");
        const auto release_start=memory_trace?clock_type::now():clock_type::time_point{};
        drain();drain_deferred();release_mxfp4_accumulator();
        {std::lock_guard<std::mutex> lock(mutex);deferred.clear();}
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex);reclaimer_armed=false;
            cancel_seeds();
            if(table)table->abort();
            while(!blocks.empty())free_last_block("release",false);
            table.reset();elastic_active=false;decode_growth_done=false;base_slots=max_slots=0;slot_locations.clear();
        }
        if(arena){cuda_check(cudaSetDevice(device));cuda_check(cudaFree(arena));arena=nullptr;}
        if(weight_type==int(GGML_TYPE_MXFP4) && prefill_transient_arena){
            cuda_check(cudaSetDevice(device));cuda_check(cudaFree(prefill_transient_arena));
            prefill_transient_arena=nullptr;prefill_transient_bytes=0;
        }
        capacity_bytes=0;have_token=false;failure.clear();active=false;pending_uploads.clear();live_ready_generation=0;
        cold_prefill_admission_phase=adopt_next_decode=false;
        prefill_live=false;prefill_generation=0;prefill_queued=false;prefill_slots.clear();prefill_input.clear();
        prefill_routes.clear();prefill_weights.clear();prefill_output.clear();prefill_transient.clear();prefill_job_stats={};
        native_batch=native_begun=native_launched=native_cancelled=false;
        native_candidates.clear();native_outputs.clear();work.clear();output_rows.clear();
        {std::lock_guard<std::mutex> lock(mutex);
            for(auto & task:host_pending)return_host_events(task);
            host_pending.clear();host_done.clear();host_down.clear();host_queued=false;layer_input_ready=false;}
        free_host_scratch();
        if(memory_trace)memory_sample("after_release",micros(release_start));
    }
    void begin(uint64_t e,uint64_t t,bool decode){
        prefill_transient_phase=!decode;
        if(!decode)++prefill_count;
        if(native_batch || !native_pins.empty())throw std::runtime_error("begin with native VRAM batch");
        if(live_ready_generation)throw std::runtime_error("begin with live VRAM residency lease");
        if(!decode || !have_token || epoch!=e || token!=t)memory_token_report();
        // Optional retention through prefill. A failure, or a decode token of a
        // different model/context epoch than the arena's, still releases it all.
        if(!decode && keep_across_requests && has_storage()){
            drain();drain_deferred();
            if(failure.empty()){
                cold_prefill_admission_phase=false;
                have_token=false;active=false;
                free_host_scratch(); // the next decode allocates it again when a batch wants it
                if(elastic_active)prepare_prefill_reclaim();
                if(weight_type==int(GGML_TYPE_MXFP4) && mxfp4_batch){
                    make_room_for_prefill_transient(mxfp4_prefill_transient_slots);
                    allocate_prefill_transient(mxfp4_prefill_transient_slots);
                }
                return;
            }
            report("phase_release");release();
        }
        // A tier first allocated outside decode has no decode epoch yet. The
        // immediately following decode adopts it; any other epoch mismatch
        // still releases the arena exactly as before.
        if(decode && adopt_next_decode && has_storage() && !have_token){
            arena_epoch=e;
        }
        if(!decode || (have_token && epoch!=e) || (decode && has_storage() && arena_epoch!=e)){if(has_storage())report("phase_release");release();}
        const bool next_token=decode && (!have_token || epoch!=e || token!=t);
        if(next_token){
            if(table)table->next_token();
            have_token=true;
        }
        epoch=e;token=t;active=decode;
        if(next_token){
            if(unified && elastic_active){
                // Device allocations of a decode step come out of the same DRAM (graph and pool growth): arm the
                // reclaimer at every token, and give blocks back first if the floor or the demand grew.
                {std::lock_guard<std::mutex> lock(lifecycle_mutex);reclaimer_armed=!poisoned;}
                reclaim_host(0,"floor");
            }
            grow_after_first_token();watermark_tick();finish_host_probe();allocate_host_scratch();start_host_probe();update_admission();
        }
        if(decode)cold_prefill_admission_phase=adopt_next_decode=false;
    }
    void worker_loop(){
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            cv.wait(lock,[&]{return queued || prefill_queued || host_queued || !upload_jobs.empty() || stopping;});if(stopping)return;
            const bool grouped=prefill_queued;
            const bool compute=queued || grouped;upload_job upload={};
            // Host-memory jobs run after the layer's resident batch, at most one engine batch at a time.
            const bool host=!compute && host_queued;std::vector<host_task> batch;
            if(grouped){prefill_queued=false;if(timing)handoff_us+=micros(launch_time);}
            else if(compute){queued=false;if(timing)handoff_us+=micros(launch_time);}
            else if(host){
                const size_t n=std::min<size_t>(16,host_pending.size());
                batch.assign(host_pending.begin(),host_pending.begin()+n);host_pending.erase(host_pending.begin(),host_pending.begin()+n);
                host_queued=!host_pending.empty();
            }
            else{upload=upload_jobs.front();upload_jobs.pop_front();}
            lock.unlock();
            auto t=clock_type::now();std::string error_text;std::vector<float> batch_down;
            try{
                cuda_check(cudaSetDevice(device));
                if(grouped){run_prefill_worker();}
                else if(compute){char error[512]={};int rc=format_mode?
                    format_api->compute(engine,work.data(),int(work.size()),input.data(),input.size(),&activation,down.data(),error,sizeof(error)):
                    compute_api->compute(engine,work.data(),int(work.size()),input.data(),input.size(),beta,linear_beta,down.data(),error,sizeof(error));
                    if(rc)throw std::runtime_error(error);}
                else if(host){
                    std::vector<vram_compute_expert> experts;experts.reserve(batch.size());
                    // A discrete GPU's jobs read device memory once their copies are done; the call is timed from there.
                    for(const auto & task:batch){if(task.end)cuda_check(cudaEventSynchronize(task.end));experts.push_back({task.device[0],task.device[1],task.device[2]});}
                    t=clock_type::now();
                    batch_down.resize(batch.size()*size_t(latent));
                    char error[512]={};
                    if(format_api->compute(engine,experts.data(),int(experts.size()),input.data(),input.size(),&activation,batch_down.data(),error,sizeof(error)))
                        throw std::runtime_error(error);
                }
                else copy_upload(upload);
            }
            catch(const std::exception & e){error_text=e.what();}
            lock.lock();
            if(compute){
                const double us=micros(t);compute_us+=us;
                // Measured GPU cost of a resident expert, for the sink's CPU/GPU choice (decode batches only).
                if(!grouped && !work.empty()){const double x=us/double(work.size());
                    resident_us_per_expert=resident_us_per_expert>0?0.9*resident_us_per_expert+0.1*x:x;
                    resident_calls.add(double(work.size()),us);}
            }
            if(host){
                for(auto & task:batch){
                    float ms=0;
                    if(task.start && error_text.empty() && cudaEventElapsedTime(&ms,task.start,task.end)==cudaSuccess){
                        const double mb=double(stride)/1e6;host_copy_bytes+=double(stride);host_copy_us+=1e3*ms;locked_copies.add(mb,1e3*double(ms));}
                    if(task.scratch>=0 && size_t(task.scratch)<host_scratch_busy.size())host_scratch_busy[size_t(task.scratch)]=0;
                    return_host_events(task);
                }
            }
            if(host && error_text.empty()){
                const double us=micros(t),x=us/double(batch.size());compute_us+=us;
                host_us_per_expert=host_us_per_expert>0?0.9*host_us_per_expert+0.1*x:x;
                host_calls.add(double(batch.size()),us);
                for(const auto & task:batch)host_done.push_back(task.job);
                host_down.insert(host_down.end(),batch_down.begin(),batch_down.end());
                host_jobs+=batch.size();++host_batches;
            }
            if(!error_text.empty()){if(failure.empty())failure=error_text;queued=false;upload_jobs.clear();
                for(auto & task:host_pending){if(task.scratch>=0 && size_t(task.scratch)<host_scratch_busy.size())host_scratch_busy[size_t(task.scratch)]=0;return_host_events(task);}
                host_pending.clear();host_queued=false;}
            done=!queued && !prefill_queued && !host_queued && upload_jobs.empty();cv.notify_all();
        }
    }

    void run_mxfp4_prefill_worker(){
        struct route { int token; float weight; };
        std::vector<std::vector<route>> by_expert(prefill_slots.size());
        // Build each expert's token list once. Duplicate routes for the same
        // token share one expert computation and have their weights summed.
        for(int token=0;token<prefill_ntokens;++token){
            for(int slot=0;slot<prefill_nslots;++slot){
                const size_t index=size_t(token)*size_t(prefill_nslots)+size_t(slot);
                const int request=prefill_routes[index];
                if(request<0)continue;
                auto & list=by_expert[size_t(request)];
                if(!list.empty() && list.back().token==token)list.back().weight+=prefill_weights[index];
                else list.push_back({token,prefill_weights[index]});
            }
        }
        std::vector<uint8_t> batch_input;
        std::vector<int> token_indices;
        std::vector<float> route_weights;
        std::vector<uint8_t> gpu_used(size_t(prefill_ntokens),0);
        batch_input.reserve(size_t(prefill_ntokens)*prefill_input_stride);
        token_indices.reserve(size_t(prefill_ntokens));
        route_weights.reserve(size_t(prefill_ntokens));
        char error[512]={};
        if(!mxfp4_batch_api->begin_accumulate(mxfp4_batch,prefill_ntokens,error,sizeof(error)))
            throw std::runtime_error(error[0]?error:"MXFP4 accumulation begin failed");
        // The first cold layer creates the batch component after the preflight
        // hook. Its successful allocation is reusable by subsequent layers.
        mxfp4_reserved_accumulator_bytes=std::max(mxfp4_reserved_accumulator_bytes,
            size_t(prefill_ntokens)*size_t(latent)*sizeof(float));
        auto compute_expert=[&](int request,const vram_compute_expert & expert){
            const auto expert_start=clock_type::now();
            batch_input.clear();token_indices.clear();route_weights.clear();
            for(const route & item:by_expert[size_t(request)]){
                if(item.weight==0)continue;
                const uint8_t * row=prefill_input.data()+size_t(item.token)*prefill_input_stride;
                batch_input.insert(batch_input.end(),row,row+prefill_input_stride);
                token_indices.push_back(item.token);
                route_weights.push_back(item.weight);
            }
            if(token_indices.empty())return;
            std::memset(error,0,sizeof(error));
            if(!mxfp4_batch_api->compute_accumulate(mxfp4_batch,expert.up,expert.gate,expert.down,
                    batch_input.data(),token_indices.data(),route_weights.data(),int(token_indices.size()),
                    prefill_activation.swiglu_limit,error,sizeof(error)))
                throw std::runtime_error(error[0]?error:"MXFP4 accumulated compute failed");
            // Per-expert sums for the caller's least-squares fit of us = fixed + per-route * routes: the worker's
            // time from the routes' gather to the computation's synchronisation.
            const double us=micros(expert_start),routes=double(token_indices.size());
            prefill_job_stats.gpu_fit_experts+=1;prefill_job_stats.gpu_fit_routes+=routes;
            prefill_job_stats.gpu_fit_routes_sq+=routes*routes;prefill_job_stats.gpu_fit_us+=us;
            prefill_job_stats.gpu_fit_route_us+=routes*us;
            for(int token:token_indices)gpu_used[size_t(token)]=1;
        };
        // Probe the ring before touching resident work. A read can complete
        // out of route order; consuming it returns its ring credit promptly.
        // The second device slot lets the next upload overlap the previous
        // transient computation on the batch stream.
        const size_t count=prefill_transient.size();
        std::vector<uint8_t> state(count,0); // pending, uploading, ready, computed
        std::vector<int> device_slot(count,-1);
        std::vector<int> free_slots={1,0};
        std::deque<size_t> ready;
        size_t remaining=count,resident=0,ready_probes=0,blocked_reads=0;
        size_t inflight=count;
        clock_type::time_point upload_start;
        double upload_enqueue_us=0;
        auto probe=[&]() -> size_t {
            for(size_t i=0;i<count;++i){
                const auto & job=prefill_transient[i];
                if(state[i]==0 && job.stage && job.stage(job.stage_context,job.request_index,2))return i;
            }
            for(size_t i=0;i<count;++i){
                if(state[i]==0 && !prefill_transient[i].stage)return i;
            }
            return count;
        };
        auto start_upload=[&](size_t i){
            const auto & job=prefill_transient[i];
            const auto read_start=clock_type::now();
            if(job.stage && !job.stage(job.stage_context,job.request_index,0))
                throw std::runtime_error("streamed SSD prefill read failed");
            prefill_job_stats.read_wait_us+=micros(read_start);
            const int slot=free_slots.back();free_slots.pop_back();
            state[i]=1;device_slot[i]=slot;inflight=i;
            upload_start=clock_type::now();
            char * dst=prefill_transient_arena+size_t(slot)*stride;
            for(int role=0;role<3;++role){
                cuda_check(cudaMemcpyAsync(dst,job.role[role],role_bytes[role],cudaMemcpyHostToDevice,upload_stream));
                dst+=role_bytes[role];
            }
            upload_enqueue_us=micros(upload_start);
        };
        auto finish_upload=[&](){
            const auto wait_start=clock_type::now();
            cuda_check(cudaStreamSynchronize(upload_stream));
            const auto & job=prefill_transient[inflight];
            if(job.stage && !job.stage(job.stage_context,job.request_index,1))
                throw std::runtime_error("streamed SSD prefill release failed");
            // Count worker time spent enqueueing and waiting. Compute on the
            // other stream may overlap the device copy between these points.
            const double elapsed=upload_enqueue_us+micros(wait_start);
            prefill_job_stats.transient_upload_us+=elapsed;
            prefill_job_stats.transient_upload_bytes+=stride;
            if(job.source==KIMI_VRAM_PREFILL_TRANSIENT_RAM){
                prefill_job_stats.ram_transient_upload_us+=elapsed;
                prefill_job_stats.ram_transient_upload_bytes+=stride;
            }else{
                prefill_job_stats.ssd_transient_upload_us+=elapsed;
                prefill_job_stats.ssd_transient_upload_bytes+=stride;
            }
            state[inflight]=2;ready.push_back(inflight);
            inflight=count;
        };
        auto compute_ready=[&](){
            const size_t i=ready.front();ready.pop_front();
            char * base=prefill_transient_arena+size_t(device_slot[i])*stride;
            compute_expert(prefill_transient[i].request_index,
                {base,base+role_bytes[0],base+role_bytes[0]+role_bytes[1]});
            free_slots.push_back(device_slot[i]);state[i]=3;--remaining;
        };
        auto compute_resident=[&](){
            while(resident<prefill_slots.size()){
                const size_t request=resident++;
                const int slot=prefill_slots[request];
                if(slot<0)continue;
                char * base=slot_base(size_t(slot));
                compute_expert(int(request),{base,base+role_bytes[0],base+role_bytes[0]+role_bytes[1]});
                return;
            }
        };
        try{
            while(resident<prefill_slots.size() || remaining){
                if(inflight!=count){
                    if(!ready.empty())compute_ready();
                    else compute_resident();
                    finish_upload();
                    continue;
                }
                if(!free_slots.empty()){
                    const size_t i=probe();
                    if(i!=count){++ready_probes;start_upload(i);continue;}
                }
                if(!ready.empty()){compute_ready();continue;}
                if(resident<prefill_slots.size()){compute_resident();continue;}
                if(remaining){
                    size_t i=0;while(i<count && state[i]!=0)++i;
                    if(i==count)throw std::runtime_error("MXFP4 transient scheduler lost a request");
                    ++blocked_reads;start_upload(i);
                }
            }
        }catch(...){
            if(inflight!=count){
                // A ring payload cannot be reused while its H2D copy runs.
                (void)cudaStreamSynchronize(upload_stream);
                const auto & job=prefill_transient[inflight];
                if(job.stage)(void)job.stage(job.stage_context,job.request_index,1);
            }
            throw;
        }
        std::fprintf(stderr,"[kimi_vram_mxfp4_schedule] transient=%zu ready_probes=%zu blocked_reads=%zu slots=%zu "
            "expert_us=%.1f read_wait_us=%.1f\n",
            count,ready_probes,blocked_reads,mxfp4_prefill_transient_slots,prefill_job_stats.gpu_fit_us,
            prefill_job_stats.read_wait_us);
        std::vector<int> output_tokens;
        for(int token=0;token<prefill_ntokens;++token)if(gpu_used[size_t(token)])output_tokens.push_back(token);
        if(!output_tokens.empty()){
            std::memset(error,0,sizeof(error));
            if(!mxfp4_batch_api->finish_accumulate(mxfp4_batch,output_tokens.data(),int(output_tokens.size()),
                    prefill_output.data(),error,sizeof(error)))
                throw std::runtime_error(error[0]?error:"MXFP4 accumulation finish failed");
        }
        std::fprintf(stderr,"[kimi_vram_mxfp4_transfer] layer_tokens=%d gpu_output_tokens=%zu "
            "batch_h2d_bytes=%llu batch_d2h_bytes=%llu\n",prefill_ntokens,output_tokens.size(),
            (unsigned long long)mxfp4_batch_api->h2d_bytes(mxfp4_batch),
            (unsigned long long)mxfp4_batch_api->d2h_bytes(mxfp4_batch));
    }
    void run_prefill_worker(){
        if(!format_mode || !format_api || !engine || !prefill_live)throw std::runtime_error("grouped prefill without an explicit-format engine");
        const auto worker_start=clock_type::now();
        std::fill(prefill_output.begin(),prefill_output.end(),0.0f);
        if(weight_type==int(GGML_TYPE_MXFP4) && mxfp4_batch){
            run_mxfp4_prefill_worker();
            prefill_job_stats.total_worker_us=micros(worker_start);
            return;
        }
        // Existing decode-resident experts stay batched per token, exactly as
        // in the original read-only grouped-prefill path.
        for(int token_index=0;token_index<prefill_ntokens;++token_index){
            work.clear();output_rows.clear();
            for(int route_slot=0;route_slot<prefill_nslots;++route_slot){
                const size_t route=size_t(token_index)*size_t(prefill_nslots)+size_t(route_slot);
                const int request=prefill_routes[route];
                if(request<0)continue;
                if(size_t(request)>=prefill_slots.size())throw std::runtime_error("grouped prefill route index");
                const int slot=prefill_slots[size_t(request)];
                if(slot<0)continue;
                char * base=slot_base(size_t(slot));
                work.push_back({base,base+role_bytes[0],base+role_bytes[0]+role_bytes[1]});
                output_rows.push_back(route);
            }
            if(work.empty())continue;
            down.resize(work.size()*size_t(latent));
            char error[512]={};
            const uint8_t * token_input=prefill_input.data()+size_t(token_index)*prefill_input_stride;
            const int rc=format_api->compute(engine,work.data(),int(work.size()),token_input,prefill_input_stride,
                    &prefill_activation,down.data(),error,sizeof(error));
            if(rc)throw std::runtime_error(error);
            float * destination=prefill_output.data()+size_t(token_index)*size_t(latent);
            // output_rows is in original route-slot order, preserving the GPU
            // subset's deterministic reduction order.
            for(size_t i=0;i<output_rows.size();++i){
                const float weight=prefill_weights[output_rows[i]];
                for(int col=0;col<latent;++col){
                    volatile float scaled=down[i*size_t(latent)+size_t(col)]*weight;
                    destination[col]+=scaled;
                }
            }
        }
        // Selected Q2 RAM experts use one dedicated scratch triplet. Uploads
        // and expert chains are deliberately serialized: the queue is bounded
        // to one scratch expert, payload leases stay valid through finish, and
        // no decode-resident slot is displaced.
        for(const auto & transient:prefill_transient){
            char * dst=prefill_transient_arena;
            const auto upload_start=clock_type::now();
            for(int role=0;role<3;++role){
                cuda_check(cudaMemcpyAsync(dst,transient.role[role],role_bytes[role],cudaMemcpyHostToDevice,upload_stream));
                dst+=role_bytes[role];
            }
            cuda_check(cudaStreamSynchronize(upload_stream));
            const double upload_elapsed=micros(upload_start);
            prefill_job_stats.transient_upload_us+=upload_elapsed;
            prefill_job_stats.transient_upload_bytes+=stride;
            if(transient.source==KIMI_VRAM_PREFILL_TRANSIENT_RAM){
                prefill_job_stats.ram_transient_upload_us+=upload_elapsed;
                prefill_job_stats.ram_transient_upload_bytes+=stride;
            }else{
                prefill_job_stats.ssd_transient_upload_us+=upload_elapsed;
                prefill_job_stats.ssd_transient_upload_bytes+=stride;
            }
            char * base=prefill_transient_arena;
            const vram_compute_expert weights_on_device={base,base+role_bytes[0],base+role_bytes[0]+role_bytes[1]};
            for(int token_index=0;token_index<prefill_ntokens;++token_index){
                output_rows.clear();
                for(int route_slot=0;route_slot<prefill_nslots;++route_slot){
                    const size_t route=size_t(token_index)*size_t(prefill_nslots)+size_t(route_slot);
                    if(prefill_routes[route]==transient.request_index)output_rows.push_back(route);
                }
                if(output_rows.empty())continue;
                down.resize(size_t(latent));
                char error[512]={};
                const uint8_t * token_input=prefill_input.data()+size_t(token_index)*prefill_input_stride;
                const int rc=format_api->compute(engine,&weights_on_device,1,token_input,prefill_input_stride,
                        &prefill_activation,down.data(),error,sizeof(error));
                if(rc)throw std::runtime_error(error);
                float * destination=prefill_output.data()+size_t(token_index)*size_t(latent);
                for(size_t route:output_rows){
                    const float weight=prefill_weights[route];
                    for(int col=0;col<latent;++col){
                        volatile float scaled=down[size_t(col)]*weight;
                        destination[col]+=scaled;
                    }
                }
            }
        }
        prefill_job_stats.total_worker_us=micros(worker_start);
    }
    void upload_loop(){
        std::unique_lock<std::mutex> lock(mutex);
        for(;;){
            auto next=deferred.end();
            upload_cv.wait(lock,[&]{
                if(stopping)return true;
                next=std::find_if(deferred.begin(),deferred.end(),[](const deferred_job & d){return !d.started;});
                return next!=deferred.end();});
            if(stopping)return;
            next->started=true;
            const uint64_t id=next->id;const int slot=next->slot;const void * data[3]={next->data[0],next->data[1],next->data[2]};
            const auto hook=deferred_hook;const auto hook_arg=deferred_hook_arg;
            lock.unlock();
            const auto t=clock_type::now();std::string error_text;bool locked=false;
            try{
                if(hook && !hook(hook_arg))throw std::runtime_error("injected deferred VRAM upload error");
                cuda_check(cudaSetDevice(device));
                if(!host_readable){cudaPointerAttributes attr{};
                    if(cudaPointerGetAttributes(&attr,data[0])==cudaSuccess)locked=attr.type==cudaMemoryTypeHost;else (void)cudaGetLastError();}
                char * dst=slot_base(size_t(slot));
                for(int i=0;i<3;++i){cuda_check(cudaMemcpyAsync(dst,data[i],role_bytes[i],cudaMemcpyHostToDevice,deferred_stream));dst+=role_bytes[i];}
                cuda_check(cudaStreamSynchronize(deferred_stream));
            }
            catch(const std::exception & e){error_text=e.what();}
            lock.lock();
            const auto job=std::find_if(deferred.begin(),deferred.end(),[&](const deferred_job & d){return d.id==id;});
            if(job!=deferred.end())job->completed=true;
            if(error_text.empty()){const double us=micros(t);upload_us+=us;++uploads;upload_bytes+=stride;++borrowed_uploads;
                upload_us_per_expert=upload_us_per_expert>0?0.9*upload_us_per_expert+0.1*us:us;
                if(locked)locked_copies.add(double(stride)/1e6,us);}
            else if(failure.empty())failure=error_text;
            cv.notify_all();
        }
    }
    // The worker and the upload thread run on the CPUs of the thread that creates them. Every sink call that can
    // create the tier pins its caller to the sink's first compute CPU (sink_caller_affinity), so the worker shares
    // that core with the sink thread that waits for it; created from an unpinned thread they ran wherever the
    // scheduler put them, and decode steps 129-256 were 9 % slower on the Spark (ysy9, 10-21). Logged so that a
    // caller that does not pin shows (cpus > 1).
    void report_thread_cpus(){
        cpu_set_t set;CPU_ZERO(&set);
        if(pthread_getaffinity_np(pthread_self(),sizeof(set),&set))return;
        int first=-1;for(int c=0;c<CPU_SETSIZE && first<0;++c)if(CPU_ISSET(c,&set))first=c;
        std::fprintf(stderr,"[kimi_vram_tier] event=threads cpus=%d first_cpu=%d\n",CPU_COUNT(&set),first);
    }
    // The upload's cost before the first admission, so that admissions are priced from the second decode token,
    // once the first token's misses are measured: one slot's roles copied, as an admission copies them, from pageable
    // host memory (where the RAM cache and the read ring keep a payload) into the empty first slot. The second of two
    // copies, past the driver's first-use cost. Measured uploads replace it.
    void probe_upload(){
        if(upload_us_per_expert>0 || !table || table->slots.empty() || !stride)return;
        std::vector<char> payload(stride);
        cuda_check(cudaSetDevice(device));
        double us=0;
        for(int pass=0;pass<2;++pass){
            char * dst=slot_base(0);const char * src=payload.data();const auto t=clock_type::now();
            for(int i=0;i<3;++i){cuda_check(cudaMemcpyAsync(dst,src,role_bytes[i],cudaMemcpyHostToDevice,upload_stream));dst+=role_bytes[i];src+=role_bytes[i];}
            cuda_check(cudaStreamSynchronize(upload_stream));
            us=micros(t);
        }
        upload_us_per_expert=us;
        std::fprintf(stderr,"[kimi_vram_admission] event=upload_probe upload_us=%.1f bytes=%zu\n",us,stride);
    }
    void copy_upload(const upload_job & job){
        char * dst=slot_base(size_t(job.slot));const auto t=clock_type::now();
        for(int i=0;i<3;++i){cuda_check(cudaMemcpyAsync(dst,job.data[i],role_bytes[i],cudaMemcpyHostToDevice,upload_stream));dst+=role_bytes[i];}
        if(!job.borrow)cuda_check(cudaStreamSynchronize(upload_stream));
        const double us=micros(t);std::lock_guard<std::mutex> lock(mutex); // the upload thread updates the same counters
        upload_us+=us;++uploads;upload_bytes+=stride;borrowed_uploads+=job.borrow;
        if(!job.borrow)upload_us_per_expert=upload_us_per_expert>0?0.9*upload_us_per_expert+0.1*us:us;
    }
    void initialize(const kimi_vram_request * r,int l,int h){
        if(format_mode)throw std::runtime_error("VRAM tier already holds an explicit expert format");
        if(l<=0 || h<=0 || l%256 || h%256 || l>16384 || h>16384)throw std::runtime_error("unsupported VRAM expert geometry");
        if(engine && (latent!=l || hidden!=h))throw std::runtime_error("VRAM geometry changed");
        if(!engine){
            latent=l;hidden=h;cuda_check(cudaGetDevice(&device));detect_unified();
            auto get=reinterpret_cast<vram_compute_get_api_fn>(dlsym(RTLD_DEFAULT,"kimi_vram_compute_get_api"));
            if(!get)throw std::runtime_error("test2 parity adapter missing; no fallback");
            compute_api=get();
            if(!compute_api || compute_api->version!=1 || compute_api->size!=sizeof(*compute_api) ||
                !compute_api->create || !compute_api->destroy || !compute_api->compute ||
                !compute_api->device_scratch_bytes || !compute_api->host_scratch_bytes)throw std::runtime_error("parity adapter ABI mismatch");
            char error[512]={};engine=compute_api->create(device,latent,hidden,16,error,sizeof(error));if(!engine)throw std::runtime_error(error);
            worker=std::thread([this]{worker_loop();});
            if(deferred_upload)upload_thread=std::thread([this]{upload_loop();});
            report_thread_cpus();
        }
        if(!upload_stream)cuda_check(cudaStreamCreateWithFlags(&upload_stream,cudaStreamNonBlocking));
        if(deferred_upload && !deferred_stream)cuda_check(cudaStreamCreateWithFlags(&deferred_stream,cudaStreamNonBlocking));
        if(table)return;
        stride=0;for(int i=0;i<3;++i){role_bytes[i]=r->role[i].bytes;stride+=role_bytes[i];}
        const size_t raw=size_t(latent/256)*110*hidden,padded=(raw+4095)&~size_t(4095);
        for(size_t b:role_bytes)if(b!=padded)throw std::runtime_error("VRAM pack padding/shape mismatch");
        size_t free_bytes=0,total=0;mem_info(&free_bytes,&total);
        if(free_bytes<=safety_bytes || std::min<uint64_t>(budget,free_bytes-safety_bytes)<stride)throw std::runtime_error("no VRAM cache capacity after safety reserve");
        std::vector<int> priority;
        const char * p=std::getenv("GGML_KIMI_CPU_SINK_CACHE_LAYER_PRIORITY");
        if(p && *p){std::string s(p);size_t start=0;do{size_t end=s.find(',',start),used=0;std::string item=s.substr(start,end-start);int value=std::stoi(item,&used);if(used!=item.size())throw std::runtime_error("invalid VRAM layer priority");priority.push_back(value);if(end==std::string::npos)break;start=end+1;if(start==s.size())throw std::runtime_error("empty VRAM layer priority");}while(start<s.size());}
        else for(int i=1;i<=92;++i)priority.push_back(i);
        if(priority.size()!=92)throw std::runtime_error("VRAM allocation requires all 92 layer IDs");
        const bool global=env_size("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL",0,1)!=0;
        const size_t count=initial_slots(free_bytes,92*896,global);
        base_slots=count;
        const char * policy=std::getenv("GGML_KIMI_CPU_SINK_VRAM_POLICY");
        if(policy && std::strcmp(policy,"lru") && std::strcmp(policy,"count_decay2"))throw std::runtime_error("invalid VRAM policy");
        table=std::make_unique<vram_slot_table>(count,priority,!policy || std::strcmp(policy,"lru"),env_count(),global,env_half_life());
        configure_elastic(global,92*896);
        memory_sample("before_allocate");
        const auto allocate_start=memory_trace?clock_type::now():clock_type::time_point{};
        if(elastic_active)allocate_base();
        else{capacity_bytes=count*stride;cuda_check(cudaMalloc(reinterpret_cast<void **>(&arena),capacity_bytes));}
        if(memory_trace)memory_sample("after_allocate",micros(allocate_start));
        probe_upload();register_seed();
        arena_epoch=epoch;table->next_token();adopt_next_decode=!active;
        // On unified memory the rest of the first decode step (the RAM cache committing its slots, graph
        // instantiation) comes out of the same DRAM: grow at the token boundary after it (grow_after_first_token,
        // which counts this step when a decode step allocated the tier).
        if(elastic_active && !unified){std::lock_guard<std::mutex> lock(lifecycle_mutex);grow_decode_locked();}
        else if(elastic_active && active)phase_decode_tokens=1;
        report("allocated");
    }
    void prepare(const kimi_vram_request * r,int count,int l,int h,int * slots){
        if(count<0 || count>16 || (count && (!r || !slots)))throw std::runtime_error("VRAM request count");
        if(count)std::fill(slots,slots+count,-1);
        if(!active || !count)return;
        initialize(r,l,h);
        for(int j=0;j<count;++j){
            if(!std::isfinite(r[j].weight) || r[j].weight<0)throw std::runtime_error("VRAM routing weight");
            for(int i=0;i<3;++i)if(r[j].role[i].bytes!=role_bytes[i])throw std::runtime_error("VRAM role bytes changed");
        }
        table->prepare(r,count);std::copy(table->hits.begin(),table->hits.end(),slots);
        for(int slot:table->hits){gpu_hits+=slot>=0;cpu_requests+=slot<0;}
    }
    void launch(const void * q,size_t bytes,float b,float lb){
        if(!active || !table || !table->in_flight)return;
        if(format_mode)throw std::runtime_error("Q3_K/SiTU launch on an explicit-format VRAM tier");
        if(bytes!=size_t(latent/256)*292 || !q || !std::isfinite(b) || b<=0 || !std::isfinite(lb))throw std::runtime_error("VRAM Q8_K input or SiTU parameter");
        std::lock_guard<std::mutex> lock(mutex);if(!done)throw std::runtime_error("GPU job already outstanding");
        work.clear();output_rows.clear();failure.clear();
        for(size_t j=0;j<table->hits.size();++j){
            int slot=table->hits[j];if(slot<0)continue;char * base=slot_base(size_t(slot));
            work.push_back({base,base+role_bytes[0],base+role_bytes[0]+role_bytes[1]});output_rows.push_back(j);
        }
        if(work.empty())return;
        input.assign(static_cast<const uint8_t *>(q),static_cast<const uint8_t *>(q)+bytes);
        down.resize(work.size()*size_t(latent));beta=b;linear_beta=lb;done=false;queued=true;
        if(timing){launch_time=clock_type::now();++launches;}
        cv.notify_all();
    }
    // Explicit-format path: geometry from the expert tensors and the owner's
    // layer list, engine from kimi_vram_format_compute_get_api. The slot table,
    // policies, uploads, residency and leases are the same as the Q3_K path.
    void initialize_format(const kimi_vram_request * r,const kimi_vram_expert_format & f,bool allocate_cache=true){
        if(!f.layers || f.n_layers<=0 || f.n_layers>1024 || f.n_experts<=0 || f.n_experts>65536 ||
                f.latent<=0 || f.hidden<=0 || f.latent>16384 || f.hidden>16384 ||
                f.weight_type<0 || f.weight_type>=int(GGML_TYPE_COUNT) || f.down_type<0 || f.down_type>=int(GGML_TYPE_COUNT))
            throw std::runtime_error("invalid VRAM expert format");
        const bool q2=f.weight_type==int(GGML_TYPE_IQ2_XXS) && f.down_type==int(GGML_TYPE_Q2_K);
        const bool mxfp4=f.weight_type==int(GGML_TYPE_MXFP4) && f.down_type==int(GGML_TYPE_MXFP4);
        if(engine && (!format_mode || latent!=f.latent || hidden!=f.hidden || weight_type!=f.weight_type ||
                down_type!=f.down_type || n_experts!=f.n_experts))throw std::runtime_error("VRAM expert format changed");
        if(!engine){
            cuda_check(cudaGetDevice(&device));detect_unified();
            // MXFP4 has its own engine; the generic one serves every format the CUDA backend has a
            // matrix-vector kernel for, and refuses the others when it is created.
            if(!mxfp4){
                if(!kimi_vram_q2_format_compute_get_api)
                    throw std::runtime_error("generic explicit-format VRAM engine missing; no fallback");
                format_api=kimi_vram_q2_format_compute_get_api();
            }
            else{
                auto get=reinterpret_cast<vram_format_compute_get_api_fn>(dlsym(RTLD_DEFAULT,"kimi_vram_format_compute_get_api"));
                if(!get)throw std::runtime_error("MXFP4 explicit-format VRAM engine missing; no fallback");
                format_api=get();
            }
            if(!format_api || format_api->version!=1 || format_api->size!=sizeof(*format_api) || !format_api->create ||
                    !format_api->destroy || !format_api->compute || !format_api->device_scratch_bytes || !format_api->host_scratch_bytes)
                throw std::runtime_error("explicit-format VRAM engine ABI mismatch");
            char error[512]={};
            engine=format_api->create(device,f.latent,f.hidden,16,f.weight_type,f.down_type,error,sizeof(error));
            if(!engine)throw std::runtime_error(error);
            latent=f.latent;hidden=f.hidden;weight_type=f.weight_type;down_type=f.down_type;n_experts=f.n_experts;format_mode=true;
            worker=std::thread([this]{worker_loop();});
            if(deferred_upload)upload_thread=std::thread([this]{upload_loop();});
            report_thread_cpus();
        }
        if(!upload_stream)cuda_check(cudaStreamCreateWithFlags(&upload_stream,cudaStreamNonBlocking));
        if(deferred_upload && !deferred_stream)cuda_check(cudaStreamCreateWithFlags(&deferred_stream,cudaStreamNonBlocking));
        if(table)return;
        stride=0;for(int i=0;i<3;++i){role_bytes[i]=r->role[i].bytes;stride+=role_bytes[i];}
        const size_t up_bytes=(ggml_row_size(ggml_type(weight_type),latent)*size_t(hidden)+4095)&~size_t(4095);
        const size_t down_bytes=(ggml_row_size(ggml_type(down_type),hidden)*size_t(latent)+4095)&~size_t(4095);
        if(role_bytes[0]!=up_bytes || role_bytes[1]!=up_bytes || role_bytes[2]!=down_bytes)throw std::runtime_error("VRAM pack padding/shape mismatch");
        // Allocate fixed batch scratch before the decode arena. The token
        // accumulator is reserved for each prefill and released before decode.
        if(mxfp4 && mxfp4_batch_enabled && !mxfp4_batch){
            auto get=reinterpret_cast<decltype(&kimi_mxfp4_batch_get_api)>(
                dlsym(RTLD_DEFAULT,"kimi_mxfp4_batch_get_api"));
            if(!get)throw std::runtime_error("MXFP4 batch component missing");
            mxfp4_batch_api=get();
            if(!mxfp4_batch_api || mxfp4_batch_api->version!=3 ||
                    mxfp4_batch_api->size!=sizeof(*mxfp4_batch_api))
                throw std::runtime_error("MXFP4 batch ABI mismatch");
            char error[512]={};
            mxfp4_batch=mxfp4_batch_api->create(device,latent,hidden,256,error,sizeof(error));
            if(!mxfp4_batch)throw std::runtime_error(error[0]?error:"MXFP4 batch scratch allocation failed");
            std::fprintf(stderr,"[kimi_vram_mxfp4_batch] status=ready device_bytes=%zu host_bytes=%zu\n",
                mxfp4_batch_api->device_bytes(mxfp4_batch),mxfp4_batch_api->host_bytes(mxfp4_batch));
        }
        // Transient scratch is outside decode residency. MXFP4 uses two slots
        // during prefill, then returns both before decode cache growth. Failure
        // is a safe CPU fallback reported by transient_available().
        if(((q2 && (prefill_transient_min_routes || prefill_ssd_transient_min_routes)) ||
                (mxfp4 && mxfp4_batch && prefill_transient_phase)) && !prefill_transient_arena){
            allocate_prefill_transient(mxfp4?mxfp4_prefill_transient_slots:1);
        }
        // Cold MXFP4 prefill only needs the transient expert and batch
        // workspace. Defer the large decode arena until the graph has created
        // its cuBLAS handles and finished capture.
        if(!allocate_cache)return;
        size_t free_bytes=0,total=0;mem_info(&free_bytes,&total);
        if(free_bytes<=safety_bytes || std::min<uint64_t>(budget,free_bytes-safety_bytes)<stride)throw std::runtime_error("no VRAM cache capacity after safety reserve");
        const std::vector<int> priority(f.layers,f.layers+f.n_layers);
        const size_t logical_max=size_t(priority.size())*size_t(n_experts);
        const bool global=env_size("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL",0,1)!=0;
        const size_t count=initial_slots(free_bytes,logical_max,global);
        base_slots=count;
        const char * policy=std::getenv("GGML_KIMI_CPU_SINK_VRAM_POLICY");
        if(policy && std::strcmp(policy,"lru") && std::strcmp(policy,"count_decay2"))throw std::runtime_error("invalid VRAM policy");
        table=std::make_unique<vram_slot_table>(count,priority,!policy || std::strcmp(policy,"lru"),env_count(),
                global,env_half_life(),0,1023,n_experts);
        configure_elastic(global,logical_max);
        memory_sample("before_allocate");
        const auto allocate_start=memory_trace?clock_type::now():clock_type::time_point{};
        if(elastic_active)allocate_base();
        else{capacity_bytes=count*stride;cuda_check(cudaMalloc(reinterpret_cast<void **>(&arena),capacity_bytes));}
        if(memory_trace)memory_sample("after_allocate",micros(allocate_start));
        probe_upload();register_seed();
        arena_epoch=epoch;table->next_token();adopt_next_decode=!active;
        // On unified memory the rest of the first decode step (the RAM cache committing its slots, graph
        // instantiation) comes out of the same DRAM: grow at the token boundary after it (grow_after_first_token,
        // which counts this step when a decode step allocated the tier).
        if(elastic_active && !unified){std::lock_guard<std::mutex> lock(lifecycle_mutex);grow_decode_locked();}
        else if(elastic_active && active)phase_decode_tokens=1;
        report("allocated");
    }
    void prepare_format(const kimi_vram_request * r,int count,const kimi_vram_expert_format * f,int * slots){
        if(count<0 || count>16 || (count && (!r || !slots)) || !f)throw std::runtime_error("VRAM request count");
        if(count)std::fill(slots,slots+count,-1);
        if(!active || !count)return;
        initialize_format(r,*f);
        for(int j=0;j<count;++j){
            if(!std::isfinite(r[j].weight) || r[j].weight<0)throw std::runtime_error("VRAM routing weight");
            for(int i=0;i<3;++i)if(r[j].role[i].bytes!=role_bytes[i])throw std::runtime_error("VRAM role bytes changed");
        }
        table->prepare(r,count);std::copy(table->hits.begin(),table->hits.end(),slots);
        for(int slot:table->hits){gpu_hits+=slot>=0;cpu_requests+=slot<0;}
    }
    void launch_activation(const void * q,size_t bytes,const kimi_vram_activation * a){
        if(!active || !table || !table->in_flight)return;
        if(!format_mode || !q || !a)throw std::runtime_error("explicit-format VRAM launch without its format");
        // the generic engine takes the f32 input; MXFP4's takes it quantized to its vec_dot type
        const bool mxfp4=weight_type==int(GGML_TYPE_MXFP4) && down_type==int(GGML_TYPE_MXFP4);
        const auto * traits=ggml_get_type_traits_cpu(ggml_type(weight_type));
        const size_t expected=!mxfp4?size_t(latent)*sizeof(float):
            (traits?ggml_row_size(traits->vec_dot_type,latent):0);
        if(!traits || bytes!=expected)throw std::runtime_error("VRAM input bytes");
        if(a->kind!=KIMI_VRAM_ACTIVATION_SITU && a->kind!=KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP)throw std::runtime_error("VRAM activation kind");
        std::lock_guard<std::mutex> lock(mutex);if(!done)throw std::runtime_error("GPU job already outstanding");
        work.clear();output_rows.clear();failure.clear();host_done.clear();host_down.clear();
        for(size_t j=0;j<table->hits.size();++j){
            int slot=table->hits[j];if(slot<0)continue;char * base=slot_base(size_t(slot));
            work.push_back({base,base+role_bytes[0],base+role_bytes[0]+role_bytes[1]});output_rows.push_back(j);
        }
        // Host-memory jobs of this layer may follow (in place, or copied on a discrete GPU); they need the input even
        // when nothing is resident.
        layer_input_ready=true;
        input.assign(static_cast<const uint8_t *>(q),static_cast<const uint8_t *>(q)+bytes);activation=*a;
        if(work.empty())return;
        down.resize(work.size()*size_t(latent));done=false;queued=true;
        if(timing){launch_time=clock_type::now();++launches;}
        cv.notify_all();
    }
    void admit(const kimi_vram_request & r,const void * const * data,bool borrow){
        if(!active || !table || !table->in_flight)return;
        for(size_t j=0;j<table->admissions.size();++j){
            int slot=table->admissions[j];if(slot<0 || table->slots[slot].valid || table->slots[slot].uploading || std::find(pending_uploads.begin(),pending_uploads.end(),j)!=pending_uploads.end() || !vram_slot_table::same(r,table->requests[j]))continue;
            for(int i=0;i<3;++i)if(!data[i])throw std::runtime_error("null VRAM admission source");
            upload_job job{j,slot,{data[0],data[1],data[2]},borrow};
            if(async_upload){
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if(!failure.empty())throw std::runtime_error(failure);
                    if(pending_uploads.size()>=16)throw std::runtime_error("VRAM upload queue capacity");
                    upload_jobs.push_back(job);pending_uploads.push_back(j);done=false;
                }
                cv.notify_all();
                // A ring pointer has no lifetime beyond admit. RAM pointers
                // are leased through finish, so the CPU need not wait here.
                if(!borrow){drain();if(!failure.empty())throw std::runtime_error(failure);}
            }else{
                cuda_check(cudaSetDevice(device));copy_upload(job);
                if(borrow)pending_uploads.push_back(j);else table->publish(j);
            }
            break;
        }
    }
    void launch_host(const kimi_vram_host_job * jobs,int count,int * accepted){
        if(!accepted)throw std::runtime_error("null host-memory VRAM acceptance");
        *accepted=0;
        if(count<=0 || !jobs || !active || !table || !table->in_flight || !layer_input_ready || !format_mode)return;
        for(int i=0;i<count;++i){
            if(jobs[i].row<0 || jobs[i].row>=int(table->requests.size()) || !std::isfinite(jobs[i].weight) || jobs[i].weight<0)
                throw std::runtime_error("invalid host-memory VRAM job");
            for(int role=0;role<3;++role)if(!jobs[i].role[role])throw std::runtime_error("null host-memory VRAM payload");
            for(int k=0;k<i;++k)if(jobs[k].row==jobs[i].row)throw std::runtime_error("duplicate host-memory VRAM job");
        }
        std::vector<host_task> tasks(static_cast<size_t>(count));
        for(int i=0;i<count;++i)tasks[size_t(i)]={jobs[i],{jobs[i].role[0],jobs[i].role[1],jobs[i].role[2]},-1,nullptr,nullptr};
        if(!host_readable && !copy_host_jobs(tasks))return;
        {
            std::lock_guard<std::mutex> lock(mutex);
            if(!failure.empty())throw std::runtime_error(failure);
            host_pending.insert(host_pending.end(),tasks.begin(),tasks.end());host_queued=true;done=false;
        }
        cv.notify_all();*accepted=1;
    }
    // A discrete GPU: every role page-locked, a device target for each job (its admission reservation, else a free
    // scratch slot), then the copies on the host copy stream, timed by a pair of events. False, with nothing copied,
    // when a source is pageable or no target is free; a batch that found no scratch slot asks for them at the next
    // token boundary.
    bool copy_host_jobs(std::vector<host_task> & tasks){
        cuda_check(cudaSetDevice(device));
        for(const auto & task:tasks)for(int role=0;role<3;++role){
            cudaPointerAttributes attr{};
            if(cudaPointerGetAttributes(&attr,task.job.role[role])!=cudaSuccess){(void)cudaGetLastError();++host_refused;return false;}
            if(attr.type!=cudaMemoryTypeHost){++host_refused;return false;}
        }
        std::vector<int> reserved(tasks.size(),-1),scratch(tasks.size(),-1);
        {
            std::lock_guard<std::mutex> lock(mutex);
            size_t free_scratch=0;for(uint8_t busy:host_scratch_busy)free_scratch+=!busy;
            for(size_t i=0;i<tasks.size();++i){
                const size_t j=size_t(tasks[i].job.row);const int slot=j<table->admissions.size()?table->admissions[j]:-1;
                if(slot>=0 && !table->slots[size_t(slot)].valid && !table->slots[size_t(slot)].uploading &&
                        std::find(pending_uploads.begin(),pending_uploads.end(),j)==pending_uploads.end()){reserved[i]=slot;continue;}
                if(!free_scratch){if(!host_scratch && !host_scratch_failed)host_scratch_wanted=true;++host_refused;return false;}
                --free_scratch;
            }
            for(size_t i=0;i<tasks.size();++i){
                if(reserved[i]>=0)continue;
                for(size_t k=0;k<host_scratch_busy.size();++k)if(!host_scratch_busy[k]){host_scratch_busy[k]=1;scratch[i]=int(k);break;}
            }
            for(size_t i=0;i<tasks.size();++i)if(reserved[i]>=0){pending_uploads.push_back(size_t(tasks[i].job.row));++host_reserved_copies;}
            while(host_events.size()<2*tasks.size()){cudaEvent_t e=nullptr;cuda_check(cudaEventCreate(&e));host_events.push_back(e);}
            for(auto & task:tasks){task.start=host_events.back();host_events.pop_back();task.end=host_events.back();host_events.pop_back();}
        }
        if(!host_copy_stream)cuda_check(cudaStreamCreateWithFlags(&host_copy_stream,cudaStreamNonBlocking));
        for(size_t i=0;i<tasks.size();++i){
            auto & task=tasks[i];task.scratch=scratch[i];
            char * dst=reserved[i]>=0?slot_base(size_t(reserved[i])):host_scratch+size_t(scratch[i])*stride;
            cuda_check(cudaEventRecord(task.start,host_copy_stream));
            for(int role=0;role<3;++role){
                cuda_check(cudaMemcpyAsync(dst,task.job.role[role],role_bytes[role],cudaMemcpyHostToDevice,host_copy_stream));
                task.device[role]=dst;dst+=role_bytes[role];
            }
            cuda_check(cudaEventRecord(task.end,host_copy_stream));
            ++host_copies;
        }
        return true;
    }
    // mutex held, or no other thread
    void return_host_events(host_task & task){
        if(task.start)host_events.push_back(task.start);
        if(task.end)host_events.push_back(task.end);
        task.start=task.end=nullptr;
    }
    // At a decode token boundary, after a batch found no scratch slot: two slots of device memory. An elastic tier
    // that grew into all free memory gives back its last block first (the free memory it reports may not be
    // allocatable: watermark growth fails the same way), and grows back into what is left at later boundaries.
    // Tried once per prefill-to-decode cycle; a failure leaves host jobs to their reservations.
    void allocate_host_scratch(){
        {std::lock_guard<std::mutex> lock(mutex);if(!host_scratch_wanted || host_scratch || host_readable || !stride)return;}
        cuda_check(cudaSetDevice(device));
        const size_t bytes=stride*host_scratch_slots;
        auto attempt=[&]{
            if(cudaMalloc(reinterpret_cast<void **>(&host_scratch),bytes)==cudaSuccess)return true;
            (void)cudaGetLastError();host_scratch=nullptr;return false;
        };
        bool ok=attempt();
        if(!ok && elastic_active){
            std::lock_guard<std::mutex> lock(lifecycle_mutex);
            if(busy())return; // a later boundary tries again
            while(!ok && blocks.size()>1 && reclaim_last_block("host_scratch",false))ok=attempt();
        }
        {std::lock_guard<std::mutex> lock(mutex);host_scratch_wanted=false;host_scratch_failed=!ok;if(ok)host_scratch_busy.assign(host_scratch_slots,0);}
        std::fprintf(stderr,"[kimi_vram_host] event=scratch status=%s slots=%zu bytes=%zu tier_slots=%zu\n",ok?"ready":"unavailable",
            host_scratch_slots,bytes,slot_locations.size());
    }
    void free_host_scratch(){
        finish_host_probe();
        if(!host_scratch)return;
        cuda_check(cudaSetDevice(device));
        if(host_copy_stream)cuda_check(cudaStreamSynchronize(host_copy_stream));
        cuda_check(cudaFree(host_scratch));host_scratch=nullptr;
        {std::lock_guard<std::mutex> lock(mutex);host_scratch_busy.clear();}
        host_scratch_failed=false;
    }
    // A page-locked range was registered (register_host): the first one measures the copy rate unless a copy already
    // has. Any thread.
    void host_registered(const void * pointer,size_t bytes){
        std::lock_guard<std::mutex> lock(mutex);
        if(host_readable || host_probe_taken || !locked_copies.s.empty())return;
        host_probe_taken=true;host_probe=pointer;host_probe_bytes=bytes;
        if(!host_scratch && !host_scratch_failed)host_scratch_wanted=true;
    }
    // Before a range is unregistered: a probe not started from it is dropped, one in flight is waited for. Sink thread.
    void host_unregistering(const void * pointer){
        bool wait=false;
        {std::lock_guard<std::mutex> lock(mutex);
            if(host_probe!=pointer)return;
            if(host_probe_start)wait=true;else host_probe=nullptr;}
        if(wait && host_copy_stream){cuda_check(cudaSetDevice(device));cuda_check(cudaStreamSynchronize(host_copy_stream));}
    }
    // At a decode token boundary, after the scratch allocation: the probe's copy, one scratch slot's bytes at most.
    void start_host_probe(){
        const void * p=nullptr;size_t bytes=0;
        {std::lock_guard<std::mutex> lock(mutex);
            if(!host_probe || host_probe_start || !host_scratch || !locked_copies.s.empty())return;
            p=host_probe;bytes=std::min(host_probe_bytes,stride);}
        cuda_check(cudaSetDevice(device));
        if(!host_copy_stream)cuda_check(cudaStreamCreateWithFlags(&host_copy_stream,cudaStreamNonBlocking));
        cudaEvent_t a=nullptr,b=nullptr;cuda_check(cudaEventCreate(&a));cuda_check(cudaEventCreate(&b));
        cuda_check(cudaEventRecord(a,host_copy_stream));
        cuda_check(cudaMemcpyAsync(host_scratch,p,bytes,cudaMemcpyHostToDevice,host_copy_stream));
        cuda_check(cudaEventRecord(b,host_copy_stream));
        std::lock_guard<std::mutex> lock(mutex);host_probe_start=a;host_probe_end=b;host_probe_copied=bytes;
    }
    // At the next boundary (or at release): the probe's time is the first page-locked copy sample.
    void finish_host_probe(){
        cudaEvent_t a=nullptr,b=nullptr;size_t bytes=0;
        {std::lock_guard<std::mutex> lock(mutex);a=host_probe_start;b=host_probe_end;bytes=host_probe_copied;}
        if(!b)return;
        cuda_check(cudaSetDevice(device));cuda_check(cudaEventSynchronize(b));
        float ms=0;const bool timed=cudaEventElapsedTime(&ms,a,b)==cudaSuccess;if(!timed)(void)cudaGetLastError();
        cuda_check(cudaEventDestroy(a));cuda_check(cudaEventDestroy(b));
        std::lock_guard<std::mutex> lock(mutex);
        host_probe_start=host_probe_end=nullptr;host_probe=nullptr;
        if(timed && ms>0){locked_copies.add(double(bytes)/1e6,1e3*double(ms));
            std::fprintf(stderr,"[kimi_vram_host] event=probe_copy bytes=%zu us=%.0f gbs=%.2f\n",bytes,1e3*double(ms),double(bytes)/1e6/double(ms));}
    }
    // The measured lines (kimi_vram_host_costs), mutex held. The residents' call: the least-squares line of its time
    // against its experts (the residents of a layer vary, 1 to 6 on DeepSeek); before that varies, the shape the
    // engine has on an RTX 5090 (150 + 20 us an expert, unifybench) at the measured level. A host call runs the same
    // engine: its own time is the residents' call, and the rest per expert, fitted through the origin (in place the
    // transfer is in it). On a discrete GPU, the device targets this layer's jobs can take.
    void costs2(kimi_vram_host_costs & c){
        c={};
        kimi_prefill_fit r;for(const auto & x:resident_calls.s)r.add(x.first,x.second);
        double a=0,b=0;kimi_prefill_fit_line(r,150.0,20.0,a,b);
        c.resident_call_us=a;c.resident_expert_us=b;
        double kk=0,ky=0;for(const auto & x:host_calls.s){kk+=x.first*x.first;ky+=x.first*std::max(0.0,x.second-a);}
        c.host_call_us=host_calls.s.empty()?0.0:a;c.host_expert_us=kk>0?ky/kk:0.0;
        double mb=0,us=0;for(const auto & x:locked_copies.s){mb+=x.first;us+=x.second;}
        c.copy_us_per_mb=host_readable || !(mb>0)?0.0:us/mb;
        size_t copying=0;for(const auto & d:deferred)copying+=!d.completed;
        c.link_busy_us=double(copying)*double(stride)/1e6*(c.copy_us_per_mb>0?c.copy_us_per_mb:0.0);
        c.in_place=host_readable;
        if(host_readable)return;
        size_t free_scratch=0;for(uint8_t busy:host_scratch_busy)free_scratch+=!busy;
        c.slots=int32_t(host_scratch?free_scratch:host_scratch_failed?0:host_scratch_slots);
        // the reservations copy_host_jobs would copy into
        if(table && table->in_flight)for(size_t j=0;j<table->requests.size() && j<64;++j){
            const int slot=j<table->admissions.size()?table->admissions[j]:-1;
            if(slot>=0 && !table->slots[size_t(slot)].valid && !table->slots[size_t(slot)].uploading &&
                    std::find(pending_uploads.begin(),pending_uploads.end(),j)==pending_uploads.end())c.reserved_rows|=uint64_t(1)<<j;
        }
    }
    void finish(float * output,size_t row_stride){
        if(!active || !table || !table->in_flight)return;
        const auto finish_start=timing?clock_type::now():clock_type::time_point{};
        drain();if(timing)finish_drain_us+=micros(finish_start);
        if(!failure.empty())throw std::runtime_error(failure);
        memory_sample();
        for(size_t j:pending_uploads)table->publish(j);
        pending_uploads.clear();
        if(!work.empty() && (!output || row_stride<size_t(latent)))throw std::runtime_error("VRAM output geometry");
        for(size_t i=0;i<work.size();++i){size_t row=output_rows[i];float weight=table->requests[row].weight;
            for(int col=0;col<latent;++col){volatile float scaled=down[i*latent+col]*weight;output[row*row_stride+col]=scaled;}
        }
        if(!host_done.empty() && (!output || row_stride<size_t(latent)))throw std::runtime_error("VRAM output geometry");
        for(size_t i=0;i<host_done.size();++i){const size_t row=size_t(host_done[i].row);const float weight=host_done[i].weight;
            for(int col=0;col<latent;++col){volatile float scaled=host_down[i*latent+col]*weight;output[row*row_stride+col]=scaled;}
        }
        host_done.clear();host_down.clear();layer_input_ready=false;
        {std::lock_guard<std::mutex> lock(mutex);std::fill(host_scratch_busy.begin(),host_scratch_busy.end(),0);}
        table->finish();work.clear();output_rows.clear();
    }
    void prepare_prefill(const kimi_vram_request * r,int count,const kimi_vram_expert_format * f,int * slots){
        if(count<0 || (count && (!r || !slots)) || !f || f->n_experts<=0 || count>f->n_experts)
            throw std::runtime_error("invalid grouped-prefill request union");
        if(count)std::fill(slots,slots+count,-1);
        if(prefill_live || prefill_queued || active || have_token)
            throw std::runtime_error("grouped prefill outside a retained prefill phase");
        if(!keep_across_requests || !count)return;
        if(unified)reclaim_host(0,"floor");
        const bool q2=f->weight_type==int(GGML_TYPE_IQ2_XXS) && f->down_type==int(GGML_TYPE_Q2_K);
        if(!has_storage() || !table || !engine){
            if(!q2 && !(f->weight_type==int(GGML_TYPE_MXFP4) && mxfp4_batch_enabled))return;
            if(q2 && !cold_prefill_admit_per_layer && !prefill_ssd_transient_min_routes)return;
            // The graph has already reserved its actual request-sized
            // workspace. Allocate the decode arena from the remaining VRAM,
            // then seed only empty slots as selected SSD experts arrive.
            initialize_format(r,*f,q2);
            // The immediately following decode adopts an arena initialized
            // here (initialize_format, outside decode); only Q2 admits into it.
            cold_prefill_admission_phase=q2;
            if(q2 && cold_prefill_admit_per_layer)std::fprintf(stderr,
                "[kimi_vram_prefill_cold_admission] event=armed per_layer=%zu slots=%zu zero_decode_heat=1 evict_existing=0\n",
                cold_prefill_admit_per_layer,table?table->slots.size():0);
        }
        if(!format_mode || latent!=f->latent || hidden!=f->hidden || weight_type!=f->weight_type ||
                down_type!=f->down_type || n_experts!=f->n_experts)
            throw std::runtime_error("grouped-prefill expert format changed");
        for(int j=0;j<count;++j){
            if(r[j].layer<0 || r[j].expert<0 || r[j].expert>=n_experts)
                throw std::runtime_error("invalid grouped-prefill expert key");
            for(int role=0;role<3;++role)if(r[j].role[role].bytes!=role_bytes[role])
                throw std::runtime_error("grouped-prefill role bytes changed");
        }
        prefill_slots.resize(size_t(count));
        if(!table){
            std::fill(prefill_slots.begin(),prefill_slots.end(),-1);
            cpu_requests+=size_t(count);
            return;
        }
        prefill_generation=table->acquire_slots(r,size_t(count),prefill_slots.data());
        std::copy(prefill_slots.begin(),prefill_slots.end(),slots);
        const size_t hits=size_t(std::count_if(prefill_slots.begin(),prefill_slots.end(),[](int slot){return slot>=0;}));
        gpu_hits+=hits;cpu_requests+=size_t(count)-hits;
        if(!hits){table->release_ready(prefill_generation);prefill_generation=0;return;}
        prefill_live=true;
    }
    int transient_prefill_available() const{
        const bool q2=weight_type==int(GGML_TYPE_IQ2_XXS) && down_type==int(GGML_TYPE_Q2_K);
        const bool mxfp4=weight_type==int(GGML_TYPE_MXFP4) && down_type==int(GGML_TYPE_MXFP4) && mxfp4_batch;
        return format_mode && prefill_transient_arena &&
            ((q2 && prefill_transient_bytes==stride) ||
             (mxfp4 && prefill_transient_bytes==stride*mxfp4_prefill_transient_slots));
    }
    int cold_prefill_admission_active() const{
        return cold_prefill_admission_phase && format_mode && table &&
            weight_type==int(GGML_TYPE_IQ2_XXS) && down_type==int(GGML_TYPE_Q2_K);
    }
    void admit_cold_prefill(const kimi_vram_request & request,const void * const * data,int * admitted_result){
        if(!admitted_result)throw std::runtime_error("null cold-prefill admission result");
        *admitted_result=0;
        if(!cold_prefill_admission_active())return;
        for(int role=0;role<3;++role){
            if(!data[role] || request.role[role].bytes!=role_bytes[role])
                throw std::runtime_error("invalid cold-prefill admission payload");
        }
        ++cold_prefill_admit_calls;
        const int slot=table->reserve_cold_prefill(request);
        if(slot<0)return;
        const auto copy_start=clock_type::now();
        try{
            if(fail_cold_prefill_admit_at && cold_prefill_admit_calls==fail_cold_prefill_admit_at)
                throw std::runtime_error("injected cold-prefill VRAM upload error");
            cuda_check(cudaSetDevice(device));
            char * destination=slot_base(size_t(slot));
            for(int role=0;role<3;++role){
                cuda_check(cudaMemcpyAsync(destination,data[role],role_bytes[role],cudaMemcpyHostToDevice,upload_stream));
                destination+=role_bytes[role];
            }
            cuda_check(cudaStreamSynchronize(upload_stream));
            const double elapsed=micros(copy_start);
            table->publish_cold_prefill(size_t(slot),request);
            ++cold_prefill_admissions;++uploads;
            cold_prefill_admission_bytes+=stride;upload_bytes+=stride;
            cold_prefill_admission_us+=elapsed;upload_us+=elapsed;
            *admitted_result=1;
        }catch(...){
            table->cancel_cold_prefill(size_t(slot),request);
            throw;
        }
    }
    void launch_prefill(const void * q,size_t input_stride,const int32_t * routes,const float * weights,
            int nslots,int ntokens,const kimi_vram_prefill_transient * transient,int transient_count,
            const kimi_vram_activation * a){
        if(transient_count<0 || (transient_count && !transient))throw std::runtime_error("invalid grouped-prefill transient list");
        if(!prefill_live && !transient_count)return;
        if(!q || !routes || !weights || !a || nslots<0 || nslots>16 || ntokens<1 ||
                (nslots && ntokens>std::numeric_limits<int>::max()/nslots))
            throw std::runtime_error("invalid grouped-prefill launch");
        const bool q2=weight_type==int(GGML_TYPE_IQ2_XXS) && down_type==int(GGML_TYPE_Q2_K);
        const auto * traits=ggml_get_type_traits_cpu(ggml_type(weight_type));
        const size_t expected=q2?size_t(latent)*sizeof(float):
            (traits?ggml_row_size(traits->vec_dot_type,latent):0);
        if(!traits || input_stride!=expected ||
                (a->kind!=KIMI_VRAM_ACTIVATION_SITU && a->kind!=KIMI_VRAM_ACTIVATION_SWIGLU_CLAMP))
            throw std::runtime_error("grouped-prefill input or activation format");
        const size_t route_count=size_t(nslots)*size_t(ntokens);
        for(size_t i=0;i<route_count;++i){
            if(routes[i]<-1 || (routes[i]>=0 && size_t(routes[i])>=prefill_slots.size()) ||
                    !std::isfinite(weights[i]) || weights[i]<0)
                throw std::runtime_error("invalid grouped-prefill route");
        }
        if(transient_count){
            if(!transient_prefill_available())
                throw std::runtime_error("grouped-prefill transient scratch is unavailable");
            for(int i=0;i<transient_count;++i){
                const int request=transient[i].request_index;
                if(request<0 || size_t(request)>=prefill_slots.size() || prefill_slots[size_t(request)]>=0)
                    throw std::runtime_error("invalid grouped-prefill transient request");
                if(transient[i].source!=KIMI_VRAM_PREFILL_TRANSIENT_RAM &&
                        transient[i].source!=KIMI_VRAM_PREFILL_TRANSIENT_SSD)
                    throw std::runtime_error("invalid grouped-prefill transient source");
                for(int role=0;role<3;++role)if(!transient[i].role[role])
                    throw std::runtime_error("null grouped-prefill transient payload");
                for(int j=0;j<i;++j)if(transient[j].request_index==request)
                    throw std::runtime_error("duplicate grouped-prefill transient request");
            }
        }
        if(size_t(ntokens)>std::numeric_limits<size_t>::max()/input_stride ||
                size_t(ntokens)>std::numeric_limits<size_t>::max()/size_t(latent))
            throw std::runtime_error("grouped-prefill workspace overflow");
        std::lock_guard<std::mutex> lock(mutex);
        if(!done)throw std::runtime_error("GPU job already outstanding");
        prefill_input.assign(static_cast<const uint8_t *>(q),
                static_cast<const uint8_t *>(q)+size_t(ntokens)*input_stride);
        prefill_routes.assign(routes,routes+route_count);
        prefill_weights.assign(weights,weights+route_count);
        prefill_transient.clear();
        if(transient_count)prefill_transient.assign(transient,transient+transient_count);
        prefill_output.resize(size_t(ntokens)*size_t(latent));
        prefill_input_stride=input_stride;prefill_nslots=nslots;prefill_ntokens=ntokens;prefill_activation=*a;
        prefill_job_stats={};
        prefill_job_stats.resident_experts=uint64_t(std::count_if(prefill_slots.begin(),prefill_slots.end(),[](int slot){return slot>=0;}));
        prefill_job_stats.transient_experts=uint64_t(transient_count);
        for(const auto & item:prefill_transient){
            if(item.source==KIMI_VRAM_PREFILL_TRANSIENT_RAM)++prefill_job_stats.ram_transient_experts;
            else ++prefill_job_stats.ssd_transient_experts;
        }
        for(size_t route=0;route<route_count;++route){
            const int request=prefill_routes[route];
            if(request<0)continue;
            if(prefill_slots[size_t(request)]>=0)++prefill_job_stats.resident_routes;
            else{
                const auto item=std::find_if(prefill_transient.begin(),prefill_transient.end(),
                    [&](const kimi_vram_prefill_transient & value){return value.request_index==request;});
                if(item!=prefill_transient.end()){
                    ++prefill_job_stats.transient_routes;
                    if(item->source==KIMI_VRAM_PREFILL_TRANSIENT_RAM)++prefill_job_stats.ram_transient_routes;
                    else ++prefill_job_stats.ssd_transient_routes;
                }
            }
        }
        prefill_live=true;
        failure.clear();done=false;prefill_queued=true;
        if(timing){launch_time=clock_type::now();++launches;}
        cv.notify_all();
    }
    void finish_prefill(float * output,size_t output_stride,kimi_vram_prefill_stats * stats){
        if(!prefill_live)return;
        const auto finish_start=timing?clock_type::now():clock_type::time_point{};
        drain();if(timing)finish_drain_us+=micros(finish_start);
        if(!failure.empty())throw std::runtime_error(failure);
        if(!output || output_stride<size_t(latent))throw std::runtime_error("grouped-prefill output geometry");
        const auto add_start=clock_type::now();
        for(int token_index=0;token_index<prefill_ntokens;++token_index){
            float * destination=output+size_t(token_index)*output_stride;
            const float * source=prefill_output.data()+size_t(token_index)*size_t(latent);
            for(int col=0;col<latent;++col)destination[col]+=source[col];
        }
        prefill_job_stats.output_add_us=micros(add_start);
        if(prefill_generation)table->release_ready(prefill_generation);
        if(stats)*stats=prefill_job_stats;
        if(prefill_job_stats.transient_experts)std::fprintf(stderr,
            "[kimi_vram_prefill_transient] experts=%llu routes=%llu upload_bytes=%llu upload_us=%.3f "
            "ram_experts=%llu ram_routes=%llu ram_upload_bytes=%llu ram_upload_us=%.3f "
            "ssd_experts=%llu ssd_routes=%llu ssd_upload_bytes=%llu ssd_upload_us=%.3f "
            "worker_us=%.3f scratch_bytes=%zu displacement_slots=0\n",
            (unsigned long long)prefill_job_stats.transient_experts,(unsigned long long)prefill_job_stats.transient_routes,
            (unsigned long long)prefill_job_stats.transient_upload_bytes,prefill_job_stats.transient_upload_us,
            (unsigned long long)prefill_job_stats.ram_transient_experts,
            (unsigned long long)prefill_job_stats.ram_transient_routes,
            (unsigned long long)prefill_job_stats.ram_transient_upload_bytes,
            prefill_job_stats.ram_transient_upload_us,
            (unsigned long long)prefill_job_stats.ssd_transient_experts,
            (unsigned long long)prefill_job_stats.ssd_transient_routes,
            (unsigned long long)prefill_job_stats.ssd_transient_upload_bytes,
            prefill_job_stats.ssd_transient_upload_us,
            prefill_job_stats.total_worker_us,prefill_transient_bytes);
        prefill_live=false;prefill_generation=0;prefill_slots.clear();prefill_input.clear();
        prefill_routes.clear();prefill_weights.clear();prefill_output.clear();prefill_transient.clear();work.clear();output_rows.clear();
    }
    void cancel_prefill(){
        {std::lock_guard<std::mutex> lock(lifecycle_mutex);cancel_seeds();}
        if(cold_prefill_admission_phase){release();return;}
        if(!prefill_live)return;
        drain();release_mxfp4_accumulator();
        if(table && prefill_generation)table->release_ready(prefill_generation);
        prefill_live=false;prefill_generation=0;prefill_slots.clear();prefill_input.clear();
        prefill_routes.clear();prefill_weights.clear();prefill_output.clear();prefill_transient.clear();work.clear();output_rows.clear();
    }
    // Deferred path (kimi_vram_deferred_api): like a borrowed async admit, but the upload may outlive finish.
    void admit_deferred(const kimi_vram_request & r,const void * const * data,int * queued){
        if(!queued)throw std::runtime_error("null deferred VRAM admission result");
        *queued=0;
        if(!deferred_upload)throw std::runtime_error("deferred VRAM upload is off in the tier (GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD)");
        if(!active || !table || !table->in_flight)return;
        for(size_t j=0;j<table->admissions.size();++j){
            const int slot=table->admissions[j];
            if(slot<0 || table->slots[slot].valid || table->slots[slot].uploading ||
                    std::find(pending_uploads.begin(),pending_uploads.end(),j)!=pending_uploads.end() || !vram_slot_table::same(r,table->requests[j]))continue;
            for(int i=0;i<3;++i)if(!data[i])throw std::runtime_error("null VRAM admission source");
            {
                std::unique_lock<std::mutex> lock(mutex);
                if(!failure.empty())throw std::runtime_error(failure);
                const auto copying=[&]{return size_t(std::count_if(deferred.begin(),deferred.end(),[](const deferred_job & d){return !d.completed;}));};
                if(copying()>=deferred_capacity){
                    ++deferred_capacity_waits;const auto t=clock_type::now();
                    cv.wait(lock,[&]{return !failure.empty() || copying()<deferred_capacity;});
                    deferred_wait_us+=micros(t);
                    if(!failure.empty())throw std::runtime_error(failure);
                }
                table->reserve_upload(j);
                deferred.push_back({++deferred_sequence,slot,table->requests[j],{data[0],data[1],data[2]},false,false,false});
                ++deferred_uploads;deferred_peak=std::max(deferred_peak,deferred.size());
            }
            upload_cv.notify_one();*queued=1;
            break;
        }
    }
    void collect(int32_t wait_layer,kimi_vram_request * out,int32_t * aborted,int capacity,int * count){
        if(!count || capacity<0 || (capacity && (!out || !aborted)) || wait_layer<-2)throw std::runtime_error("invalid deferred VRAM collect");
        *count=0;
        if(!deferred_upload)throw std::runtime_error("deferred VRAM upload is off in the tier (GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD)");
        if((table && (table->in_flight || table->lease_generation)) || live_ready_generation || native_batch)
            throw std::runtime_error("deferred VRAM collect inside a layer or a residency lease");
        std::unique_lock<std::mutex> lock(mutex);
        const auto copying=[&]{return std::any_of(deferred.begin(),deferred.end(),[&](const deferred_job & d){
            return !d.completed && (wait_layer==-1 || d.key.layer==wait_layer);});};
        if(wait_layer!=-2 && copying()){
            ++deferred_layer_waits;const auto t=clock_type::now();
            cv.wait(lock,[&]{return !copying();});
            deferred_wait_us+=micros(t);
        }
        if(!failure.empty())throw std::runtime_error(failure);
        for(auto it=deferred.begin();it!=deferred.end() && *count<capacity;){
            if(!it->completed){++it;continue;}
            if(!it->aborted){
                if(!table)throw std::runtime_error("deferred VRAM upload without a slot table");
                table->publish_upload(size_t(it->slot),it->key);
                ++deferred_published;
            }
            aborted[*count]=it->aborted;out[(*count)++]=it->key;it=deferred.erase(it);
        }
    }
    void wait_deferred(){
        drain_deferred();
        std::lock_guard<std::mutex> lock(mutex);
        if(!failure.empty())throw std::runtime_error(failure);
    }
    void acquire_ready(uint64_t e,const kimi_vram_request * r,int count,kimi_vram_ready_lease * lease,int * ready){
        if(!active || !have_token || e!=epoch || !table || !lease || live_ready_generation || ready_sequence==UINT64_MAX || count<0)
            throw std::runtime_error("invalid VRAM residency epoch or phase");
        table->acquire_ready(r,size_t(count),ready);
        live_ready_generation=++ready_sequence;*lease={epoch,live_ready_generation};
    }
    void validate_ready(const kimi_vram_ready_lease & lease) const{
        if(!table || lease.epoch!=epoch || !lease.generation || lease.generation!=live_ready_generation)
            throw std::runtime_error("stale VRAM residency confirmation");
        table->validate_ready(table->lease_generation);
    }
    void release_ready(const kimi_vram_ready_lease & lease){
        validate_ready(lease);table->release_ready(table->lease_generation);live_ready_generation=0;
    }
    // The residency ABI's held(): under lifecycle_mutex, which every resize from another thread (host or OOM reclaim,
    // a prefill's seeds) takes.
    void held(int32_t layer,const int32_t * experts,int count,uint8_t * out){
        std::lock_guard<std::mutex> lock(lifecycle_mutex);
        for(int i=0;i<count;++i)out[i]=table && table->holds(layer,experts[i]);
    }
    #include "vram-location-provider.inc"
};
int fail(tier * t,char * error,size_t size,const std::exception & e){
    if(size)std::snprintf(error,size,"%s",e.what());
    try{t->release();}catch(const std::exception & drain_error){std::fprintf(stderr,"[kimi_vram_tier] cannot drain after %s: %s\n",e.what(),drain_error.what());std::abort();}
    return 1;
}
void * api_create(uint64_t bytes,char * error,size_t n){try{return new tier(bytes);}catch(const std::exception & e){if(n)std::snprintf(error,n,"%s",e.what());return nullptr;}}
void api_destroy(void * p){delete static_cast<tier *>(p);}
int api_begin(void * p,uint64_t epoch,uint64_t token,int decode,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->begin(epoch,token,decode);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int api_prepare(void * p,const kimi_vram_request * r,int count,int latent,int hidden,int * slots,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->prepare(r,count,latent,hidden,slots);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int api_launch(void * p,const void * q,size_t bytes,float beta,float linear,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->launch(q,bytes,beta,linear);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int api_admit(void * p,const kimi_vram_request * r,const void * up,const void * gate,const void * down,int borrow,char * e,size_t n){auto * t=static_cast<tier *>(p);try{const void * data[]={up,gate,down};t->admit(*r,data,borrow!=0);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int api_finish(void * p,float * output,size_t stride,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->finish(output,stride);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int api_cancel(void * p,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->release();return 0;}catch(const std::exception & x){if(n)std::snprintf(e,n,"%s",x.what());return 1;}}
int deferred_admit(void * p,const kimi_vram_request * r,const void * up,const void * gate,const void * down,int * queued,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{if(!r)throw std::runtime_error("null deferred VRAM admission");const void * data[]={up,gate,down};t->admit_deferred(*r,data,queued);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int deferred_collect(void * p,int32_t wait_layer,kimi_vram_request * out,int32_t * aborted,int capacity,int * count,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->collect(wait_layer,out,aborted,capacity,count);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int deferred_wait(void * p,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->wait_deferred();return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int elastic_phase_complete(void * p,int decode,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->phase_complete(decode!=0);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int elastic_reserve_accumulate(void * p,int tokens,char * e,size_t n){auto * t=static_cast<tier *>(p);try{t->reserve_mxfp4_accumulator(tokens);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int elastic_reserve(void * p,const kimi_vram_request * r,const kimi_vram_expert_format * f,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{if(!r || !f)throw std::runtime_error("null VRAM reserve geometry");t->reserve(*r,*f);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int format_prepare(void * p,const kimi_vram_request * r,int count,const kimi_vram_expert_format * f,int * slots,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->prepare_format(r,count,f,slots);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int format_launch(void * p,const void * q,size_t bytes,const kimi_vram_activation * a,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->launch_activation(q,bytes,a);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int prefill_prepare(void * p,const kimi_vram_request * r,int count,const kimi_vram_expert_format * f,int * slots,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->prepare_prefill(r,count,f,slots);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int prefill_transient_available(void * p){return p?static_cast<tier *>(p)->transient_prefill_available():0;}
int prefill_launch(void * p,const void * q,size_t stride,const int32_t * routes,const float * weights,
        int nslots,int ntokens,const kimi_vram_prefill_transient * transient,int transient_count,
        const kimi_vram_activation * a,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->launch_prefill(q,stride,routes,weights,nslots,ntokens,transient,transient_count,a);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int prefill_finish(void * p,float * output,size_t stride,kimi_vram_prefill_stats * stats,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{if(stats)*stats={};t->finish_prefill(output,stride,stats);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int prefill_cancel(void * p,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->cancel_prefill();return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int prefill_cold_admission_active(void * p){return p?static_cast<tier *>(p)->cold_prefill_admission_active():0;}
int prefill_admit_cold(void * p,const kimi_vram_request * r,const void * up,const void * gate,const void * down,
        int * admitted,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{if(!r)throw std::runtime_error("null cold-prefill VRAM request");
        const void * data[]={up,gate,down};t->admit_cold_prefill(*r,data,admitted);return 0;
    }catch(const std::exception & x){return fail(t,e,n,x);}}
int residency_acquire(void * p,uint64_t epoch,const kimi_vram_request * r,int count,kimi_vram_ready_lease * lease,int * ready,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->acquire_ready(epoch,r,count,lease,ready);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int residency_validate(void * p,const kimi_vram_ready_lease * lease,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{if(!lease)throw std::runtime_error("null VRAM lease");t->validate_ready(*lease);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
int residency_release(void * p,const kimi_vram_ready_lease * lease,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{if(!lease)throw std::runtime_error("null VRAM lease");t->release_ready(*lease);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
void residency_held(void * p,int32_t layer,const int32_t * experts,int count,uint8_t * out){
    if(!out || count<=0)return;
    std::fill(out,out+count,uint8_t(0));if(p && experts)static_cast<tier *>(p)->held(layer,experts,count,out);}
// Native callbacks preserve any returned leases on error. The consumer owns
// cancel+release; the legacy fail() would free their payload too early.
template<class F> int native_call(F call,char * error,size_t size){
    try{call();return 0;}catch(const std::exception & e){if(error&&size)std::snprintf(error,size,"%s",e.what());return 1;}
}
int location_configure(void * p,int l,int h,char * e,size_t n){return native_call([&]{static_cast<tier *>(p)->native_configure(l,h);},e,n);}
int location_acquire(void * p,const kimi_cache_location_candidate * c,int count,kimi_cache_location_lease * leases,char * e,size_t n){
    return native_call([&]{static_cast<tier *>(p)->native_acquire(c,count,leases);},e,n);}
int location_begin(void * p,const kimi_cache_location_candidate * c,int count,uint32_t base,uint32_t pin,char * e,size_t n){
    return native_call([&]{static_cast<tier *>(p)->native_begin_layer(c,count,base,pin);},e,n);}
int location_launch(void * p,const kimi_cache_location_job * jobs,int count,const void * q,size_t bytes,int l,int h,float b,float lb,char * e,size_t n){
    return native_call([&]{static_cast<tier *>(p)->native_launch(jobs,count,q,bytes,l,h,b,lb);},e,n);}
int location_admit(void * p,const kimi_cache_location_candidate * c,const void * up,const void * gate,const void * down,int borrow,char * e,size_t n){
    return native_call([&]{if(!c)throw std::runtime_error("null native VRAM admission");const void * data[]={up,gate,down};static_cast<tier *>(p)->native_admit(*c,data,borrow!=0);},e,n);}
int location_finish(void * p,char * e,size_t n){return native_call([&]{static_cast<tier *>(p)->native_finish();},e,n);}
int location_cancel(void * p,char * e,size_t n){return native_call([&]{static_cast<tier *>(p)->native_cancel();},e,n);}
int location_release(void * p,const kimi_cache_location_lease * leases,int count,char * e,size_t n){
    return native_call([&]{static_cast<tier *>(p)->native_release(leases,count);},e,n);}
int host_available(void * p){return p && static_cast<tier *>(p)->format_mode;}
int host_launch(void * p,const kimi_vram_host_job * jobs,int count,int * accepted,char * e,size_t n){
    auto * t=static_cast<tier *>(p);try{t->launch_host(jobs,count,accepted);return 0;}catch(const std::exception & x){return fail(t,e,n,x);}}
void host_costs(void * p,double * resident_us,double * host_us){
    auto * t=static_cast<tier *>(p);std::lock_guard<std::mutex> lock(t->mutex);
    if(resident_us)*resident_us=t->resident_us_per_expert;if(host_us)*host_us=t->host_us_per_expert;}
void host_costs2(void * p,kimi_vram_host_costs * c){
    if(!c)return;*c={};auto * t=static_cast<tier *>(p);if(!t)return;
    std::lock_guard<std::mutex> lock(t->mutex);t->costs2(*c);}
// Not through fail(): a refused registration leaves the tier as it was; the caller keeps the memory pageable.
int host_register(void * p,const void * pointer,size_t bytes,char * e,size_t n){
    auto * t=static_cast<tier *>(p);
    if(!t || !pointer || !bytes){if(n)std::snprintf(e,n,"invalid host registration");return 1;}
    if(t->host_readable)return 0;
    if(t->device<0){if(n)std::snprintf(e,n,"host registration before the tier knows its device");return 1;}
    cudaError_t rc=cudaSetDevice(t->device);
    if(rc==cudaSuccess)rc=cudaHostRegister(const_cast<void *>(pointer),bytes,cudaHostRegisterDefault);
    if(rc!=cudaSuccess){(void)cudaGetLastError();if(n)std::snprintf(e,n,"cudaHostRegister: %s",cudaGetErrorString(rc));return 1;}
    t->host_registered(pointer,bytes);
    return 0;}
int host_unregister(void * p,const void * pointer,char * e,size_t n){
    auto * t=static_cast<tier *>(p);
    if(!t || !pointer){if(n)std::snprintf(e,n,"invalid host unregistration");return 1;}
    if(t->host_readable)return 0;
    try{t->host_unregistering(pointer);}catch(const std::exception & x){if(n)std::snprintf(e,n,"%s",x.what());return 1;}
    cudaError_t rc=t->device>=0?cudaSetDevice(t->device):cudaSuccess;
    if(rc==cudaSuccess)rc=cudaHostUnregister(const_cast<void *>(pointer));
    if(rc!=cudaSuccess){(void)cudaGetLastError();if(n)std::snprintf(e,n,"cudaHostUnregister: %s",cudaGetErrorString(rc));return 1;}
    return 0;}
void policy_set_host_demand(void * p,uint64_t bytes){
    auto * t=static_cast<tier *>(p);if(!t)return;
    std::lock_guard<std::mutex> lock(t->lifecycle_mutex);
    if(t->host_demand==size_t(bytes))return;
    const size_t before=t->host_demand;
    t->host_demand=size_t(bytes);t->publish_reserve();
    if(t->unified && (before>t->host_demand?before-t->host_demand:t->host_demand-before)>=(size_t(256)<<20))
        std::fprintf(stderr,"[kimi_vram_host_reserve] reason=host_demand demand_before=%zu demand=%zu floor=%zu\n",before,t->host_demand,t->host_floor);}
void policy_set_host_floor(void * p,uint64_t bytes){
    auto * t=static_cast<tier *>(p);if(!t)return;
    std::lock_guard<std::mutex> lock(t->lifecycle_mutex);
    if(t->host_floor==size_t(bytes))return;
    const size_t before=t->host_floor;
    t->host_floor=size_t(bytes);t->publish_reserve();
    if(t->unified)std::fprintf(stderr,"[kimi_vram_host_reserve] reason=floor floor_before=%zu floor=%zu demand=%zu\n",before,t->host_floor,t->host_demand);}
uint64_t policy_host_bytes(void * p){
    auto * t=static_cast<tier *>(p);if(!t)return 0;
    std::lock_guard<std::mutex> lock(t->lifecycle_mutex);
    return t->unified?t->capacity_bytes+t->prefill_transient_bytes:0;}
int policy_reclaim_host(void * p,uint64_t bytes){
    auto * t=static_cast<tier *>(p);if(!t)return 1;
    try{return t->reclaim_host(size_t(std::min<uint64_t>(bytes,SIZE_MAX)),bytes?"host_reserve":"floor")?0:1;}catch(const std::exception & x){
        std::fprintf(stderr,"[kimi_vram_elastic] event=host_reclaim status=fail error=%s\n",x.what());return 1;}}
void policy_set_miss_cost(void * p,double us){
    auto * t=static_cast<tier *>(p);if(!t || !std::isfinite(us) || us<0)return;
    std::lock_guard<std::mutex> lock(t->mutex);t->miss_us=t->miss_us>0?0.9*t->miss_us+0.1*us:us;}
} // namespace
extern "C" const kimi_vram_policy_api * kimi_vram_policy_get_api(){
    static const kimi_vram_policy_api api={3,sizeof(kimi_vram_policy_api),policy_set_miss_cost,policy_set_host_demand,
        policy_set_host_floor,policy_reclaim_host,policy_host_bytes};return &api;
}
extern "C" const kimi_vram_host_api * kimi_vram_host_get_api(){
    static const kimi_vram_host_api api={2,sizeof(kimi_vram_host_api),host_available,host_launch,host_costs,
        host_register,host_unregister,host_costs2};return &api;
}
extern "C" const kimi_vram_tier_api * kimi_vram_tier_get_api(){
    static const kimi_vram_tier_api api={2,sizeof(kimi_vram_tier_api),api_create,api_destroy,api_begin,api_prepare,api_launch,api_admit,api_finish,api_cancel};return &api;
}
extern "C" const kimi_vram_residency_api * kimi_vram_residency_get_api(){
    static const kimi_vram_residency_api api={2,sizeof(kimi_vram_residency_api),residency_acquire,residency_validate,residency_release,
        residency_held};return &api;
}
extern "C" const kimi_vram_location_api * kimi_vram_location_get_api(){
    static const kimi_cache_location_api view={1,sizeof(kimi_cache_location_api),location_acquire,location_begin,
        location_launch,location_admit,location_finish,location_cancel,location_release,nullptr};
    static const kimi_vram_location_api api={1,sizeof(kimi_vram_location_api),&view,location_configure};return &api;
}
extern "C" const kimi_vram_deferred_api * kimi_vram_deferred_get_api(){
    static const kimi_vram_deferred_api api={2,sizeof(kimi_vram_deferred_api),deferred_admit,deferred_collect,deferred_wait};return &api;
}
extern "C" const kimi_vram_elastic_api * kimi_vram_elastic_get_api(){
    static const kimi_vram_elastic_api api={3,sizeof(kimi_vram_elastic_api),elastic_phase_complete,elastic_reserve_accumulate,elastic_reserve};return &api;
}
extern "C" const kimi_vram_format_api * kimi_vram_format_get_api(){
    static const kimi_vram_format_api api={1,sizeof(kimi_vram_format_api),format_prepare,format_launch};return &api;
}
extern "C" const kimi_vram_prefill_api * kimi_vram_prefill_get_api(){
    static const kimi_vram_prefill_api api={5,sizeof(kimi_vram_prefill_api),prefill_prepare,
        prefill_transient_available,prefill_launch,prefill_finish,prefill_cancel,
        prefill_cold_admission_active,prefill_admit_cold};return &api;
}
