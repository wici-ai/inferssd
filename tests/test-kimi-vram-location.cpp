// Real CUDA provider and the real shared-location consumer, at K3 geometry.
// The oracle evaluates the final selected set using the original CPU path.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main

int main() { try {
    ggml_cpu_init();
    setenv("GGML_KIMI_CPU_SINK_READY_FIRST","1",1);
    setenv("GGML_KIMI_CPU_SINK_SPLIT_CV","1",1);
    setenv("GGML_KIMI_CPU_SINK_EXPERT_JOB","0",1);
    setenv("GGML_KIMI_CPU_SINK_EARLY_UP_GATE","0",1);
    setenv("GGML_KIMI_CPU_SINK_CACHE_POLICY","lru",1);
    setenv("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION","uniform",1);
    setenv("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS","0",1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_POLICY","count_decay2",1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT","2.5",1);
    fixture f;f.latent=3584;f.hidden=3072;f.experts=6;make_synthetic(f);
    const auto unit=triplet_bytes(f);const auto budget=std::to_string(92*2*unit);
    const int32_t native[]={0,1,2,3,4,5};const float nw[]={.32f,.24f,.16f,.12f,.10f,.06f};
    const int32_t ids[]={0,1,2,3};const float weights[]={.32f/.84f,.24f/.84f,.16f/.84f,.12f/.84f};
    std::vector<float> input(f.latent);
    for(int i=0;i<f.latent;++i)input[i]=.031f*std::sin(float(i*7+31))+.1f;
    char error[512]={};size_t compared=0,cases=0;
    for(int depth:{1,8})for(bool exclusive:{false,true})for(int mode=0;mode<=3;++mode)for(bool guarded:{false,true}) {
        setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES","0",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE","0",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS","0",1);
        auto cpu=make_sink(f,16,depth);
        setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES",budget.c_str(),1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE",exclusive?"1":"0",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD","1",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN","1",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_CACHE_LOCATIONS","1",1);
        auto mixed=make_sink(f,16,depth);configure_cache(mixed.get(),4*unit,{1},4);
        tensors meta(f,1);std::vector<float> warm(f.latent);
        // Four ordinary demands cross the unchanged count_decay2 threshold.
        // A later one-off demand fills RAM without displacing GPU IDs 0/4.
        // IDs 2/3 stay on SSD, so cached GPU ID 4 is the highest bonus
        // candidate even when shrink removes the lower-ranked original slots.
        for(int token=1;token<=5;++token) {
            begin_cache(mixed.get(),17,token);
            const std::vector<int32_t> selected=token<5?std::vector<int32_t>{0,4}:std::vector<int32_t>{1,5};
            const std::vector<float> w(selected.size(),1.0f/selected.size());
            require(kimi_cpu_sink_run(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),selected.data(),w.data(),int(selected.size()),4,25,warm.data(),nullptr,error,sizeof(error)),error);
            if(token==4&&exclusive)require(cache_snapshot(mixed.get()).valid_bytes==0,"GPU warm weights retained RAM duplicates");
        }
        begin_cache(mixed.get(),17,6);
        kimi_cache_compute_config config{1,mode,1,2,0,.5f,0,.65f,2,0,1};
        kimi_cache_compute_stats audit{};kimi_cpu_sink_stats stats{};
        std::vector<float> actual(f.latent),expected(f.latent);
        require(kimi_cpu_sink_run_cache_compute(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids,weights,4,native,nw,6,1,guarded,&config,4,25,actual.data(),&stats,&audit,error,sizeof(error)),error);
        require(!stats.pending_reads&&!stats.read_errors&&!stats.fallback_count&&stats.read_requests==stats.read_completions,"provider did not drain");
        require(stats.demand_bytes==uint64_t(__builtin_popcount(audit.plan.base_mask))*unit,"ordinary demand changed with cache location");
        require(stats.read_bytes==((guarded||mode!=KIMI_CACHE_COMPUTE_SHRINK)?2*unit:0),"RAM/GPU resident set did not prevent the expected SSD reads");
        if(!guarded&&mode!=KIMI_CACHE_COMPUTE_OFF)require(audit.plan.final_mask&(1u<<4),"GPU bonus/replacement was not selected");
        std::vector<int32_t> final_ids;std::vector<float> final_weights;
        for(int n=0;n<6;++n)if(audit.plan.final_mask&(1u<<n)){final_ids.push_back(native[n]);final_weights.push_back(audit.final_weights[n]);}
        require(kimi_cpu_sink_run(cpu.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),final_ids.data(),final_weights.data(),int(final_ids.size()),4,25,expected.data(),nullptr,error,sizeof(error)),error);
        require(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"actual CUDA provider changed final-set output");
        compared+=actual.size();++cases;
        // Reuse after native lease release, then invalidate on prefill/epoch.
        begin_cache(mixed.get(),17,7,false);begin_cache(mixed.get(),18,8);
        require(kimi_cpu_sink_run_cache_compute(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids,weights,4,native,nw,6,1,true,&config,4,25,actual.data(),&stats,&audit,error,sizeof(error)),error);
        require(audit.plan.final_mask==15&&stats.read_bytes==4*unit,"epoch/prefill did not invalidate old residency");
        require(kimi_cpu_sink_run(cpu.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids,weights,4,4,25,expected.data(),nullptr,error,sizeof(error)),error);
        require(!std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float)),"cold provider changed S output");
        compared+=actual.size();++cases;
        std::printf("REAL_LOCATION_CASE depth=%d exclusive=%d mode=%d guarded=%d PASS\n",depth,int(exclusive),mode,int(guarded));
    }
    std::printf("REAL_GPU_LOCATION_PASS cases=%zu compared=%zu\n",cases,compared);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"REAL_GPU_LOCATION_FAIL: %s\n",e.what());return 1;} }
