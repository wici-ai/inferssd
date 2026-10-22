// Exercise the real CPU sink, io_uring, borrowed H2D and test2 GPU math at the
// actual K3 expert shape. Synthetic packed weights avoid a model dependency;
// the operator gate separately checks the real model's packed weights.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main

int main() { try {
    ggml_cpu_init();
    setenv("GGML_KIMI_CPU_SINK_READY_FIRST","1",1);
    setenv("GGML_KIMI_CPU_SINK_SPLIT_CV","1",1);
    setenv("GGML_KIMI_CPU_SINK_CACHE_POLICY","lru",1);
    setenv("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION","uniform",1);
    setenv("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS","0",1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_POLICY","lru",1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT","0",1);
    fixture f;f.latent=3584;f.hidden=3072;f.experts=4;make_synthetic(f);
    const auto ram_bytes=2*triplet_bytes(f);
    const auto gpu_bytes=std::to_string(92*triplet_bytes(f)); // one GPU slot/layer
    struct options { bool exclusive,async,hidden; };
    size_t compared=0,checks=0;
    for(const auto cfg : {options{false,false,false},options{true,false,false},options{true,true,false},options{true,true,true},options{false,true,true}}) {
        setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES","0",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE","0",1);
        auto cpu=make_sink(f,16,2);
        setenv("GGML_KIMI_CPU_SINK_VRAM_BYTES",gpu_bytes.c_str(),1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE",cfg.exclusive?"1":"0",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD",cfg.async?"1":"0",1);
        setenv("GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN",cfg.hidden?"1":"0",1);
        auto mixed=make_sink(f,16,2);
        configure_cache(cpu.get(),ram_bytes,{1,2},1);configure_cache(mixed.get(),ram_bytes,{1,2},1);
        for(int token=1;token<=10;++token) {
            const bool decode=token!=7;
            const uint64_t epoch=token<9?17:18;
            begin_cache(cpu.get(),epoch,token,decode);begin_cache(mixed.get(),epoch,token,decode);
            for(int layer:{1,2}) {
                const std::vector<int32_t> ids=token==3?std::vector<int32_t>{2,1,3}:
                    token==4?std::vector<int32_t>{2,1}:token==5?std::vector<int32_t>{2,2,0}:std::vector<int32_t>{0,1,2};
                std::vector<float> weights(ids.size(),.2f);weights[0]=.6f;
                std::vector<float> input(f.latent);
                for(int i=0;i<f.latent;++i)input[i]=.031f*std::sin(float(i*7+token*31))+.1f;
                std::vector<float> a(f.latent),b(f.latent);tensors meta(f,layer);char error[512]={};
                kimi_cpu_sink_stats stats={};
                for(int arm=0;arm<2;++arm){auto sink=arm?mixed.get():cpu.get();auto & out=arm?b:a;
                    require(kimi_cpu_sink_run(sink,meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),ids.size(),4,25,out.data(),&stats,error,sizeof(error)),error);
                    require(stats.read_errors==0 && stats.pending_reads==0 && stats.fallback_count==0,"hybrid work did not drain");++checks;
                }
                require(!std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"exclusive/async/hidden chain changed output");compared+=a.size();
                const auto state=cache_snapshot(mixed.get());
                if(token==1 && cfg.exclusive) { require(state.valid_bytes==0,"new GPU copy still occupied RAM");++checks; }
            }
        }
        // Resolve a later request beyond EOF after other cache reads were
        // submitted. Any queued uploads must drain before RAM is released.
        begin_cache(mixed.get(),18,11,false);begin_cache(mixed.get(),18,12);
        f.fault=fixture::past_eof;f.fault_expert=3;f.fault_role=2;
        tensors meta(f,1);std::vector<int32_t> ids={0,1,3};std::vector<float> weights(3,.3f),input(f.latent,.1f),out(f.latent);
        char error[512]={};kimi_cpu_sink_stats stats={};
        require(!kimi_cpu_sink_run(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),3,4,25,out.data(),&stats,error,sizeof(error)),"I/O failure accepted");
        require(stats.pending_reads==0 && stats.fallback_count==0,"I/O failure returned with live work");++checks;
        f.fault=fixture::none;f.fault_expert=f.fault_role=-1;
        begin_cache(mixed.get(),18,13,false);begin_cache(mixed.get(),18,14);
        require(kimi_cpu_sink_run(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),3,4,25,out.data(),&stats,error,sizeof(error)),error);++checks;
        std::printf("EXCLUSIVE_MODE_PASS exclusive=%d async=%d gpu_hidden=%d\n",int(cfg.exclusive),int(cfg.async),int(cfg.hidden));
    }
    std::printf("REAL_GPU_EXCLUSIVE_PASS compared=%zu assertions=%zu\n",compared,checks);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"REAL_GPU_EXCLUSIVE_FAIL: %s\n",e.what());return 1;} }
