#define main original_async_test_main
#include "test-kimi-vram-async.cpp"
#undef main
#include <unistd.h>

int main(){try{
    setenv("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD","1",1);
    // This small transport fixture uses the base GPU component. The separate
    // K3-shape location test covers the actual153 GPU hidden chain.
    setenv("GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN","0",1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_POLICY","lru",1);
    setenv("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT","0",1);
    const size_t role=28672;char path[]="/tmp/kimi-native-provider-XXXXXX";
    int temporary=mkstemp(path);require(temporary>=0,"temporary source create");
    require(ftruncate(temporary,9*role)==0,"temporary source size");close(temporary);
    const int fd=open(path,O_RDONLY|O_DIRECT);require(fd>=0,"direct source open");
    struct file_cleanup{int fd;const char * path;~file_cleanup(){close(fd);unlink(path);}} file{fd,path};
    std::vector<unsigned char> payload(role,0),q8(292,0);payload.back()=37;
    const void * data[]={payload.data(),payload.data(),payload.data()};
    kimi_cache_location_candidate candidates[3]={};
    for(int j=0;j<3;++j){auto & c=candidates[j];c.epoch=17;c.token=1;c.layer=1;c.expert=j;
        for(int i=0;i<3;++i)c.role[i]={fd,uint64_t(3*j+i)*role,role};}
    vram_compute_api instrumented={};tier t(92*2*3*role);
    struct cleanup{~cleanup(){unblock();}} cleanup_on_error;
    kimi_cache_location_lease leases[3]={};std::vector<float> raw(256);char error[512]={};
    auto begin=[&](uint64_t epoch,uint64_t token){for(auto & c:candidates){c.epoch=epoch;c.token=token;}t.begin(epoch,token,true);t.native_configure(256,256);};
    auto warm=[&](uint64_t epoch){begin(epoch,1);t.native_acquire(candidates,3,leases);require(!leases[0].handle&&!leases[1].handle,"cold query returned READY");
        t.native_begin_layer(candidates,3,1,0);t.native_admit(candidates[0],data,true);t.native_finish();};
    warm(17);real_compute=t.compute_api;instrumented=*real_compute;instrumented.compute=blocked_compute;t.compute_api=&instrumented;
    begin(17,2);const auto previous_clock=t.table->clock,previous_step=t.table->step;
    const auto previous_counts=t.table->layers[1].counts;
    t.native_acquire(candidates,3,leases);
    require(leases[0].handle&&!leases[1].handle&&!leases[2].handle,"native READY query mismatch");
    require(t.table->clock==previous_clock&&t.table->step==previous_step&&t.table->layers[1].counts==previous_counts,"native lookup refreshed policy");
    const auto old=leases[0];t.native_begin_layer(candidates,3,3,1);
    const int destination=t.table->admissions[1];require(destination>=0,"ordinary admission was not reserved");
    const kimi_cache_location_job job{leases[0],0,raw.data()};
    t.native_launch(&job,1,q8.data(),q8.size(),256,256,4,25);
    {std::unique_lock<std::mutex> lock(barrier_mutex);require(barrier_cv.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),"native worker did not enter");}
    t.native_admit(candidates[1],data,true);
    require(!t.table->slots[destination].valid&&t.pending_uploads.size()==1,"pending native upload was published READY");
    require(location_release(&t,leases,1,error,sizeof(error))!=0&&!t.native_pins.empty(),"early release did not preserve compute pin");
    rejected([&]{t.begin(18,1,true);});
    int ready[3]={};kimi_vram_ready_lease ready_lease={};
    rejected([&]{t.acquire_ready(17,t.native_candidates.data(),3,&ready_lease,ready);});
    unblock();t.native_finish();require(t.table->slots[destination].valid&&!t.native_pins.empty(),"finish lost READY or compute pin");
    rejected([&]{t.acquire_ready(17,t.native_candidates.data(),3,&ready_lease,ready);});
    t.native_release(leases,1);require(t.native_pins.empty(),"completed native pin leaked");
    t.acquire_ready(17,t.native_candidates.data(),3,&ready_lease,ready);require(ready[0]&&ready[1]&&!ready[2],"post-finish RAM drop confirmation mismatch");t.release_ready(ready_lease);
    begin(17,3);t.native_acquire(candidates,3,leases);require(leases[0].handle&&leases[1].handle,"warm native query missed");
    t.native_release(&leases[1],1);t.native_begin_layer(candidates,3,5,1);
    require(t.table->admissions[2]==destination&&t.table->slots[leases[0].handle-1].key.expert==0,"native pin did not veto LRU eviction");
    {std::lock_guard<std::mutex> lock(barrier_mutex);entered=proceed=false;inject_error=true;}
    const kimi_cache_location_job failed_job{leases[0],0,raw.data()};
    t.native_launch(&failed_job,1,q8.data(),q8.size(),256,256,4,25);
    {std::unique_lock<std::mutex> lock(barrier_mutex);require(barrier_cv.wait_for(lock,std::chrono::seconds(5),[]{return entered;}),"native error worker did not enter");}
    t.native_admit(candidates[2],data,true);unblock();
    require(location_finish(&t,error,sizeof(error))!=0&&std::strstr(error,"injected worker error")&&t.arena&&!t.native_pins.empty(),"GPU error freed the consumer lease");
    t.native_cancel();require(t.done&&t.upload_jobs.empty()&&!t.native_pins.empty(),"cancel did not drain and retain releasable leases");
    t.native_release(leases,1);require(!t.arena&&!t.table&&t.native_pins.empty(),"cancel/release retained owned payload");
    t.compute_api=real_compute;warm(18);begin(18,2);t.native_acquire(candidates,3,leases);
    require(leases[0].generation>old.generation,"new arena reused native generation");
    require(location_release(&t,&old,1,error,sizeof(error))!=0,"stale native release accepted");
    candidates[0].role[0].offset+=4096;rejected([&]{t.native_begin_layer(candidates,3,1,1);});candidates[0].role[0].offset-=4096;
    ++candidates[0].epoch;rejected([&]{t.native_begin_layer(candidates,3,1,1);});--candidates[0].epoch;
    t.native_begin_layer(candidates,3,1,1);t.native_finish();t.native_release(leases,1);
    require(location_release(&t,leases,1,error,sizeof(error))!=0,"double native release accepted");
    std::printf("REAL_CUDA_LOCATION_LIFECYCLE_PASS assertions=%d\n",assertions);return 0;
}catch(const std::exception & e){unblock();std::fprintf(stderr,"LOCATION_LIFECYCLE_FAIL: %s\n",e.what());return 1;}}
