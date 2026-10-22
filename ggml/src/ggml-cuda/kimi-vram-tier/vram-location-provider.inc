// Methods on the existing tier. No separate arena, policy, worker or GPU math.
static kimi_vram_request native_key(const kimi_cache_location_candidate & c) {
    if(c.layer<1 || c.layer>92 || c.expert<0 || c.expert>=896)throw std::runtime_error("invalid native VRAM key");
    kimi_vram_request r={};r.layer=c.layer;r.expert=c.expert;r.weight=1;
    for(int role=0;role<3;++role){
        const auto & source=c.role[role];struct stat st={};
        const int flags=fcntl(source.fd_direct,F_GETFL);
        if(source.fd_direct<0 || flags<0 || !(flags&O_DIRECT) || fstat(source.fd_direct,&st) || !S_ISREG(st.st_mode) || st.st_size<0 ||
                source.offset%4096 || !source.nbytes || source.nbytes%4096 ||
                source.offset>uint64_t(st.st_size) || source.nbytes>uint64_t(st.st_size)-source.offset)
            throw std::runtime_error("invalid native VRAM source identity/range");
        r.role[role]={uint64_t(st.st_dev),uint64_t(st.st_ino),uint64_t(st.st_size),source.offset,source.nbytes,
            st.st_mtim.tv_sec,st.st_mtim.tv_nsec,st.st_ctim.tv_sec,st.st_ctim.tv_nsec};
    }
    return r;
}
void native_configure(int l,int h) {
    if(format_mode)throw std::runtime_error("the explicit-format VRAM tier has no cache-location provider");
    if(deferred_upload)throw std::runtime_error("deferred VRAM uploads do not support the cache-location provider");
    if(native_batch || !native_pins.empty() || (table&&table->in_flight) || live_ready_generation ||
            l<=0 || h<=0 || l>16384 || h>16384 || l%256 || h%256)
        throw std::runtime_error("invalid native VRAM geometry/phase");
    if(engine&&(latent!=l||hidden!=h))throw std::runtime_error("native VRAM geometry changed");
    native_latent=l;native_hidden=h;
}
void native_acquire(const kimi_cache_location_candidate * c,int count,kimi_cache_location_lease * result) {
    if(!active || !have_token || native_batch || !native_pins.empty() || live_ready_generation ||
            !native_latent || count<=0 || count>16 || !c || !result || (table&&table->in_flight))
        throw std::runtime_error("invalid native VRAM acquire phase");
    std::fill(result,result+count,kimi_cache_location_lease{});
    native_candidates.clear();native_candidates.reserve(count);native_pins.reserve(count);
    for(int i=0;i<count;++i){
        if(c[i].epoch!=epoch || c[i].token!=token || c[i].layer!=c[0].layer)
            throw std::runtime_error("native VRAM epoch/token/layer changed");
        native_candidates.push_back(native_key(c[i]));
    }
    initialize(native_candidates.data(),native_latent,native_hidden);
    for(const auto & r:native_candidates)for(int j=0;j<3;++j)
        if(r.role[j].bytes!=role_bytes[j])throw std::runtime_error("native VRAM role bytes changed");
    native_batch=true;native_begun=native_launched=native_cancelled=false;
    native_outputs.clear();work.clear();output_rows.clear();failure.clear();
    native_queries+=count;
    const auto & l=table->layers[c[0].layer];
    for(int j=0;j<count;++j)for(size_t i=l.begin;i<l.begin+l.count;++i){
        auto & e=table->slots[i];
        if(e.valid&&vram_slot_table::same(e.key,native_candidates[j])){
            if(native_generation==UINT64_MAX || e.native_refs==UINT32_MAX)throw std::runtime_error("native VRAM lease overflow");
            const kimi_cache_location_lease lease={uint64_t(i)+1,++native_generation};
            native_pins.push_back({lease,int(i),e.key});++e.native_refs;result[j]=lease;++native_query_hits;break;
        }
    }
}
native_pin & native_validate(const kimi_cache_location_lease & lease) {
    auto p=std::find_if(native_pins.begin(),native_pins.end(),[&](const native_pin & v){
        return v.lease.handle==lease.handle&&v.lease.generation==lease.generation;
    });
    if(!lease.handle || !lease.generation || p==native_pins.end() || !table ||
            p->slot<0 || size_t(p->slot)>=table->slots.size())throw std::runtime_error("stale/double native VRAM lease");
    const auto & e=table->slots[p->slot];
    if(!e.native_refs || !vram_slot_table::same(e.key,p->key) || (!e.valid&&!native_cancelled))
        throw std::runtime_error("native VRAM lease lost its payload");
    return *p;
}
void native_begin_layer(const kimi_cache_location_candidate * c,int count,uint32_t base,uint32_t pin) {
    if(!native_batch || native_begun || native_cancelled || !c || count!=int(native_candidates.size()) ||
            ((base|pin)>>count))throw std::runtime_error("invalid native VRAM layer masks");
    for(int j=0;j<count;++j){
        if(c[j].epoch!=epoch || c[j].token!=token || !vram_slot_table::same(native_key(c[j]),native_candidates[j]))
            throw std::runtime_error("native VRAM source changed before admission");
    }
    for(const auto & p:native_pins){
        bool selected=false;
        for(int j=0;j<count;++j)selected|=(pin&(1u<<j))&&vram_slot_table::same(p.key,native_candidates[j]);
        if(!selected)throw std::runtime_error("unused native VRAM pin still held at admission");
    }
    table->prepare(native_candidates.data(),native_candidates.size(),base);
    for(int j=0;j<count;++j)if(base&(1u<<j)){gpu_hits+=table->hits[j]>=0;cpu_requests+=table->hits[j]<0;}
    native_begun=true;
}
void native_launch(const kimi_cache_location_job * jobs,int count,const void * q,size_t bytes,int l,int h,float b,float lb) {
    if(!native_begun || native_launched || native_cancelled || !jobs || count<=0 || count>16 ||
            l!=latent || h!=hidden || !q || bytes!=size_t(latent/256)*292 || !std::isfinite(b) || b<=0 || !std::isfinite(lb))
        throw std::runtime_error("invalid native VRAM launch");
    std::lock_guard<std::mutex> lock(mutex);
    if(!failure.empty())throw std::runtime_error(failure);
    // Uploads may already be queued by a CPU job; the same worker gives this
    // selected compute batch priority at its next job boundary.
    if(queued)throw std::runtime_error("native VRAM compute already queued");
    work.clear();native_outputs.clear();
    for(int i=0;i<count;++i){
        const auto & job=jobs[i];const auto & p=native_validate(job.lease);
        if(!job.raw_down || job.native_slot<0 || size_t(job.native_slot)>=native_candidates.size() ||
                !vram_slot_table::same(p.key,native_candidates[job.native_slot]))throw std::runtime_error("native VRAM job/key mismatch");
        for(int j=0;j<i;++j)if(jobs[j].native_slot==job.native_slot)throw std::runtime_error("duplicate native VRAM job");
        char * src=slot_base(size_t(p.slot));
        work.push_back({src,src+role_bytes[0],src+role_bytes[0]+role_bytes[1]});native_outputs.push_back(job.raw_down);
    }
    input.assign(static_cast<const uint8_t *>(q),static_cast<const uint8_t *>(q)+bytes);
    down.resize(size_t(count)*latent);beta=b;linear_beta=lb;native_gpu_jobs+=count;native_launched=true;done=false;queued=true;if(timing){launch_time=clock_type::now();++launches;}cv.notify_all();
}
void native_admit(const kimi_cache_location_candidate & c,const void * const * data,bool borrow) {
    if(!native_begun || native_cancelled || c.epoch!=epoch || c.token!=token)throw std::runtime_error("invalid native VRAM admission phase");
    const auto key=native_key(c);admit(key,data,borrow);
}
void native_finish() {
    if(!native_begun || native_cancelled)throw std::runtime_error("invalid native VRAM finish phase");
    drain();if(!failure.empty())throw std::runtime_error(failure);
    for(size_t j:pending_uploads){table->publish(j);}
    pending_uploads.clear();
    for(size_t j=0;j<native_outputs.size();++j)std::copy_n(down.data()+j*latent,latent,native_outputs[j]);
    table->finish();work.clear();native_outputs.clear();native_begun=false;
    // The final compute leases still protect READY until the consumer finishes
    // ordered reduction and releases them. RAM dropping follows that release.
    native_batch=false;
}
void native_cancel() {
    drain();native_cancelled=true;native_begun=false;work.clear();native_outputs.clear();pending_uploads.clear();
    if(table){
        table->invalidate();
        table->in_flight=false;table->lease_generation=0;
    }
    if(native_pins.empty())release();
}
void native_release(const kimi_cache_location_lease * leases,int count) {
    if(count<0 || count>16 || (count&&!leases))throw std::runtime_error("invalid native VRAM release batch");
    {
        std::lock_guard<std::mutex> lock(mutex);
        if(native_begun || (native_launched&&!done))throw std::runtime_error("native VRAM release before finish/cancel");
    }
    for(int i=0;i<count;++i){
        native_validate(leases[i]);
        for(int j=0;j<i;++j)if(leases[i].handle==leases[j].handle&&leases[i].generation==leases[j].generation)
            throw std::runtime_error("duplicate native VRAM release batch");
    }
    for(int i=0;i<count;++i){
        const auto slot=native_validate(leases[i]).slot;--table->slots[slot].native_refs;
        native_pins.erase(std::find_if(native_pins.begin(),native_pins.end(),[&](const native_pin & p){
            return p.lease.handle==leases[i].handle&&p.lease.generation==leases[i].generation;
        }));
    }
    if(native_pins.empty()&&!native_begun){
        if(native_cancelled)release();
        else if(native_launched)native_batch=false;
    }
}
