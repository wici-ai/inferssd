#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-kimi-q3k-parity.h"
#include "kimi-cpu-sink.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

static void need(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static size_t checks = 0, floats = 0, bad = 0;
static void equal(const char * phase, const float * a, const float * b, size_t n) {
    size_t unequal = 0, first = n;
    double max_abs = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::memcmp(a+i, b+i, 4)) { ++unequal; first = std::min(first, i); }
        need(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite test output");
        max_abs = std::max(max_abs, std::abs(double(a[i])-double(b[i])));
    }
    ++checks; floats += n; bad += unequal;
    std::printf("{\"phase\":\"%s\",\"elements\":%zu,\"unequal_bits\":%zu,\"max_abs\":%.12g,\"first\":%zu}\n", phase, n, unequal, max_abs, first);
    if (unequal) std::printf("DIFF %s %zu cpu=%a gpu=%a\n",phase,first,a[first],b[first]);
    std::fflush(stdout);
}
static void same_bytes(const char * phase, const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    need(a.size()==b.size(), "byte shape");
    size_t unequal = 0;
    for (size_t i = 0; i < a.size(); ++i) unequal += a[i]!=b[i];
    bad += unequal; ++checks;
    std::printf("{\"phase\":\"%s\",\"bytes\":%zu,\"unequal_bytes\":%zu}\n",phase,a.size(),unequal);
}
static std::vector<uint8_t> quant(const std::vector<float> & x) {
    std::vector<uint8_t> q(x.size()/256*sizeof(block_q8_K), 0);
    ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float(x.data(),q.data(),x.size());
    return q;
}
static std::vector<float> dot(const std::vector<uint8_t> & w, int columns, int rows, const std::vector<uint8_t> & q) {
    std::vector<float> y(rows);
    const auto fn = ggml_get_type_traits_cpu(GGML_TYPE_Q3_K)->vec_dot;
    const size_t stride = ggml_row_size(GGML_TYPE_Q3_K,columns);
    for (int r=0;r<rows;++r) fn(columns,&y[r],0,w.data()+r*stride,0,q.data(),0,1);
    return y;
}
static std::vector<float> situ(const std::vector<float> & up, const std::vector<float> & gate, float beta, float linear) {
    std::vector<float> result(up.size());
    for (size_t r=0;r<up.size();++r) {
        float u=up[r], g=gate[r];
        const float a=(beta*std::tanh(g*(1.0f/beta)))*(1.0f/(1.0f+std::exp(-g)));
        if(linear>0) u=linear*std::tanh(u*(1.0f/linear));
        result[r]=a*u;
    }
    return result;
}
struct disk_entry { char name[128]; int32_t expert; uint32_t reserved; uint64_t offset, bytes; };
struct pack {
    int fd=-1, direct=-1;
    std::map<std::pair<std::string,int>,disk_entry> entries;
    explicit pack(const char * path) {
        fd=open(path,O_RDONLY);direct=open(path,O_RDONLY|O_DIRECT);
        need(fd>=0&&direct>=0,"pack open");
        unsigned char h[40];need(pread(fd,h,40,0)==40&&!std::memcmp(h,"GGMLMOEPACKv1",12),"pack header");
        uint64_t n;std::memcpy(&n,h+24,8);need(n<1000000,"pack count");
        std::vector<disk_entry> index(n);
        need(pread(fd,index.data(),n*sizeof(disk_entry),40)==ssize_t(n*sizeof(disk_entry)),"pack index");
        for (const auto & e:index) entries[{e.name,e.expert}]=e;
    }
    ~pack(){if(fd>=0)close(fd);if(direct>=0)close(direct);}
    static std::string name(int l,const char * role){return "blk."+std::to_string(l)+".ffn_"+role+"_exps.weight";}
    std::vector<uint8_t> read(int l,int e,const char * role){
        const auto & x=entries.at({name(l,role),e});std::vector<uint8_t> b(x.bytes);
        need(pread(fd,b.data(),b.size(),x.offset)==ssize_t(b.size()),"weight read");return b;
    }
    static bool resolve(void * p,const char * name,int32_t id,kimi_cpu_sink_source * out){
        auto & self=*static_cast<pack *>(p);auto it=self.entries.find({name,id});
        if(it==self.entries.end())return false;
        *out={self.direct,it->second.offset,size_t(it->second.bytes)};return true;
    }
};
struct gpu {
    const kimi_q3k_gpu_api_v1 * api;
    kimi_q3k_gpu * handle=nullptr;
    std::vector<void *> weights;
    explicit gpu(const char * path) {
        void * so=dlopen(path,RTLD_NOW|RTLD_LOCAL);need(so,dlerror());
        auto get=reinterpret_cast<const kimi_q3k_gpu_api_v1 *(*)()>(dlsym(so,"kimi_q3k_gpu_get_api_v1"));
        need(get,"API export");api=get();need(api->version==1&&api->job_size==sizeof(kimi_q3k_gpu_job),"ABI");
        kimi_q3k_gpu_config cfg{0,32,7168,7168};char err[256];
        unsetenv("GGML_KIMI_Q3K_GPU_PARITY");need(!api->create(&cfg,err,sizeof(err)),"default off");
        setenv("GGML_KIMI_Q3K_GPU_PARITY","1",1);handle=api->create(&cfg,err,sizeof(err));need(handle,err);
        std::printf("{\"host_scratch_bytes\":%zu,\"device_scratch_bytes\":%zu}\n",api->host_bytes(handle),api->device_bytes(handle));
    }
    ~gpu(){for(void * p:weights)api->release_weights(handle,p);if(handle)api->destroy(handle);}
    void clear(){need(api->wait(handle),api->error(handle));for(void * p:weights)need(api->release_weights(handle,p),api->error(handle));weights.clear();}
    void * upload(const std::vector<uint8_t> & w){void * p=nullptr;need(api->upload(handle,w.data(),w.size(),&p),api->error(handle));weights.push_back(p);return p;}
    std::vector<std::vector<float>> run(const std::vector<kimi_q3k_gpu_job> & jobs,const std::vector<uint8_t> & q){
        need(api->submit(handle,jobs.data(),jobs.size(),q.data(),q.size()),api->error(handle));
        need(!api->submit(handle,jobs.data(),jobs.size(),q.data(),q.size()),"reject in-flight reuse");
        need(api->wait(handle),api->error(handle));std::vector<std::vector<float>> result;
        for(size_t i=0;i<jobs.size();++i){int n;const float * p=api->result(handle,i,&n);need(p&&n==jobs[i].rows,"output shape");result.emplace_back(p,p+n);}return result;
    }
};
static kimi_q3k_gpu_job job(void * w,size_t bytes,int k,int rows,size_t off=0,float weight=1,int scaled=0){
    return {w,bytes,size_t(ggml_row_size(GGML_TYPE_Q3_K,k)),off,k,rows,weight,scaled};
}
int main(int argc,char ** argv){
    try{
        need(argc==3,"usage: test component.so expert-pack");
        auto backend=ggml_backend_cpu_init();need(backend,"CPU backend");
        gpu g(argv[1]);pack p(argv[2]);std::mt19937 rng(20260911);std::normal_distribution<float> normal(0,1);
        // Real Q3_K pack bytes. Several independent activation rows cover low,
        // high and clipping regimes; full-model replay supplies live activations.
        for(int layer:{1,3,29,30,63,92})for(int expert:{0,65,895}){
            std::printf("{\"case\":\"real_expert\",\"layer\":%d,\"expert\":%d}\n",layer,expert);
            g.clear();auto wu=p.read(layer,expert,"up"),wg=p.read(layer,expert,"gate"),wd=p.read(layer,expert,"down");
            need(wu.size()==4730880&&wg.size()==wu.size()&&wd.size()==wu.size(),"actual expert bytes");
            void * du=g.upload(wu),*dg=g.upload(wg),*dd=g.upload(wd);
            for(int input_case=0;input_case<5;++input_case){
                std::vector<float> input(7168);
                for(size_t i=0;i<input.size();++i) input[i]=input_case==0?0:input_case==1?normal(rng):input_case==2?normal(rng)*64:input_case==3?(i%2?1.0f:-1.0f):normal(rng)*0.001f;
                auto q=quant(input);auto u=dot(wu,7168,1536,q),v=dot(wg,7168,1536,q);
                auto uv=g.run({job(du,wu.size(),7168,1536),job(dg,wg.size(),7168,1536)},q);
                equal("up",u.data(),uv[0].data(),u.size());equal("gate",v.data(),uv[1].data(),v.size());
                for(float linear:{0.0f,7.0f}){
                    auto h=situ(u,v,7.0f,linear),gh=situ(uv[0],uv[1],7.0f,linear);
                    equal("host_situ",h.data(),gh.data(),h.size());auto qh=quant(h),gqh=quant(gh);
                    same_bytes("hidden_Q8_K",qh,gqh);auto down=dot(wd,1536,7168,qh);
                    auto gd=g.run({job(dd,wd.size(),1536,7168)},gqh);equal("down",down.data(),gd[0].data(),down.size());
                    for(float weight:{0.0f,-0.0f,0.2734375f,-0.1f}){
                        std::vector<float> scaled(down.size());for(size_t i=0;i<down.size();++i){volatile float x=down[i]*weight;scaled[i]=x;}
                        auto gs=g.run({job(dd,wd.size(),1536,7168,0,weight,1)},gqh);equal("weighted",scaled.data(),gs[0].data(),scaled.size());
                    }
                }
            }
        }
        // Independent rows/broadcast/repeated weight jobs and partial CUDA block.
        g.clear();auto w=p.read(1,65,"up");void * dw=g.upload(w);
        std::vector<float>x(7168);for(auto & v:x)v=normal(rng);auto q=quant(x);auto expected=dot(w,7168,1536,q);
        for(int n:{1,2,3,4,5,8,16,32}){
            std::vector<kimi_q3k_gpu_job> jobs;for(int i=0;i<n;++i)jobs.push_back(job(dw,w.size(),7168,1+(i*17)%1536));
            auto ys=g.run(jobs,q);for(const auto & y:ys)equal("job_shapes",expected.data(),y.data(),y.size());
        }
        auto invalid=job(dw,w.size(),7167,1);need(!g.api->submit(g.handle,&invalid,1,q.data(),q.size()),"reject columns");
        invalid=job(dw,1,7168,1);need(!g.api->submit(g.handle,&invalid,1,q.data(),q.size()),"reject weight range");
        invalid=job(dw,w.size(),7168,1,1);need(!g.api->submit(g.handle,&invalid,1,q.data(),q.size()),"reject Q8 alignment");
        invalid=job(dw,w.size(),7168,1);need(!g.api->submit(g.handle,&invalid,33,q.data(),q.size()),"reject job cap");
        need(g.api->query(g.handle)==1,"idle after rejected requests");
        // Actual deployed CPU sink is an independent end-to-end reference,
        // including its SiTU code generation and READY_FIRST ordered addition.
        setenv("GGML_KIMI_CPU_SINK_SPLIT_CV","1",1);setenv("GGML_KIMI_CPU_SINK_READY_FIRST","1",1);
        setenv("GGML_KIMI_CPU_SINK_EXPERT_JOB","0",1);setenv("GGML_KIMI_CPU_SINK_CACHE_BYTES","0",1);
        char err[256];auto sink=kimi_cpu_sink_create(16,8,pack::resolve,&p,err,sizeof(err));need(sink,err);
        auto ctx=ggml_init({size_t(1)<<20,nullptr,true});need(ctx,"metadata context");
        auto up=ggml_new_tensor_3d(ctx,GGML_TYPE_Q3_K,7168,1536,896);
        auto gate=ggml_new_tensor_3d(ctx,GGML_TYPE_Q3_K,7168,1536,896);
        auto down=ggml_new_tensor_3d(ctx,GGML_TYPE_Q3_K,1536,7168,896);
        for(int layer:{1,3,29,63,92}){
            ggml_set_name(up,pack::name(layer,"up").c_str());ggml_set_name(gate,pack::name(layer,"gate").c_str());ggml_set_name(down,pack::name(layer,"down").c_str());
            const int32_t ids[]={65,0,895};const float weights[]={0.27f,0.43f,0.3f};
            for(float linear:{0.0f,7.0f}){
                for(auto & v:x)v=normal(rng);q=quant(x);std::vector<float> cpu(7168),mixed(7168,0);
                kimi_cpu_sink_stats stats{};need(kimi_cpu_sink_run(sink,up,gate,down,x.data(),ids,weights,3,7.0f,linear,cpu.data(),&stats,err,sizeof(err)),err);
                for(int i=0;i<3;++i){
                    g.clear();auto wu=p.read(layer,ids[i],"up"),wg=p.read(layer,ids[i],"gate"),wd=p.read(layer,ids[i],"down");
                    auto du=g.upload(wu),dg=g.upload(wg),dd=g.upload(wd);
                    auto uv=g.run({job(du,wu.size(),7168,1536),job(dg,wg.size(),7168,1536)},q);
                    auto hidden=situ(uv[0],uv[1],7.0f,linear);auto qh=quant(hidden);
                    auto y=g.run({job(dd,wd.size(),1536,7168,0,weights[i],1)},qh);
                    for(size_t r=0;r<mixed.size();++r)mixed[r]+=y[0][r];
                }
                equal("deployed_CPU_sink_ordered_sum",cpu.data(),mixed.data(),cpu.size());
            }
        }
        kimi_cpu_sink_destroy(sink);ggml_free(ctx);ggml_backend_free(backend);
        std::printf("{\"final\":true,\"checks\":%zu,\"floats\":%zu,\"unequal\":%zu}\n",checks,floats,bad);
        return bad?1:0;
    }catch(const std::exception & e){std::fprintf(stderr,"parity test: %s\n",e.what());return 2;}
}
