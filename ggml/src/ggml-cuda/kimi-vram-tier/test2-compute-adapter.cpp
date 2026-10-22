#include "vram-compute-api.h"
#include "ggml-kimi-q3k-parity.h"
#include "ggml-kimi-q3k-chain.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

namespace {
struct bridge {
    const kimi_q3k_gpu_api_v1 * api = nullptr;
    kimi_q3k_gpu * gpu = nullptr;
    const kimi_q3k_gpu_chain_api_v1 * chain_api = nullptr;
    kimi_q3k_gpu_chain * chain = nullptr;
    int latent, hidden, max_hits;
    ggml_from_float_t quantize = nullptr;
    std::vector<float> activation;
    std::vector<unsigned char> quant_hidden;
    std::vector<kimi_q3k_gpu_job> jobs;
    std::vector<kimi_q3k_gpu_job> down_jobs;

    bridge(int device, int l, int h, int n) : latent(l), hidden(h), max_hits(n) {
        auto get = reinterpret_cast<const kimi_q3k_gpu_api_v1 * (*)()>(dlsym(RTLD_DEFAULT,"kimi_q3k_gpu_get_api_v1"));
        if (!get) throw std::runtime_error("test2 Q3_K component missing");
        api = get();
        if (!api || api->version != 1 || api->job_size != sizeof(kimi_q3k_gpu_job) || !api->create ||
                !api->submit || !api->wait || !api->result || !api->error || !api->destroy ||
                !api->device_bytes || !api->host_bytes) throw std::runtime_error("test2 Q3_K ABI mismatch");
        bool gpu_hidden = false;
        if (const char * flag = std::getenv("GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN")) {
            if (std::strcmp(flag,"0") && std::strcmp(flag,"1")) throw std::runtime_error("VRAM_GPU_HIDDEN must be 0 or 1");
            gpu_hidden = !std::strcmp(flag,"1");
        }
        if (gpu_hidden) {
            auto get_chain = reinterpret_cast<const kimi_q3k_gpu_chain_api_v1 * (*)()>(dlsym(RTLD_DEFAULT,"kimi_q3k_gpu_get_chain_api_v1"));
            Dl_info base_info={},chain_info={};
            if (!get_chain || !dladdr(reinterpret_cast<void *>(get),&base_info) ||
                    !dladdr(reinterpret_cast<void *>(get_chain),&chain_info) || base_info.dli_fbase!=chain_info.dli_fbase)
                throw std::runtime_error("base and hidden chain must come from the same test2 DSO");
            chain_api=get_chain();
            if (!chain_api || chain_api->version!=1 || chain_api->size!=sizeof(*chain_api) || !chain_api->create ||
                    !chain_api->submit || !chain_api->destroy || !chain_api->host_bytes || !chain_api->device_bytes)
                throw std::runtime_error("test2 hidden chain ABI mismatch");
            if (l!=3584 || h!=3072) throw std::runtime_error("GPU hidden chain requires the 3584/3072 K3 expert shape");
            down_jobs.reserve(n);
        } else {
            quantize = ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
            if (!quantize) throw std::runtime_error("CPU Q8_K quantizer unavailable");
            activation.resize(hidden);
            quant_hidden.resize(size_t(n)*(hidden/256)*292);
        }
        jobs.reserve(2*n);
        kimi_q3k_gpu_config config{device,2*n,std::max(l,h),std::max(l,h)};
        char error[512] = {};
        gpu=api->create(&config,error,sizeof(error));
        if (!gpu) throw std::runtime_error(error);
        if (chain_api) {
            chain=chain_api->create(gpu);
            if (!chain) {
                const std::string message=api->error(gpu);
                if (!api->destroy(gpu)) std::abort();
                gpu=nullptr;throw std::runtime_error(message);
            }
        }
        std::fprintf(stderr,"[kimi_vram_bridge] gpu_hidden=%d hidden_quantizer=%s\n",int(gpu_hidden),gpu_hidden?"actual153_gpu_mul_then_add":"linked_ggml_cpu");
    }
    void close() noexcept {
        if (chain && !chain_api->destroy(chain)) {
            std::fprintf(stderr,"[kimi_vram_bridge] failed to drain test2 hidden chain at destroy\n");std::abort();
        }
        chain = nullptr;
        if (gpu && !api->destroy(gpu)) {
            std::fprintf(stderr,"[kimi_vram_bridge] failed to drain test2 component at destroy\n");
            std::abort();
        }
        gpu = nullptr;
    }
    ~bridge() { close(); }
    void submit(const void * q8,size_t bytes) {
        if (!api->submit(gpu,jobs.data(),int(jobs.size()),q8,bytes) || !api->wait(gpu))
            throw std::runtime_error(api->error(gpu));
    }
    const float * result(int job,int expected) {
        int rows=0;const float * p=api->result(gpu,job,&rows);
        if (!p || rows!=expected) throw std::runtime_error("test2 result shape mismatch");
        return p;
    }
    void compute(const vram_compute_expert * weights,int hits,const void * q8,size_t bytes,
                 float beta,float linear,float * down) {
        if (!gpu) throw std::runtime_error("test2 bridge closed after component error");
        if (hits<1 || hits>max_hits || !weights || !down || bytes!=size_t(latent/256)*292)
            throw std::runtime_error("test2 bridge input shape");
        const size_t up_stride=size_t(latent/256)*110;
        const size_t down_stride=size_t(hidden/256)*110;
        jobs.clear();
        for(int i=0;i<hits;++i) {
            jobs.push_back({weights[i].up,up_stride*hidden,up_stride,0,latent,hidden,1.f,0});
            jobs.push_back({weights[i].gate,up_stride*hidden,up_stride,0,latent,hidden,1.f,0});
        }
        if (chain) {
            const size_t hidden_bytes=size_t(hidden/256)*292;
            down_jobs.clear();
            for(int i=0;i<hits;++i)
                down_jobs.push_back({weights[i].down,down_stride*latent,down_stride,i*hidden_bytes,hidden,latent,1.f,0});
            if (!chain_api->submit(chain,jobs.data(),down_jobs.data(),hits,q8,bytes,beta,linear) || !api->wait(gpu))
                throw std::runtime_error(api->error(gpu));
            for(int i=0;i<hits;++i)std::memcpy(down+size_t(i)*latent,result(i,latent),latent*sizeof(float));
            return;
        }
        submit(q8,bytes);
        const size_t hidden_bytes=size_t(hidden/256)*292;
        for(int i=0;i<hits;++i) {
            const float * up=result(2*i,hidden),* gate=result(2*i+1,hidden);
            for(int row=0;row<hidden;++row) {
                float u=up[row],g=gate[row];
                // The expression and compiler flags match the CPU sink. This
                // bridge is part of the algorithm, never a CUDA-error fallback.
                const float a=(beta*std::tanh(g*(1.0f/beta)))*(1.0f/(1.0f+std::exp(-g)));
                if(linear>0.f)u=linear*std::tanh(u*(1.0f/linear));
                activation[row]=a*u;
            }
            quantize(activation.data(),quant_hidden.data()+i*hidden_bytes,hidden);
        }
        jobs.clear();
        for(int i=0;i<hits;++i)
            jobs.push_back({weights[i].down,down_stride*latent,down_stride,i*hidden_bytes,hidden,latent,1.f,0});
        submit(quant_hidden.data(),hits*hidden_bytes);
        for(int i=0;i<hits;++i)std::memcpy(down+size_t(i)*latent,result(i,latent),latent*sizeof(float));
    }
};
void * create(int device,int latent,int hidden,int n,char * error,size_t size) {
    try{return new bridge(device,latent,hidden,n);}
    catch(const std::exception & e){if(size)std::snprintf(error,size,"%s",e.what());return nullptr;}
}
void destroy(void * p){delete static_cast<bridge *>(p);}
int compute(void * p,const vram_compute_expert * weights,int hits,const void * q8,size_t bytes,
            float beta,float linear,float * down,char * error,size_t size) {
    try{static_cast<bridge *>(p)->compute(weights,hits,q8,bytes,beta,linear,down);return 0;}
    catch(const std::exception & e){
        if(size)std::snprintf(error,size,"%s",e.what());
        // A failed submit may already have queued work. wait rejects a failed
        // test2 handle; destroy is its unconditional stream-drain boundary.
        // Keep cache-owned weights pinned until both handles have closed.
        static_cast<bridge *>(p)->close();
        return 1;
    }
}
size_t device_bytes(void * p){auto * b=static_cast<bridge *>(p);return b->api->device_bytes(b->gpu)+(b->chain?b->chain_api->device_bytes(b->chain):0);}
size_t host_bytes(void * p){auto * b=static_cast<bridge *>(p);return b->api->host_bytes(b->gpu)+(b->chain?b->chain_api->host_bytes(b->chain):0)+b->activation.capacity()*sizeof(float)+b->quant_hidden.capacity()+(b->jobs.capacity()+b->down_jobs.capacity())*sizeof(kimi_q3k_gpu_job);}
}
extern "C" const vram_compute_api * kimi_vram_compute_get_api(){
    static const vram_compute_api api{1,sizeof(vram_compute_api),create,destroy,compute,device_bytes,host_bytes};return &api;
}
