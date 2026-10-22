// Standalone extension DSO: compile the unchanged accepted dot-product source
// and this optional chain together. Do not link two parity DSOs into one run.
#include "kimi-q3k-parity.cu"
#include "ggml-kimi-q3k-chain.h"
#include "kimi-q3k-reference-math.cuh"
namespace {
__global__ void chain_situ(const float * ug,float * hidden,int pairs,float beta,float linear) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=pairs*3072)return;
 int p=i/3072,row=i%3072;
 hidden[i]=reference_situ(ug[(2*p)*3072+row],ug[(2*p+1)*3072+row],beta,linear);
}
__global__ void chain_quant(const float * x,block_q8_K * y) {
 const int b=blockIdx.x,j=threadIdx.x;const float * v=x+b*256;
 __shared__ float iscale;__shared__ int zero;
 if(j==0){float max=0,amax=0;for(int k=0;k<256;++k){float a=fabsf(v[k]);if(a>amax){amax=a;max=v[k];}}
  zero=amax==0;iscale=zero?0:__fdiv_rn(-127.0f,max);y[b].d=zero?0:__fdiv_rn(1.0f,iscale);}
 __syncthreads();
 if(zero){y[b].qs[j]=0;return;} // Like the CPU, bsums are untouched for zero.
 int q=int(__float_as_uint(__fadd_rn(__fmul_rn(iscale,v[j]),12582912.0f))&0x7fffff)-0x400000;
 y[b].qs[j]=min(127,q);__syncthreads();
 if(j<16){int sum=0;for(int k=0;k<16;++k)sum+=y[b].qs[j*16+k];y[b].bsums[j]=sum;}
}
}
struct kimi_q3k_gpu_chain {
 kimi_q3k_gpu * base=nullptr;
 device_job * host_up=nullptr,*device_up=nullptr;
 float * hidden=nullptr;
 size_t jobs_bytes=0,hidden_bytes=0;
};
namespace {
int chain_destroy(kimi_q3k_gpu_chain * h){
 if(!h)return 1;auto c=h->base;device_scope scope(c->config.device);
 bool ok=scope.status==cudaSuccess;
 if(ok){
  if(c->stream)ok=(cudaStreamSynchronize(c->stream)==cudaSuccess)&&ok;
  if(h->host_up)ok=(cudaFreeHost(h->host_up)==cudaSuccess)&&ok;
  if(h->device_up)ok=(cudaFree(h->device_up)==cudaSuccess)&&ok;
  if(h->hidden)ok=(cudaFree(h->hidden)==cudaSuccess)&&ok;
 }
 delete h;return ok;
}
kimi_q3k_gpu_chain * chain_create(kimi_q3k_gpu * c){
 if(!c||c->busy||c->failed)return nullptr;
 if(c->config.max_columns<3584||c->config.max_rows<3584||c->config.max_jobs<2){c->invalid("chain requires K3 capacities");return nullptr;}
 device_scope scope(c->config.device);if(!c->check(scope.status,"chain device"))return nullptr;
 auto h=new(std::nothrow) kimi_q3k_gpu_chain;if(!h){c->invalid("chain host allocation");return nullptr;}
 h->base=c;h->jobs_bytes=c->config.max_jobs*sizeof(device_job);h->hidden_bytes=(c->config.max_jobs/2)*3072*sizeof(float);
 if(!c->check(cudaMallocHost(&h->host_up,h->jobs_bytes),"chain host descriptors")||
    !c->check(cudaMalloc(&h->device_up,h->jobs_bytes),"chain device descriptors")||
    !c->check(cudaMalloc(&h->hidden,h->hidden_bytes),"chain hidden")){chain_destroy(h);return nullptr;}
 return h;
}
int chain_submit(kimi_q3k_gpu_chain * h,const kimi_q3k_gpu_job * up,const kimi_q3k_gpu_job * down,
                 int pairs,const void * q8,size_t bytes,float beta,float linear){
 if(!h)return 0;auto c=h->base;if(c->failed)return 0;
 if(c->busy)return c->invalid("chain handle is busy");
 if(!up||!down||!q8||pairs<1||pairs*2>c->config.max_jobs||bytes>c->q8_bytes||
    !std::isfinite(beta)||beta<=0||!std::isfinite(linear))return c->invalid("chain shape/SiTU parameters");
 device_scope scope(c->config.device);if(!c->check(scope.status,"chain submit device"))return 0;
 // Validate every source before launching or publishing a new result layout.
 for(int i=0;i<3*pairs;++i){
  bool is_down=i>=2*pairs;const auto & j=is_down?down[i-2*pairs]:up[i];
  const int columns=is_down?3072:3584,rows=is_down?3584:3072;
  const size_t stride=is_down?1320:1540, qbytes=is_down?3504:4088;
  const size_t capacity=is_down?size_t(pairs)*3504:bytes;
  if(!j.weights||j.columns!=columns||j.rows!=rows||j.row_stride!=stride||j.weight_bytes<4730880||
     j.apply_scale||j.q8_offset%4||j.q8_offset>capacity||qbytes>capacity-j.q8_offset)
   return c->invalid("unsupported chain job layout");
  if(is_down&&j.q8_offset!=size_t(i-2*pairs)*3504)return c->invalid("chain down activation order");
  cudaPointerAttributes attr{};if(!c->check(cudaPointerGetAttributes(&attr,j.weights),"chain weight pointer"))return 0;
  if(attr.type!=cudaMemoryTypeDevice||attr.device!=c->config.device)return c->invalid("chain weights on wrong device");
 }
 for(int i=0;i<2*pairs;++i){const auto & j=up[i];h->host_up[i]={static_cast<const char *>(j.weights),j.row_stride,j.q8_offset,size_t(i)*3072,j.columns,j.rows,1.0f,0};}
 for(int i=0;i<pairs;++i){const auto & j=down[i];c->h_jobs[i]={static_cast<const char *>(j.weights),j.row_stride,j.q8_offset,size_t(i)*3584,j.columns,j.rows,1.0f,0};}
 c->njobs=pairs;std::memcpy(c->h_q8,q8,bytes);
 if(!c->check(cudaMemcpyAsync(h->device_up,h->host_up,2*pairs*sizeof(device_job),cudaMemcpyHostToDevice,c->stream),"chain up jobs H2D")||
    !c->check(cudaMemcpyAsync(c->d_jobs,c->h_jobs,pairs*sizeof(device_job),cudaMemcpyHostToDevice,c->stream),"chain down jobs H2D")||
    !c->check(cudaMemcpyAsync(c->d_q8,c->h_q8,bytes,cudaMemcpyHostToDevice,c->stream),"chain input H2D"))return 0;
 q3k_avx2<<<dim3(3072/16,2*pairs),128,0,c->stream>>>(h->device_up,c->d_q8,c->d_output);
 if(!c->check(cudaGetLastError(),"chain up/gate"))return 0;
 chain_situ<<<(pairs*3072+127)/128,128,0,c->stream>>>(c->d_output,h->hidden,pairs,beta,linear);
 if(!c->check(cudaGetLastError(),"chain SiTU"))return 0;
 chain_quant<<<pairs*12,256,0,c->stream>>>(h->hidden,reinterpret_cast<block_q8_K *>(c->d_q8));
 if(!c->check(cudaGetLastError(),"chain Q8_K"))return 0;
 q3k_avx2<<<dim3(3584/16,pairs),128,0,c->stream>>>(c->d_jobs,c->d_q8,c->d_output);
 if(!c->check(cudaGetLastError(),"chain down")||
    !c->check(cudaMemcpyAsync(c->h_output,c->d_output,size_t(pairs)*3584*sizeof(float),cudaMemcpyDeviceToHost,c->stream),"chain final D2H")||
    !c->check(cudaEventRecord(c->event,c->stream),"chain completion"))return 0;
 c->busy=true;return 1;
}
size_t chain_host_bytes(kimi_q3k_gpu_chain * h){return h?h->jobs_bytes:0;}
size_t chain_device_bytes(kimi_q3k_gpu_chain * h){return h?h->jobs_bytes+h->hidden_bytes:0;}
const kimi_q3k_gpu_chain_api_v1 chain_api={1,sizeof(kimi_q3k_gpu_chain_api_v1),chain_create,chain_submit,chain_destroy,chain_host_bytes,chain_device_bytes};
}
extern "C" KIMI_Q3K_EXPORT const kimi_q3k_gpu_chain_api_v1 * kimi_q3k_gpu_get_chain_api_v1(){return &chain_api;}
