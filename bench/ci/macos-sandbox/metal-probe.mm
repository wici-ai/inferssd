#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
int main() {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice(); if (!d) return 10;
        NSError * e = nil;
        NSString * source = @"#include <metal_stdlib>\nusing namespace metal; kernel void probe(device uint *p [[buffer(0)]], uint i [[thread_position_in_grid]]) {p[i]=i*7+3;}";
        id<MTLLibrary> l = [d newLibraryWithSource:source options:nil error:&e]; if (!l) return 11;
        id<MTLFunction> f = [l newFunctionWithName:@"probe"];
        id<MTLComputePipelineState> p = [d newComputePipelineStateWithFunction:f error:&e]; if (!p) return 12;
        id<MTLBuffer> b = [d newBufferWithLength:256*sizeof(unsigned) options:MTLResourceStorageModeShared]; if (!b) return 13;
        id<MTLCommandQueue> q = [d newCommandQueue]; id<MTLCommandBuffer> c = [q commandBuffer];
        id<MTLComputeCommandEncoder> x = [c computeCommandEncoder]; [x setComputePipelineState:p]; [x setBuffer:b offset:0 atIndex:0];
        [x dispatchThreads:MTLSizeMake(256,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)]; [x endEncoding]; [c commit]; [c waitUntilCompleted];
        if (c.status != MTLCommandBufferStatusCompleted) return 14;
        unsigned * out = (unsigned *)b.contents; for (unsigned i=0;i<256;++i) if(out[i]!=i*7+3)return 15;
        std::printf("METAL_SANDBOX_PASS values=256 device=%s\n",d.name.UTF8String);
    }
    return 0;
}
